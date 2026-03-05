"""
HemoLens — Color Feature Extraction (v2)

Extracts handcrafted color features from nail-bed and skin ROIs.
Adds P5/P95 percentile features and ratio features on top of the
original mean/std/median from the research notebook.

Produces: ../data/processed/color_features.csv

Usage:
    python extract_color_features.py
"""

import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW_DIR = Path("../data/raw")
PHOTO_DIR = RAW_DIR / "photo"
METADATA_CSV = RAW_DIR / "metadata.csv"
OUTPUT_DIR = Path("../data/processed")


def extract_color_features(roi: np.ndarray, prefix: str = "") -> dict:
    """
    Extract color statistics from a single ROI.

    v2 additions: P5/P95 percentiles for RGB channels (captures distribution tails).

    Returns dict with ~27 features per ROI (was 21 in v1).
    """
    features = {}

    # --- RGB: 3 channels × (mean, std, median, p5, p95) = 15 ---
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    for i, ch in enumerate(["R", "G", "B"]):
        vals = rgb[:, :, i].astype(float)
        features[f"{prefix}rgb_{ch}_mean"] = vals.mean()
        features[f"{prefix}rgb_{ch}_std"] = vals.std()
        features[f"{prefix}rgb_{ch}_median"] = np.median(vals)
        features[f"{prefix}rgb_{ch}_p5"] = np.percentile(vals, 5)
        features[f"{prefix}rgb_{ch}_p95"] = np.percentile(vals, 95)

    # --- LAB: 3 channels × (mean, std) = 6 ---
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    for i, ch in enumerate(["L", "A", "B_lab"]):
        vals = lab[:, :, i].astype(float)
        features[f"{prefix}lab_{ch}_mean"] = vals.mean()
        features[f"{prefix}lab_{ch}_std"] = vals.std()

    # --- HSV: 3 channels × (mean, std) = 6 ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    for i, ch in enumerate(["H", "S", "V"]):
        vals = hsv[:, :, i].astype(float)
        features[f"{prefix}hsv_{ch}_mean"] = vals.mean()
        features[f"{prefix}hsv_{ch}_std"] = vals.std()

    return features  # 27 features per ROI


def extract_sample_features(image: np.ndarray, nail_bboxes: list, skin_bboxes: list) -> dict:
    """
    Extract features for one sample: nail ROIs + skin ROIs + contrast + ratios.
    """
    # Average features across 3 nail crops
    nail_feats_list = []
    for bbox in nail_bboxes:
        ymin, xmin, ymax, xmax = bbox
        h, w = image.shape[:2]
        ymin, xmin = max(0, ymin), max(0, xmin)
        ymax, xmax = min(h, ymax), min(w, xmax)
        roi = image[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            continue
        nail_feats_list.append(extract_color_features(roi, prefix="nail_"))

    skin_feats_list = []
    for bbox in skin_bboxes:
        ymin, xmin, ymax, xmax = bbox
        h, w = image.shape[:2]
        ymin, xmin = max(0, ymin), max(0, xmin)
        ymax, xmax = min(h, ymax), min(w, xmax)
        roi = image[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            continue
        skin_feats_list.append(extract_color_features(roi, prefix="skin_"))

    if not nail_feats_list or not skin_feats_list:
        return {}

    # Average across ROIs
    features = {}
    for key in nail_feats_list[0]:
        features[key] = np.mean([f[key] for f in nail_feats_list])
    for key in skin_feats_list[0]:
        features[key] = np.mean([f[key] for f in skin_feats_list])

    # Contrast features: nail_mean − skin_mean
    nail_mean_keys = [k for k in features if k.startswith("nail_") and "_mean" in k]
    for nk in nail_mean_keys:
        sk = nk.replace("nail_", "skin_")
        if sk in features:
            contrast_key = nk.replace("nail_", "contrast_")
            features[contrast_key] = features[nk] - features[sk]

    # Ratio features: nail / (skin + eps) — partially corrects for illumination
    # (not truly invariant: specular reflections, surface normals break the assumption)
    for ch in ["R", "G", "B"]:
        nail_key = f"nail_rgb_{ch}_mean"
        skin_key = f"skin_rgb_{ch}_mean"
        if nail_key in features and skin_key in features:
            features[f"ratio_rgb_{ch}"] = features[nail_key] / (features[skin_key] + 1e-6)

    # Redness index: a* channel ratio (clinically relevant for Hb)
    nail_a = features.get("nail_lab_A_mean", 0)
    skin_a = features.get("skin_lab_A_mean", 0)
    features["redness_index"] = nail_a / (skin_a + 1e-6)

    return features


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_CSV)
    df["NAIL_BOUNDING_BOXES"] = df["NAIL_BOUNDING_BOXES"].apply(ast.literal_eval)
    df["SKIN_BOUNDING_BOXES"] = df["SKIN_BOUNDING_BOXES"].apply(ast.literal_eval)
    df["hb_gdL"] = df["HB_LEVEL_GperL"] / 10.0

    records = []
    skipped = 0

    for _, row in df.iterrows():
        pid = row["PATIENT_ID"]
        img_path = PHOTO_DIR / f"{pid}.jpg"
        if not img_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        feats = extract_sample_features(img, row["NAIL_BOUNDING_BOXES"], row["SKIN_BOUNDING_BOXES"])
        if not feats:
            skipped += 1
            continue

        feats["PATIENT_ID"] = pid
        feats["hb_gdL"] = row["hb_gdL"]
        feats["HB_LEVEL_GperL"] = row["HB_LEVEL_GperL"]
        records.append(feats)

    features_df = pd.DataFrame(records)
    n_feat = len([c for c in features_df.columns if c not in ("PATIENT_ID", "hb_gdL", "HB_LEVEL_GperL")])
    print(f"Extracted {n_feat} color features for {len(features_df)} patients")
    if skipped:
        print(f"  (skipped {skipped} missing/unreadable images)")

    output_path = OUTPUT_DIR / "color_features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
