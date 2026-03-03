"""
HemoLens — Dataset Preparation

Crop nail-bed ROIs from raw photos using metadata bounding boxes,
save as individual images, and create a metadata_splits.csv with
patient-level train/val/test splits.

This produces the image dataset consumed by `train.py`.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --val_size 0.15 --test_size 0.15
"""

import argparse
import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("../data/raw")
PHOTO_DIR = RAW_DIR / "photo"
METADATA_CSV = RAW_DIR / "metadata.csv"
OUTPUT_DIR = Path("../data/processed")
CROP_DIR = OUTPUT_DIR / "nail_crops"

RANDOM_STATE = 42


def crop_roi(image: np.ndarray, bbox: list) -> np.ndarray:
    """Crop [ymin, xmin, ymax, xmax] from image with bounds clamping."""
    ymin, xmin, ymax, xmax = bbox
    h, w = image.shape[:2]
    ymin, xmin = max(0, ymin), max(0, xmin)
    ymax, xmax = min(h, ymax), min(w, xmax)
    return image[ymin:ymax, xmin:xmax]


def prepare(val_size: float = 0.15, test_size: float = 0.15):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    df = pd.read_csv(METADATA_CSV)
    df["NAIL_BOUNDING_BOXES"] = df["NAIL_BOUNDING_BOXES"].apply(ast.literal_eval)
    df["hb_gdL"] = df["HB_LEVEL_GperL"] / 10.0

    # ------------------------------------------------------------------
    # 1. Crop all nail ROIs → individual images
    # ------------------------------------------------------------------
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

        for j, bbox in enumerate(row["NAIL_BOUNDING_BOXES"]):
            roi = crop_roi(img, bbox)
            if roi.size == 0:
                continue

            crop_name = f"{pid}_nail{j}.jpg"
            crop_path = CROP_DIR / crop_name
            cv2.imwrite(str(crop_path), roi)

            records.append({
                "patient_id": pid,
                "image_path": f"nail_crops/{crop_name}",
                "hb_value": row["hb_gdL"],
                "hb_gperL": row["HB_LEVEL_GperL"],
                "nail_index": j,
            })

    crop_df = pd.DataFrame(records)
    print(f"Cropped {len(crop_df)} nail ROIs from {crop_df['patient_id'].nunique()} patients")
    if skipped:
        print(f"  (skipped {skipped} missing/unreadable images)")

    # ------------------------------------------------------------------
    # 2. Patient-level stratified splits (prevent data leakage)
    # ------------------------------------------------------------------
    patients = crop_df[["patient_id", "hb_value"]].drop_duplicates()

    # Bin Hb for stratification (quartile-based)
    patients["hb_bin"] = pd.qcut(patients["hb_value"], q=4, labels=False, duplicates="drop")

    # First split: train vs (val+test)
    holdout_size = val_size + test_size
    train_pids, holdout_pids = train_test_split(
        patients["patient_id"],
        test_size=holdout_size,
        random_state=RANDOM_STATE,
        stratify=patients["hb_bin"],
    )

    # Second split: val vs test
    holdout_patients = patients[patients["patient_id"].isin(holdout_pids)]
    relative_test = test_size / holdout_size
    val_pids, test_pids = train_test_split(
        holdout_patients["patient_id"],
        test_size=relative_test,
        random_state=RANDOM_STATE,
        stratify=holdout_patients["hb_bin"],
    )

    # Assign splits
    def assign_split(pid):
        if pid in set(train_pids):
            return "train"
        elif pid in set(val_pids):
            return "val"
        else:
            return "test"

    crop_df["split"] = crop_df["patient_id"].apply(assign_split)

    # ------------------------------------------------------------------
    # 3. Save metadata CSV
    # ------------------------------------------------------------------
    output_csv = OUTPUT_DIR / "metadata_splits.csv"
    crop_df.to_csv(output_csv, index=False)

    # Summary
    print(f"\nSplit summary (patient-level, stratified):")
    for split in ["train", "val", "test"]:
        subset = crop_df[crop_df["split"] == split]
        n_patients = subset["patient_id"].nunique()
        n_images = len(subset)
        hb_range = f"[{subset['hb_value'].min():.1f}, {subset['hb_value'].max():.1f}]"
        print(f"  {split:5s}: {n_patients:3d} patients, {n_images:3d} images, Hb {hb_range} g/dL")

    print(f"\nSaved → {output_csv}")
    return crop_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HemoLens dataset preparation")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    args = parser.parse_args()
    prepare(val_size=args.val_size, test_size=args.test_size)
