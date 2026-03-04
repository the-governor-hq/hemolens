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
from sklearn.model_selection import GroupShuffleSplit

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
                "session": row["MEASUREMENT_DATE"],
            })

    crop_df = pd.DataFrame(records)
    print(f"Cropped {len(crop_df)} nail ROIs from {crop_df['patient_id'].nunique()} patients")
    if skipped:
        print(f"  (skipped {skipped} missing/unreadable images)")

    # ------------------------------------------------------------------
    # 2. Session-aware patient-level splits (prevent data leakage)
    #    Patients from the same MEASUREMENT_DATE share lighting/camera
    #    conditions — they MUST stay in the same split.
    # ------------------------------------------------------------------
    patients = crop_df[["patient_id", "hb_value", "session"]].drop_duplicates(subset="patient_id")

    # Use MEASUREMENT_DATE as the group key so same-session patients
    # never leak across train/val/test boundaries.
    groups = patients["session"].values

    # First split: train vs (val+test)
    holdout_size = val_size + test_size
    gss1 = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=RANDOM_STATE)
    train_idx, holdout_idx = next(gss1.split(patients, groups=groups))
    train_pids = patients.iloc[train_idx]["patient_id"]
    holdout_patients = patients.iloc[holdout_idx]

    # Second split: val vs test
    holdout_groups = holdout_patients["session"].values
    relative_test = test_size / holdout_size
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss2.split(holdout_patients, groups=holdout_groups))
    val_pids = holdout_patients.iloc[val_idx]["patient_id"]
    test_pids = holdout_patients.iloc[test_idx]["patient_id"]

    # Assign splits
    train_set, val_set, test_set = set(train_pids), set(val_pids), set(test_pids)
    def assign_split(pid):
        if pid in train_set:
            return "train"
        elif pid in val_set:
            return "val"
        else:
            return "test"

    crop_df["split"] = crop_df["patient_id"].apply(assign_split)

    # Verify no session leaks across splits
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        sessions_1 = set(crop_df[crop_df["split"] == s1]["session"])
        sessions_2 = set(crop_df[crop_df["split"] == s2]["session"])
        leaked = sessions_1 & sessions_2
        if leaked:
            print(f"  WARNING: {len(leaked)} sessions leak between {s1} and {s2}!")
        else:
            print(f"  ✓ No session leakage between {s1}/{s2}")

    # ------------------------------------------------------------------
    # 3. Save metadata CSV
    # ------------------------------------------------------------------
    output_csv = OUTPUT_DIR / "metadata_splits.csv"
    crop_df.to_csv(output_csv, index=False)

    # Summary
    n_sessions = crop_df["session"].nunique()
    print(f"\nSplit summary (session-aware, {n_sessions} unique sessions):")
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
