"""
HemoLens — Prepare YOLO-format dataset for nail detection.

Converts the existing [ymin, xmin, ymax, xmax] bounding-box annotations
from raw/metadata.csv into YOLO txt format (class cx cy w h — normalized)
and creates train/val/test image symlinks + a data.yaml config.

The resulting dataset lives under data/nail_detection/ and can be consumed
directly by `ultralytics` (YOLOv8) for training.

Usage:
    python prepare_yolo_dataset.py
    python prepare_yolo_dataset.py --test-size 0.15 --val-size 0.15
"""

import argparse
import ast
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("../data/raw")
PHOTO_DIR = RAW_DIR / "photo"
METADATA_CSV = RAW_DIR / "metadata.csv"
OUTPUT_DIR = Path("../data/nail_detection")

RANDOM_STATE = 42

# Class IDs
CLASS_NAIL = 0
CLASS_SKIN = 1
CLASS_NAMES = ["nail", "skin"]


def bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    """
    Convert [ymin, xmin, ymax, xmax] (absolute pixels) to YOLO format
    (cx, cy, w, h) normalized to [0,1].
    """
    ymin, xmin, ymax, xmax = bbox
    # Clamp
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img_w, xmax), min(img_h, ymax)

    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h

    return cx, cy, bw, bh


def main(val_size: float = 0.15, test_size: float = 0.15):
    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # Load metadata
    df = pd.read_csv(METADATA_CSV)
    df["NAIL_BOUNDING_BOXES"] = df["NAIL_BOUNDING_BOXES"].apply(ast.literal_eval)
    df["SKIN_BOUNDING_BOXES"] = df["SKIN_BOUNDING_BOXES"].apply(ast.literal_eval)

    # ── Split patients by session (same as Hb model) ────────────────────
    patients = df[["PATIENT_ID", "MEASUREMENT_DATE"]].drop_duplicates(subset="PATIENT_ID")
    groups = patients["MEASUREMENT_DATE"].values

    holdout_size = val_size + test_size
    gss1 = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=RANDOM_STATE)
    train_idx, holdout_idx = next(gss1.split(patients, groups=groups))
    train_pids = set(patients.iloc[train_idx]["PATIENT_ID"])
    holdout_patients = patients.iloc[holdout_idx]

    holdout_groups = holdout_patients["MEASUREMENT_DATE"].values
    relative_test = test_size / holdout_size
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss2.split(holdout_patients, groups=holdout_groups))
    val_pids = set(holdout_patients.iloc[val_idx]["PATIENT_ID"])
    test_pids = set(holdout_patients.iloc[test_idx]["PATIENT_ID"])

    def get_split(pid):
        if pid in train_pids:
            return "train"
        elif pid in val_pids:
            return "val"
        else:
            return "test"

    # ── Create YOLO directory structure ──────────────────────────────────
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0}
    box_count = 0

    for _, row in df.iterrows():
        pid = row["PATIENT_ID"]
        src_img = PHOTO_DIR / f"{pid}.jpg"
        if not src_img.exists():
            continue

        split = get_split(pid)
        stats[split] += 1

        # Get image dimensions
        with Image.open(src_img) as img:
            img_w, img_h = img.size

        # Copy image to YOLO dir
        dst_img = OUTPUT_DIR / "images" / split / f"{pid}.jpg"
        shutil.copy2(src_img, dst_img)

        # Write YOLO label file
        label_path = OUTPUT_DIR / "labels" / split / f"{pid}.txt"
        lines = []

        # Nail boxes
        for bbox in row["NAIL_BOUNDING_BOXES"]:
            cx, cy, bw, bh = bbox_to_yolo(bbox, img_w, img_h)
            lines.append(f"{CLASS_NAIL} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            box_count += 1

        # Skin boxes
        for bbox in row["SKIN_BOUNDING_BOXES"]:
            cx, cy, bw, bh = bbox_to_yolo(bbox, img_w, img_h)
            lines.append(f"{CLASS_SKIN} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            box_count += 1

        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Write data.yaml ─────────────────────────────────────────────────
    data_yaml = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("HemoLens — YOLO Nail Detection Dataset")
    print("=" * 60)
    print(f"Output:     {OUTPUT_DIR.resolve()}")
    print(f"Classes:    {CLASS_NAMES}")
    print(f"Total images: {sum(stats.values())}")
    print(f"Total boxes:  {box_count}")
    print()
    for split, n in stats.items():
        n_imgs = len(list((OUTPUT_DIR / "images" / split).glob("*.jpg")))
        n_labels = len(list((OUTPUT_DIR / "labels" / split).glob("*.txt")))
        print(f"  {split:5s}: {n_imgs:3d} images, {n_labels:3d} labels")
    print(f"\ndata.yaml -> {yaml_path}")
    print("Ready for: yolo detect train data=data.yaml model=yolov8n.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()
    main(val_size=args.val_size, test_size=args.test_size)
