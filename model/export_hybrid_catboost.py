"""
HemoLens — Export CatBoost Head via Two-Stage ONNX

Creates TWO ONNX models for chained web inference:
  1. hemolens_backbone.onnx          image[1,3,224,224] → features[1,1280]
  2. hemolens_catboost_head.onnx     combined[N,1347]   → predictions[N]

Web inference chain:
  backbone(image)     → cnn_features  (1280-d)
  backbone(image_flip)→ cnn_features' (1280-d)       (TTA)
  avg_features = (cnn_features + cnn_features') / 2
  concat(avg_features, color_features)  → combined (1347-d)
  catboost_head(combined)               → Hb prediction

This achieves CatBoost-level accuracy (test MAE ≈ 1.305 vs Ridge 1.459).

Usage:
    python export_hybrid_catboost.py --config configs/mobilenet_edge.yaml
"""

import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

# Avoid cp1252 encoding errors from torch.onnx checkmark emoji on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("Install timm: pip install timm")

from transforms import get_val_transforms

warnings.filterwarnings("ignore", category=UserWarning)


# ── Canonical colour-feature ordering (must match JS) ──────────────────────

COLOR_FEATURE_NAMES = [
    # Nail ROI — RGB (15)
    "nail_rgb_R_mean", "nail_rgb_R_std", "nail_rgb_R_median", "nail_rgb_R_p5", "nail_rgb_R_p95",
    "nail_rgb_G_mean", "nail_rgb_G_std", "nail_rgb_G_median", "nail_rgb_G_p5", "nail_rgb_G_p95",
    "nail_rgb_B_mean", "nail_rgb_B_std", "nail_rgb_B_median", "nail_rgb_B_p5", "nail_rgb_B_p95",
    # Nail ROI — LAB (6)
    "nail_lab_L_mean", "nail_lab_L_std",
    "nail_lab_A_mean", "nail_lab_A_std",
    "nail_lab_B_lab_mean", "nail_lab_B_lab_std",
    # Nail ROI — HSV (6)
    "nail_hsv_H_mean", "nail_hsv_H_std",
    "nail_hsv_S_mean", "nail_hsv_S_std",
    "nail_hsv_V_mean", "nail_hsv_V_std",
    # Skin ROI — RGB (15)
    "skin_rgb_R_mean", "skin_rgb_R_std", "skin_rgb_R_median", "skin_rgb_R_p5", "skin_rgb_R_p95",
    "skin_rgb_G_mean", "skin_rgb_G_std", "skin_rgb_G_median", "skin_rgb_G_p5", "skin_rgb_G_p95",
    "skin_rgb_B_mean", "skin_rgb_B_std", "skin_rgb_B_median", "skin_rgb_B_p5", "skin_rgb_B_p95",
    # Skin ROI — LAB (6)
    "skin_lab_L_mean", "skin_lab_L_std",
    "skin_lab_A_mean", "skin_lab_A_std",
    "skin_lab_B_lab_mean", "skin_lab_B_lab_std",
    # Skin ROI — HSV (6)
    "skin_hsv_H_mean", "skin_hsv_H_std",
    "skin_hsv_S_mean", "skin_hsv_S_std",
    "skin_hsv_V_mean", "skin_hsv_V_std",
    # Cross-ROI contrast (9)
    "contrast_rgb_R_mean", "contrast_rgb_G_mean", "contrast_rgb_B_mean",
    "contrast_lab_L_mean", "contrast_lab_A_mean", "contrast_lab_B_lab_mean",
    "contrast_hsv_H_mean", "contrast_hsv_S_mean", "contrast_hsv_V_mean",
    # Cross-ROI ratios (3)
    "ratio_rgb_R", "ratio_rgb_G", "ratio_rgb_B",
    # Redness index (1)
    "redness_index",
]

N_COLOR = len(COLOR_FEATURE_NAMES)  # 67


# ── Backbone-only wrapper for ONNX export ─────────────────────────────────

class BackboneOnly(nn.Module):
    """Thin wrapper: image → frozen backbone → feature vector."""

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True,
                                          num_classes=0)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


# ── CNN feature extraction (reused from train_hybrid.py) ──────────────────

def extract_cnn_features(
    backbone_name: str, data_root: Path, metadata_csv: Path,
    input_size: int, val_cfg: dict, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract frozen backbone features for every crop (with TTA flip)."""
    print(f"\n{'='*60}")
    print(f"Extracting frozen {backbone_name} features (TTA)")
    print(f"{'='*60}")

    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    backbone.eval().to(device)
    tf = get_val_transforms(input_size, val_cfg)
    df = pd.read_csv(metadata_csv)
    print(f"Loaded {len(df)} crops from {df['patient_id'].nunique()} patients")

    features_list, hb_list, pid_list, split_list, session_list = [], [], [], [], []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CNN features"):
            img_path = data_root / row["image_path"]
            img = Image.open(img_path).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(device)
            feat = backbone(tensor).squeeze(0).cpu().numpy()

            # TTA: average with horizontally-flipped image
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            tensor_flip = tf(flipped).unsqueeze(0).to(device)
            feat_flip = backbone(tensor_flip).squeeze(0).cpu().numpy()
            feat = (feat + feat_flip) / 2.0

            features_list.append(feat)
            hb_list.append(row["hb_value"])
            pid_list.append(row["patient_id"])
            split_list.append(row["split"])
            session_list.append(row.get("session", row["patient_id"]))

    features = np.stack(features_list)
    print(f"CNN features shape: {features.shape}")
    return (
        features,
        np.array(hb_list, dtype=np.float32),
        np.array(pid_list),
        np.array(split_list),
        np.array(session_list),
    )


# ── Build patient-level combined feature matrix ───────────────────────────

def build_patient_features(cnn_features, hb_values, patient_ids, splits,
                           sessions, color_csv: Path):
    """Average CNN per patient, merge with colour features, split."""
    unique_pids = np.unique(patient_ids)
    print(f"\nAggregating {len(cnn_features)} crops -> {len(unique_pids)} patients")

    patient_cnn, patient_hb, patient_split = {}, {}, {}
    for pid in unique_pids:
        m = patient_ids == pid
        patient_cnn[pid] = cnn_features[m].mean(axis=0)
        patient_hb[pid] = hb_values[m][0]
        patient_split[pid] = splits[m][0]

    ordered = sorted(unique_pids)
    X_cnn = np.stack([patient_cnn[p] for p in ordered])
    y = np.array([patient_hb[p] for p in ordered])
    split_arr = np.array([patient_split[p] for p in ordered])

    # Load and align colour features
    color_df = pd.read_csv(color_csv)
    feat_cols = [c for c in color_df.columns
                 if c not in ("PATIENT_ID", "hb_gdL", "HB_LEVEL_GperL")]
    pid_to_color = {
        int(row["PATIENT_ID"]): row[feat_cols].values.astype(np.float64)
        for _, row in color_df.iterrows()
    }
    X_color = np.stack([pid_to_color.get(int(p), np.zeros(len(feat_cols)))
                        for p in ordered])

    # Reorder colour columns to canonical ordering
    col_idx = [feat_cols.index(name) for name in COLOR_FEATURE_NAMES]
    X_color = X_color[:, col_idx]

    X = np.hstack([X_cnn, X_color])
    cnn_dim = X_cnn.shape[1]
    print(f"  CNN: {X_cnn.shape}  Color: {X_color.shape}  Combined: {X.shape}")

    data = {}
    for split in ("train", "val", "test"):
        m = split_arr == split
        if m.sum() > 0:
            data[split] = {"X": X[m], "y": y[m]}
            print(f"  {split}: {m.sum()} patients")
    return data, cnn_dim


# ── WHO severity breakdown ────────────────────────────────────────────────

_WHO_BINS = [
    ("Severe",   0.0,  8.0),
    ("Moderate", 8.0, 11.0),
    ("Mild",    11.0, 13.0),
    ("Normal",  13.0, 99.0),
]


def _severity_breakdown(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error as mae_fn
    report = {}
    for label, lo, hi in _WHO_BINS:
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        report[label] = {
            "n": n,
            "mae": float(mae_fn(y_true[mask], y_pred[mask])) if n > 0 else float("nan"),
        }
    return report


# ── Train CatBoost + export both ONNX models ─────────────────────────────

def train_and_export(data: dict, cnn_dim: int, backbone_name: str,
                     save_dir: Path):
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val     = data["val"]["X"],   data["val"]["y"]
    X_test, y_test   = data["test"]["X"],  data["test"]["y"]

    total_dim = X_train.shape[1]
    color_dim = total_dim - cnn_dim
    assert color_dim == N_COLOR, f"Expected {N_COLOR} colour features, got {color_dim}"

    print(f"\n{'='*60}")
    print(f"Training CatBoost on CNN({cnn_dim}) + Color({N_COLOR}) = {total_dim}")
    print(f"{'='*60}")

    # ── 1. Train CatBoost (same hyper-params as train_hybrid.py) ──
    cb = CatBoostRegressor(
        iterations=500, depth=4, learning_rate=0.03,
        l2_leaf_reg=10, random_seed=42, verbose=50,
    )
    cb.fit(X_train.astype(np.float32), y_train.astype(np.float32))

    for name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        pred = cb.predict(X.astype(np.float32))
        mae = mean_absolute_error(y, pred)
        r2  = r2_score(y, pred)
        print(f"  {name}: MAE={mae:.3f}  R2={r2:.4f}")
        if name == "Test":
            sev = _severity_breakdown(y, pred)
            for cls, m in sev.items():
                print(f"    {cls:8s}: n={m['n']:3d}, MAE={m['mae']:.3f}")

    # ── 2. Export CatBoost head as ONNX ──
    cb_onnx_path = save_dir / "hemolens_catboost_head.onnx"
    cb.save_model(
        str(cb_onnx_path),
        format="onnx",
        export_parameters={
            "onnx_domain": "ai.onnx.ml",
            "onnx_model_version": 1,
        },
    )
    cb_size = os.path.getsize(cb_onnx_path) / 1024
    print(f"\n  CatBoost head ONNX -> {cb_onnx_path}  ({cb_size:.0f} KB)")

    # Verify round-trip with ORT
    import onnxruntime as ort
    cb_sess = ort.InferenceSession(str(cb_onnx_path),
                                   providers=["CPUExecutionProvider"])
    cb_in_name = cb_sess.get_inputs()[0].name
    cb_out_name = cb_sess.get_outputs()[0].name
    print(f"  CatBoost ONNX input: '{cb_in_name}'  output: '{cb_out_name}'")

    ort_pred = cb_sess.run(None, {cb_in_name: X_test[:5].astype(np.float32)})
    cb_pred  = cb.predict(X_test[:5].astype(np.float32))
    delta = np.abs(ort_pred[0].flatten() - cb_pred).max()
    print(f"  ORT vs CatBoost max delta: {delta:.6f}")

    # ── 3. Export backbone-only ONNX ──
    print(f"\n  Exporting backbone-only ONNX...")
    backbone_model = BackboneOnly(backbone_name)
    backbone_model.eval()

    dummy_img = torch.randn(1, 3, 224, 224)
    backbone_onnx_path = save_dir / "hemolens_backbone.onnx"

    torch.onnx.export(
        backbone_model,
        (dummy_img,),
        str(backbone_onnx_path),
        input_names=["image"],
        output_names=["features"],
        opset_version=13,
        dynamic_axes={
            "image":    {0: "batch"},
            "features": {0: "batch"},
        },
    )
    bb_size = os.path.getsize(backbone_onnx_path) / 1024 / 1024
    print(f"  Backbone ONNX -> {backbone_onnx_path}  ({bb_size:.1f} MB)")

    # Verify backbone produces correct dim
    bb_sess = ort.InferenceSession(str(backbone_onnx_path),
                                   providers=["CPUExecutionProvider"])
    bb_out = bb_sess.run(None, {"image": dummy_img.numpy()})
    print(f"  Backbone output shape: {bb_out[0].shape}  "
          f"(expected [1, {cnn_dim}])")

    # ── 4. Copy to web-demo/model/ ──
    web_dir = save_dir.parent / ".." / "web-demo" / "model"
    web_dir.mkdir(parents=True, exist_ok=True)

    for src in [backbone_onnx_path, cb_onnx_path]:
        dst = web_dir / src.name
        shutil.copy2(src, dst)
        print(f"  Copied -> {dst}")

    # ── 5. Save manifest ──
    test_pred = cb.predict(X_test.astype(np.float32))
    manifest = {
        "type": "catboost_two_stage",
        "backbone_onnx": "hemolens_backbone.onnx",
        "catboost_onnx": "hemolens_catboost_head.onnx",
        "catboost_input_name": cb_in_name,
        "catboost_output_name": cb_out_name,
        "cnn_dim": cnn_dim,
        "color_dim": color_dim,
        "total_dim": total_dim,
        "catboost_params": {
            "iterations": 500, "depth": 4, "learning_rate": 0.03,
            "l2_leaf_reg": 10,
        },
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "color_feature_names": COLOR_FEATURE_NAMES,
    }
    manifest_path = save_dir / "catboost_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest -> {manifest_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export CatBoost two-stage ONNX for HemoLens web demo")
    parser.add_argument("--config", default="configs/mobilenet_edge.yaml")
    parser.add_argument("--color-features",
                        default="../data/processed/color_features.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = (torch.device("cuda")
              if args.device == "auto" and torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    data_root = Path(cfg["data"]["root"])
    metadata_csv = Path(cfg["data"]["metadata_csv"])
    color_csv = Path(args.color_features)
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: extract CNN features
    cnn_feats, hb, pids, splits, sessions = extract_cnn_features(
        cfg["model"]["backbone"], data_root, metadata_csv,
        cfg["model"]["input_size"], cfg["augmentation"]["val"], device,
    )

    # Step 2: build patient-level combined matrix
    data, cnn_dim = build_patient_features(
        cnn_feats, hb, pids, splits, sessions, color_csv,
    )

    # Step 3: train CatBoost + export both ONNX models
    train_and_export(data, cnn_dim, cfg["model"]["backbone"], save_dir)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
