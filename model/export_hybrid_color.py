"""
HemoLens — Export Dual-Input ONNX Model (CNN + Color Features)

Creates an ONNX model with two inputs:
  1. image:          [1, 3, 224, 224]  — nail photo (ImageNet-normalised)
  2. color_features: [1, N_COLOR]      — 67 handcrafted color statistics

This closes the performance gap between the offline CatBoost model (which
uses hand-crafted color features) and the deployed CNN-only Ridge model.

Architecture:
  frozen_backbone(image) → CNN features (1280-d)
  concat(CNN features, color_features) → linear head → Hb prediction

The StandardScaler normalisation is baked into the linear head weights,
so the JS client passes **raw** (un-normalised) color features.

Usage:
    python export_hybrid_color.py --config configs/mobilenet_edge.yaml
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("Install timm: pip install timm")

from transforms import get_val_transforms

warnings.filterwarnings("ignore", category=UserWarning)


# ── Feature name ordering (must match JS color_features.js) ────────────────

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


# ── Dual-input PyTorch module ──────────────────────────────────────────────

class HemoLensHybridColor(nn.Module):
    """
    Dual-input model: frozen backbone (CNN) + raw color features → Hb.

    The linear head's weights absorb the StandardScaler normalisation,
    so both inputs can be passed in their natural (un-scaled) form.
    """

    def __init__(self, backbone_name: str, cnn_dim: int, color_dim: int,
                 weights: np.ndarray, bias: float):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        total_dim = cnn_dim + color_dim
        self.head = nn.Linear(total_dim, 1)

        with torch.no_grad():
            self.head.weight.copy_(
                torch.from_numpy(weights.reshape(1, -1).astype(np.float32))
            )
            self.head.bias.copy_(torch.tensor([bias], dtype=torch.float32))

    def forward(self, image: torch.Tensor, color_features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cnn_feat = self.backbone(image)
        combined = torch.cat([cnn_feat, color_features], dim=1)
        return self.head(combined).squeeze(-1)


# ── CNN feature extraction (reused from train_hybrid.py) ──────────────────

def extract_cnn_features(
    backbone_name: str, data_root: Path, metadata_csv: Path,
    input_size: int, val_cfg: dict, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract frozen backbone features for every crop (with TTA)."""
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
    """
    Average CNN features per patient, merge with hand-crafted color features,
    return train/val/test splits.
    """
    unique_pids = np.unique(patient_ids)
    print(f"\nAggregating {len(cnn_features)} crops → {len(unique_pids)} patients")

    patient_cnn, patient_hb, patient_split, patient_session = {}, {}, {}, {}
    for pid in unique_pids:
        m = patient_ids == pid
        patient_cnn[pid] = cnn_features[m].mean(axis=0)
        patient_hb[pid] = hb_values[m][0]
        patient_split[pid] = splits[m][0]
        patient_session[pid] = sessions[m][0]

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

    # Reorder colour columns to match canonical ordering
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


# ── Train Ridge → de-scale → export ──────────────────────────────────────

def train_and_export(data: dict, cnn_dim: int, backbone_name: str,
                     save_dir: Path):
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["val"]["X"], data["val"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]

    total_dim = X_train.shape[1]
    color_dim = total_dim - cnn_dim
    assert color_dim == N_COLOR, f"Expected {N_COLOR} color features, got {color_dim}"

    # Fit scaler + Ridge
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    ridge = RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)
    ridge.fit(X_train_s, y_train)

    for name, Xs, ys in [("Val", X_val_s, y_val), ("Test", X_test_s, y_test)]:
        pred = ridge.predict(Xs)
        print(f"  {name}: MAE={mean_absolute_error(ys, pred):.3f}  "
              f"R²={r2_score(ys, pred):.4f}")

    # De-scale weights so the model accepts raw features
    coefs = ridge.coef_
    intercept = ridge.intercept_
    scale = scaler.scale_
    mean = scaler.mean_
    w_raw = coefs / scale                          # (total_dim,)
    b_raw = intercept - np.sum(coefs * mean / scale)

    # ── Build & verify PyTorch module ──
    model = HemoLensHybridColor(backbone_name, cnn_dim, color_dim, w_raw, b_raw)
    model.eval()

    # Sanity-check: predict on one training sample
    with torch.no_grad():
        x_cnn = torch.from_numpy(X_train[0:1, :cnn_dim].astype(np.float32))
        x_col = torch.from_numpy(X_train[0:1, cnn_dim:].astype(np.float32))
        # Manual forward through head only (skip backbone — backbone features
        # were already extracted).
        combined = torch.cat([x_cnn, x_col], dim=1)
        hb_pt = model.head(combined).squeeze(-1).item()
        hb_sk = ridge.predict(scaler.transform(X_train[0:1]))[0]
        print(f"\n  Sanity check: sklearn={hb_sk:.3f}  pytorch={hb_pt:.3f}  "
              f"delta={abs(hb_sk - hb_pt):.6f}")

    # ── Export ONNX ──
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_color = torch.randn(1, color_dim)
    onnx_path = save_dir / "hemolens_hybrid_color.onnx"

    torch.onnx.export(
        model,
        (dummy_img, dummy_color),
        str(onnx_path),
        input_names=["image", "color_features"],
        output_names=["hb_prediction"],
        opset_version=13,
        dynamic_axes={
            "image":          {0: "batch"},
            "color_features": {0: "batch"},
            "hb_prediction":  {0: "batch"},
        },
    )
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"\n  ONNX saved → {onnx_path}  ({size_mb:.1f} MB)")

    # Verify with ONNX Runtime
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "image": dummy_img.numpy(),
        "color_features": dummy_color.numpy(),
    })
    print(f"  ONNX test inference: Hb = {out[0][0]:.2f} g/dL")

    # Also copy to web-demo/model/
    web_model_dir = save_dir.parent / ".." / "web-demo" / "model"
    web_model_dir.mkdir(parents=True, exist_ok=True)
    web_path = web_model_dir / "hemolens_hybrid_color.onnx"
    import shutil
    shutil.copy2(onnx_path, web_path)
    print(f"  Copied → {web_path}")

    # Save feature order manifest for JS
    manifest = {
        "cnn_dim": cnn_dim,
        "color_dim": color_dim,
        "color_feature_names": COLOR_FEATURE_NAMES,
        "ridge_alpha": float(ridge.alpha_),
        "test_mae": float(mean_absolute_error(y_test, ridge.predict(X_test_s))),
        "test_r2": float(r2_score(y_test, ridge.predict(X_test_s))),
    }
    manifest_path = save_dir / "hybrid_color_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest → {manifest_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export dual-input (CNN + color) ONNX model for HemoLens web demo")
    parser.add_argument("--config", default="configs/mobilenet_edge.yaml")
    parser.add_argument("--color-features", default="../data/processed/color_features.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    data_root = Path(cfg["data"]["root"])
    metadata_csv = Path(cfg["data"]["metadata_csv"])
    color_csv = Path(args.color_features)
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — extract CNN features
    cnn_feats, hb, pids, splits, sessions = extract_cnn_features(
        cfg["model"]["backbone"], data_root, metadata_csv,
        cfg["model"]["input_size"], cfg["augmentation"]["val"], device,
    )

    # Step 2 — build patient-level matrix with colour features
    data, cnn_dim = build_patient_features(
        cnn_feats, hb, pids, splits, sessions, color_csv,
    )

    # Step 3 — train Ridge + export dual-input ONNX
    print(f"\n{'='*60}")
    print(f"Training Ridge on CNN({cnn_dim}) + Color({N_COLOR}) features")
    print(f"{'='*60}")
    train_and_export(data, cnn_dim, cfg["model"]["backbone"], save_dir)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
