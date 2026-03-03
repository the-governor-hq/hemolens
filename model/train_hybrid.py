"""
HemoLens — Hybrid Feature Extraction + Regression

Strategy:
  1. Frozen MobileNetV4-Conv-Small backbone → 1280-dim CNN features per crop
  2. Average 3 crops per patient → 1280-dim patient vector
  3. Concatenate with 51 handcrafted color features → 1331-dim
  4. Train Ridge / ElasticNet / small MLP with patient-level CV
  5. Export end-to-end model (backbone + head) for TFLite

Why: With only 250 patients, fine-tuning 2.5M params overfits.
     Frozen ImageNet features + classical head is the proven small-data approach.

Usage:
    python train_hybrid.py --config configs/mobilenet_edge.yaml
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("Install timm: pip install timm")

from transforms import get_val_transforms

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. Extract frozen CNN features
# ---------------------------------------------------------------------------

def extract_cnn_features(
    backbone_name: str,
    data_root: Path,
    metadata_csv: Path,
    input_size: int,
    val_cfg: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features from a frozen pretrained backbone for all crops.

    Returns:
        features: (N_crops, embed_dim) — CNN features per crop
        hb_values: (N_crops,) — Hb labels
        patient_ids: (N_crops,) — patient IDs for grouping
        splits: (N_crops,) — split labels
    """
    print(f"\n{'='*60}")
    print(f"Extracting frozen {backbone_name} features")
    print(f"{'='*60}")

    # Load backbone in eval mode — no gradients
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    backbone.eval()
    backbone.to(device)

    # Deterministic val transforms (no augmentation for feature extraction)
    tf = get_val_transforms(input_size, val_cfg)

    # Load metadata
    df = pd.read_csv(metadata_csv)
    print(f"Loaded {len(df)} crops from {df['patient_id'].nunique()} patients")

    features_list = []
    hb_list = []
    pid_list = []
    split_list = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            img_path = data_root / row["image_path"]
            img = Image.open(img_path).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(device)

            feat = backbone(tensor).squeeze(0).cpu().numpy()
            features_list.append(feat)
            hb_list.append(row["hb_value"])
            pid_list.append(row["patient_id"])
            split_list.append(row["split"])

    features = np.stack(features_list)
    hb_values = np.array(hb_list, dtype=np.float32)
    patient_ids = np.array(pid_list)
    splits = np.array(split_list)

    print(f"CNN features shape: {features.shape}")
    return features, hb_values, patient_ids, splits


# ---------------------------------------------------------------------------
# 2. Aggregate crops to patient level & merge color features
# ---------------------------------------------------------------------------

def build_patient_features(
    cnn_features: np.ndarray,
    hb_values: np.ndarray,
    patient_ids: np.ndarray,
    splits: np.ndarray,
    color_features_csv: Path | None = None,
) -> dict:
    """
    Average CNN features across 3 crops per patient, optionally merge with color features.

    Returns dict with train/val/test splits.
    """
    unique_pids = np.unique(patient_ids)
    print(f"\nAggregating {len(cnn_features)} crops → {len(unique_pids)} patients")

    # Average crops per patient
    patient_cnn = {}
    patient_hb = {}
    patient_split = {}
    for pid in unique_pids:
        mask = patient_ids == pid
        patient_cnn[pid] = cnn_features[mask].mean(axis=0)
        patient_hb[pid] = hb_values[mask][0]  # same for all crops
        patient_split[pid] = splits[mask][0]

    # Stack into arrays (ordered by pid)
    ordered_pids = sorted(unique_pids)
    X_cnn = np.stack([patient_cnn[p] for p in ordered_pids])
    y = np.array([patient_hb[p] for p in ordered_pids])
    split_arr = np.array([patient_split[p] for p in ordered_pids])
    pid_arr = np.array(ordered_pids)

    print(f"  CNN features: {X_cnn.shape}")

    # Merge color features if available
    if color_features_csv and color_features_csv.exists():
        color_df = pd.read_csv(color_features_csv)
        # Match by patient ID
        color_df = color_df.sort_values("PATIENT_ID").reset_index(drop=True)

        # Get only the actual feature columns (exclude ID, label columns)
        feat_cols = [c for c in color_df.columns if c not in ("PATIENT_ID", "hb_gdL", "HB_LEVEL_GperL")]
        color_mat = color_df[feat_cols].values  # (250, 51)

        # Align by patient ID
        color_pid_order = color_df["PATIENT_ID"].values
        pid_to_color = {pid: color_mat[i] for i, pid in enumerate(color_pid_order)}

        X_color = np.stack([pid_to_color.get(p, np.zeros(len(feat_cols))) for p in ordered_pids])
        X = np.hstack([X_cnn, X_color])
        print(f"  Color features: {X_color.shape}")
        print(f"  Combined: {X.shape}")
    else:
        X = X_cnn
        print(f"  No color features found, using CNN only: {X.shape}")

    # Split
    result = {}
    for split_name in ["train", "val", "test"]:
        mask = split_arr == split_name
        if mask.sum() > 0:
            result[split_name] = {
                "X": X[mask],
                "y": y[mask],
                "pids": pid_arr[mask],
            }
            print(f"  {split_name}: {mask.sum()} patients")

    return result


# ---------------------------------------------------------------------------
# 3. Train & evaluate models
# ---------------------------------------------------------------------------

def train_and_evaluate(data: dict) -> dict:
    """
    Train multiple regression models on patient-level features.
    Evaluate on val + test sets.
    """
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["val"]["X"], data["val"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]

    print(f"\n{'='*60}")
    print(f"Training on {X_train.shape[0]} patients ({X_train.shape[1]} features)")
    print(f"{'='*60}")

    # Models to try
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                n_alphas=50, cv=5, max_iter=5000,
            )),
        ]),
        "Ridge+PCA128": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=128)),
            ("model", RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)),
        ]),
        "MLP_small": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-2,
                learning_rate="adaptive",
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )),
        ]),
        "MLP_tiny": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=1e-2,
                learning_rate="adaptive",
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )),
        ]),
    }

    results = {}
    best_mae = float("inf")
    best_name = None
    best_model = None

    for name, pipeline in models.items():
        print(f"\n--- {name} ---")
        pipeline.fit(X_train, y_train)

        # Validate
        val_pred = pipeline.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)

        # Test
        test_pred = pipeline.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)

        print(f"  Val:  MAE={val_mae:.3f} | RMSE={val_rmse:.3f} | R²={val_r2:.4f}")
        print(f"  Test: MAE={test_mae:.3f} | RMSE={test_rmse:.3f} | R²={test_r2:.4f}")

        results[name] = {
            "val_mae": float(val_mae), "val_rmse": float(val_rmse), "val_r2": float(val_r2),
            "test_mae": float(test_mae), "test_rmse": float(test_rmse), "test_r2": float(test_r2),
        }

        if val_mae < best_mae:
            best_mae = val_mae
            best_name = name
            best_model = pipeline

    print(f"\n{'='*60}")
    print(f"BEST: {best_name} — Val MAE={best_mae:.3f} g/dL")
    print(f"{'='*60}")

    return results, best_name, best_model


# ---------------------------------------------------------------------------
# 4. Cross-validation for robust estimate
# ---------------------------------------------------------------------------

def cross_validate_best(data: dict, n_splits: int = 5) -> dict:
    """
    Patient-level GroupKFold CV on train+val combined for robust MAE estimate.
    """
    # Combine train + val for CV
    X = np.vstack([data["train"]["X"], data["val"]["X"]])
    y = np.concatenate([data["train"]["y"], data["val"]["y"]])
    pids = np.concatenate([data["train"]["pids"], data["val"]["pids"]])

    print(f"\n{'='*60}")
    print(f"5-Fold Patient-Level Cross-Validation ({len(pids)} patients)")
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=n_splits)
    fold_maes = []
    fold_r2s = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=pids), 1):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)),
        ])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_vl)

        mae = mean_absolute_error(y_vl, pred)
        r2 = r2_score(y_vl, pred)
        fold_maes.append(mae)
        fold_r2s.append(r2)
        print(f"  Fold {fold}: MAE={mae:.3f}, R²={r2:.4f}")

    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    mean_r2 = np.mean(fold_r2s)

    print(f"\n  CV MAE: {mean_mae:.3f} ± {std_mae:.3f} g/dL")
    print(f"  CV R²:  {mean_r2:.4f}")

    return {"cv_mae_mean": float(mean_mae), "cv_mae_std": float(std_mae), "cv_r2_mean": float(mean_r2)}


# ---------------------------------------------------------------------------
# 5. Export end-to-end PyTorch model (for TFLite conversion)
# ---------------------------------------------------------------------------

class HemoLensHybrid(nn.Module):
    """
    End-to-end model: frozen backbone + learned linear head.
    For TFLite export — takes raw image tensor, outputs Hb prediction.
    """

    def __init__(self, backbone_name: str, weights: np.ndarray, bias: float):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        in_dim = weights.shape[0]
        self.head = nn.Linear(in_dim, 1)
        # Load sklearn Ridge weights
        with torch.no_grad():
            self.head.weight.copy_(torch.from_numpy(weights[:in_dim].reshape(1, -1).astype(np.float32)))
            self.head.bias.copy_(torch.tensor([bias], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features).squeeze(-1)


def export_pytorch_model(
    backbone_name: str,
    best_pipeline,
    cnn_dim: int,
    save_dir: Path,
):
    """
    Extract Ridge weights and build end-to-end PyTorch model for TFLite export.
    Only uses CNN features (not color features) for the edge model.
    """
    scaler = best_pipeline.named_steps["scaler"]
    ridge = best_pipeline.named_steps.get("model")
    if ridge is None:
        print("Cannot export: model step not found in pipeline")
        return

    # Ridge coefficients — extract only CNN feature weights
    coefs = ridge.coef_[:cnn_dim]
    intercept = ridge.intercept_

    # Account for scaler: w_raw = w_scaled / scale, b_raw = intercept - sum(w_scaled * mean / scale)
    scale = scaler.scale_[:cnn_dim]
    mean = scaler.mean_[:cnn_dim]
    w_raw = coefs / scale
    b_raw = intercept - np.sum(coefs * mean / scale)

    model = HemoLensHybrid(backbone_name, w_raw, b_raw)
    model.eval()

    save_path = save_dir / "hemolens_hybrid.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "backbone": backbone_name,
        "cnn_dim": cnn_dim,
        "ridge_alpha": float(ridge.alpha_) if hasattr(ridge, "alpha_") else None,
    }, save_path)
    print(f"\nSaved end-to-end model → {save_path}")

    # Also save ONNX for TFLite conversion
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = save_dir / "hemolens_hybrid.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"],
        output_names=["hb_prediction"],
        opset_version=13,
        dynamic_axes={"image": {0: "batch"}, "hb_prediction": {0: "batch"}},
    )
    print(f"Saved ONNX → {onnx_path}")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HemoLens Hybrid Training")
    parser.add_argument("--config", type=str, default="configs/mobilenet_edge.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--color-features", type=str, default="../data/processed/color_features.csv",
                        help="Path to handcrafted color features CSV")
    parser.add_argument("--no-color", action="store_true",
                        help="Use only CNN features (no color features)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip ONNX export")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Paths
    data_root = Path(cfg["data"]["root"])
    metadata_csv = Path(cfg["data"]["metadata_csv"])
    color_csv = Path(args.color_features) if not args.no_color else None

    # Step 1: Extract frozen CNN features
    cnn_features, hb_values, patient_ids, splits = extract_cnn_features(
        backbone_name=cfg["model"]["backbone"],
        data_root=data_root,
        metadata_csv=metadata_csv,
        input_size=cfg["model"]["input_size"],
        val_cfg=cfg["augmentation"]["val"],
        device=device,
    )
    cnn_dim = cnn_features.shape[1]

    # Step 2: Aggregate to patient level + merge color features
    data = build_patient_features(
        cnn_features, hb_values, patient_ids, splits,
        color_features_csv=color_csv,
    )

    # Step 3: Train & evaluate
    results, best_name, best_model = train_and_evaluate(data)

    # Step 4: Cross-validation
    cv_results = cross_validate_best(data)

    # Save results
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"models": results, "cv": cv_results, "best_model": best_name, "cnn_dim": cnn_dim}
    results_path = save_dir / "hybrid_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # Step 5: Export for TFLite (CNN-only head for edge)
    if not args.no_export and "Ridge" in best_name and "PCA" not in best_name:
        export_pytorch_model(
            cfg["model"]["backbone"],
            best_model,
            cnn_dim,
            save_dir,
        )
    elif not args.no_export:
        print(f"\nNote: ONNX export only supported for Ridge (no PCA). Best was '{best_name}'.")
        # Train a pure Ridge on CNN-only features for export
        print("Training Ridge on CNN-only features for edge export...")
        cnn_data = build_patient_features(
            cnn_features, hb_values, patient_ids, splits,
            color_features_csv=None,
        )
        export_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)),
        ])
        X_tr = np.vstack([cnn_data["train"]["X"], cnn_data["val"]["X"]])
        y_tr = np.concatenate([cnn_data["train"]["y"], cnn_data["val"]["y"]])
        export_pipe.fit(X_tr, y_tr)

        # Evaluate on test
        test_pred = export_pipe.predict(cnn_data["test"]["X"])
        test_mae = mean_absolute_error(cnn_data["test"]["y"], test_pred)
        print(f"  CNN-only Ridge test MAE: {test_mae:.3f}")

        export_pytorch_model(
            cfg["model"]["backbone"],
            export_pipe,
            cnn_dim,
            save_dir,
        )

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
