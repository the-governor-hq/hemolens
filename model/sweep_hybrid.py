"""
HemoLens — Hybrid Model Sweep

Systematically sweeps backbone × head combinations to find the best
frozen-feature + classical-head configuration.

Backbones tested:
  - mobilenetv4_conv_small (baseline, 2.49M)
  - efficientnet_b0 (5.3M, stronger features)
  - mobilenetv3_small_100 (2.5M, lighter)
  - tf_efficientnetv2_b0 (7.1M, recent)

Heads tested:
  - Ridge (full features)
  - Ridge+PCA64
  - Ridge+PCA32
  - CatBoost (gradient boosting, strong on tabular data)

Uses 3-fold session-aware CV (17 sessions → ~5-6 per val fold) for
more stable cross-validation estimates than 5-fold.

Usage:
    python sweep_hybrid.py
    python sweep_hybrid.py --backbones mobilenetv4_conv_small efficientnet_b0
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from transforms import get_val_transforms

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# WHO anemia thresholds
_WHO_BINS = [
    ("Severe",   0.0,  8.0),
    ("Moderate", 8.0, 11.0),
    ("Mild",    11.0, 13.0),
    ("Normal",  13.0, 99.0),
]


def severity_breakdown(y_true, y_pred):
    report = {}
    for label, lo, hi in _WHO_BINS:
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        mae = float(mean_absolute_error(y_true[mask], y_pred[mask])) if n > 0 else float("nan")
        report[label] = {"n": n, "mae": mae}
    return report


def extract_features(backbone_name, data_root, metadata_csv, input_size, val_cfg, device, tta=True):
    """Extract frozen CNN features with optional TTA."""
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    backbone.eval().to(device)
    tf = get_val_transforms(input_size, val_cfg)
    df = pd.read_csv(metadata_csv)

    features, hb_vals, pids, splits, sessions = [], [], [], [], []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {backbone_name}"):
            img = Image.open(data_root / row["image_path"]).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(device)
            feat = backbone(tensor).squeeze(0).cpu().numpy()

            if tta:
                flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                feat_flip = backbone(tf(flipped).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                feat = (feat + feat_flip) / 2.0

            features.append(feat)
            hb_vals.append(row["hb_value"])
            pids.append(row["patient_id"])
            splits.append(row["split"])
            sessions.append(row.get("session", row["patient_id"]))

    embed_dim = len(features[0])
    print(f"  -> {len(features)} crops, embed_dim={embed_dim}")
    return (np.stack(features), np.array(hb_vals, dtype=np.float32),
            np.array(pids), np.array(splits), np.array(sessions))


def aggregate_patients(cnn_feats, hb_vals, pids, splits, sessions, color_csv=None):
    """Average per-patient, optionally merge color features."""
    unique = np.unique(pids)
    p_cnn, p_hb, p_split, p_ses = {}, {}, {}, {}
    for pid in unique:
        m = pids == pid
        p_cnn[pid] = cnn_feats[m].mean(axis=0)
        p_hb[pid] = hb_vals[m][0]
        p_split[pid] = splits[m][0]
        p_ses[pid] = sessions[m][0]

    ordered = sorted(unique)
    X_cnn = np.stack([p_cnn[p] for p in ordered])
    y = np.array([p_hb[p] for p in ordered])
    split_arr = np.array([p_split[p] for p in ordered])
    pid_arr = np.array(ordered)
    ses_arr = np.array([p_ses[p] for p in ordered])

    if color_csv and Path(color_csv).exists():
        cdf = pd.read_csv(color_csv)
        feat_cols = [c for c in cdf.columns if c not in ("PATIENT_ID", "hb_gdL", "HB_LEVEL_GperL")]
        cmat = cdf.set_index("PATIENT_ID")[feat_cols]
        X_color = np.stack([cmat.loc[p].values if p in cmat.index else np.zeros(len(feat_cols)) for p in ordered])
        X = np.hstack([X_cnn, X_color])
    else:
        X = X_cnn

    result = {}
    for s in ["train", "val", "test"]:
        m = split_arr == s
        if m.sum() > 0:
            result[s] = {"X": X[m], "y": y[m], "pids": pid_arr[m], "sessions": ses_arr[m]}
    return result, X_cnn.shape[1]


def build_heads():
    """Build model zoo of heads to sweep."""
    heads = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)),
        ]),
        "Ridge+PCA64": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=64)),
            ("model", RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)),
        ]),
        "Ridge+PCA32": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=32)),
            ("model", RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)),
        ]),
    }
    if HAS_CATBOOST:
        heads["CatBoost"] = CatBoostRegressor(
            iterations=500, depth=4, learning_rate=0.03,
            l2_leaf_reg=10, random_seed=42, verbose=0,
        )
    return heads


def evaluate_holdout(model, data):
    """Evaluate on val + test splits."""
    results = {}
    for split in ["val", "test"]:
        if split not in data:
            continue
        X, y = data[split]["X"], data[split]["y"]
        if hasattr(model, "predict"):
            pred = model.predict(X)
        else:
            pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        sev = severity_breakdown(y, pred)
        results[split] = {"mae": float(mae), "r2": float(r2), "severity": sev}
    return results


def cross_validate(data, n_splits=3):
    """Session-aware GroupKFold CV."""
    X = np.vstack([data["train"]["X"], data["val"]["X"]])
    y = np.concatenate([data["train"]["y"], data["val"]["y"]])
    sessions = np.concatenate([data["train"]["sessions"], data["val"]["sessions"]])

    n_groups = len(np.unique(sessions))
    effective = min(n_splits, n_groups)

    gkf = GroupKFold(n_splits=effective)
    maes, r2s = [], []

    for train_idx, val_idx in gkf.split(X, y, groups=sessions):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[val_idx])
        maes.append(mean_absolute_error(y[val_idx], pred))
        r2s.append(r2_score(y[val_idx], pred))

    return {
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "cv_r2_mean": float(np.mean(r2s)),
        "n_folds": effective,
        "n_groups": n_groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobilenet_edge.yaml")
    parser.add_argument("--backbones", nargs="+", default=[
        "mobilenetv4_conv_small",
        "efficientnet_b0",
        "mobilenetv3_small_100",
        "tf_efficientnetv2_b0",
    ])
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--color-features", default="../data/processed/color_features.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(cfg["data"]["root"])
    metadata_csv = Path(cfg["data"]["metadata_csv"])
    color_csv = None if args.no_color else args.color_features
    input_size = cfg["model"]["input_size"]
    val_cfg = cfg["augmentation"]["val"]

    all_results = {}

    for backbone in args.backbones:
        print(f"\n{'='*60}")
        print(f"BACKBONE: {backbone}")
        print(f"{'='*60}")

        try:
            feats, hb, pids, splits, sessions = extract_features(
                backbone, data_root, metadata_csv, input_size, val_cfg, device,
            )
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        data, cnn_dim = aggregate_patients(feats, hb, pids, splits, sessions, color_csv)
        print(f"  Train: {data['train']['X'].shape}, Val: {data['val']['X'].shape}, "
              f"Test: {data['test']['X'].shape}")

        backbone_results = {}
        heads = build_heads()

        for head_name, model in heads.items():
            print(f"\n  --- {head_name} ---")
            X_tr, y_tr = data["train"]["X"], data["train"]["y"]

            if isinstance(model, Pipeline):
                # Check PCA n_components doesn't exceed features
                for step_name, step in model.steps:
                    if isinstance(step, PCA):
                        max_comp = min(step.n_components, X_tr.shape[0], X_tr.shape[1])
                        if max_comp < step.n_components:
                            step.n_components = max_comp
                model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr)

            holdout = evaluate_holdout(model, data)
            for split, m in holdout.items():
                print(f"    {split}: MAE={m['mae']:.3f}, R²={m['r2']:.4f}")
                for cls, sv in m["severity"].items():
                    if sv["n"] > 0:
                        print(f"      {cls:8s}: n={sv['n']:3d}, MAE={sv['mae']:.3f}")

            backbone_results[head_name] = holdout

        # CV on best head (Ridge) — uses combined train+val
        cv = cross_validate(data, n_splits=args.cv_folds)
        print(f"\n  {args.cv_folds}-Fold Session-Aware CV: "
              f"MAE={cv['cv_mae_mean']:.3f} +/- {cv['cv_mae_std']:.3f}, "
              f"R2={cv['cv_r2_mean']:.4f} ({cv['n_groups']} groups)")

        backbone_results["cv"] = cv
        backbone_results["cnn_dim"] = cnn_dim
        all_results[backbone] = backbone_results

    # Summary table
    print(f"\n\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Backbone':<28} {'Head':<14} {'Val MAE':>8} {'Test MAE':>9} {'Test R2':>8} {'CV MAE':>12}")
    print("-" * 80)

    best_combo = None
    best_test_mae = float("inf")

    for backbone, br in all_results.items():
        cv_str = f"{br['cv']['cv_mae_mean']:.3f}+/-{br['cv']['cv_mae_std']:.3f}" if "cv" in br else "-"
        for head in ["Ridge", "Ridge+PCA64", "Ridge+PCA32", "CatBoost"]:
            if head not in br:
                continue
            vm = br[head].get("val", {}).get("mae", float("nan"))
            tm = br[head].get("test", {}).get("mae", float("nan"))
            tr = br[head].get("test", {}).get("r2", float("nan"))
            print(f"{backbone:<28} {head:<14} {vm:>8.3f} {tm:>9.3f} {tr:>8.4f} {cv_str:>12}")

            if tm < best_test_mae:
                best_test_mae = tm
                best_combo = (backbone, head)

    if best_combo:
        print(f"\n* BEST: {best_combo[0]} + {best_combo[1]} -- Test MAE={best_test_mae:.3f} g/dL")

    # Save results
    save_path = Path("checkpoints/sweep_results.json")
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved -> {save_path}")


if __name__ == "__main__":
    main()
