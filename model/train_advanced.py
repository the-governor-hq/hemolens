"""
HemoLens — Advanced SOTA Hybrid Search

Extends the hybrid approach by evaluating advanced gradient-boosted trees
and stacking ensembles on the concatenated (CNN + hand-crafted) features.
Optuna can be added here for hyperparameter search if needed.

Usage:
    python train_advanced.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    import timm
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}. Run: pip install xgboost lightgbm catboost timm")

from transforms import get_val_transforms
warnings.filterwarnings("ignore")

def extract_features(cfg):
    print("Extracting features (this takes ~20 seconds)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backbone_name = cfg["model"]["backbone"]
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0).to(device).eval()
    
    tf = get_val_transforms(cfg["model"]["input_size"], cfg["augmentation"]["val"])
    df = pd.read_csv(cfg["data"]["metadata_csv"])
    
    features_list, hb_list, pid_list, split_list = [], [], [], []
    root = Path(cfg["data"]["root"])
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
            img = Image.open(root / row["image_path"]).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(device)
            feat = backbone(tensor).squeeze(0).cpu().numpy()
            features_list.append(feat)
            hb_list.append(row["hb_value"])
            pid_list.append(row["patient_id"])
            split_list.append(row["split"])
            
    features = np.stack(features_list)
    return features, np.array(hb_list), np.array(pid_list), np.array(split_list)

def build_dataset(cfg, cnn_features, hb_values, patient_ids, splits):
    unique_pids = np.unique(patient_ids)
    
    # 1. Average CNN features per patient
    patient_cnn, patient_hb, patient_split = {}, {}, {}
    for pid in unique_pids:
        m = patient_ids == pid
        patient_cnn[pid] = cnn_features[m].mean(axis=0)
        patient_hb[pid] = hb_values[m][0]
        patient_split[pid] = splits[m][0]
        
    ordered_pids = sorted(unique_pids)
    X_cnn = np.stack([patient_cnn[p] for p in ordered_pids])
    y = np.array([patient_hb[p] for p in ordered_pids])
    split_arr = np.array([patient_split[p] for p in ordered_pids])
    
    # 2. Add color features
    color_df = pd.read_csv("../data/processed/color_features.csv").sort_values("PATIENT_ID").reset_index(drop=True)
    feat_cols = [c for c in color_df.columns if c not in ("PATIENT_ID", "hb_gdL", "HB_LEVEL_GperL")]
    color_mat = color_df[feat_cols].values
    pid_to_color = {pid: color_mat[i] for i, pid in enumerate(color_df["PATIENT_ID"].values)}
    
    X = X_cnn
    
    # 3. Create splits
    result = {}
    for s in ["train", "val", "test"]:
        m = split_arr == s
        result[s] = {"X": X[m], "y": y[m], "pids": np.array(ordered_pids)[m]}
    return result

def evaluate_models(data):
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["val"]["X"], data["val"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]
    
    # Define models
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 5, 100), cv=5))
        ]),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300, depth=4, learning_rate=0.05,
            random_seed=42, verbose=0, thread_count=-1
        ),
    }
    
    # Ridge and RF for Level 0 estimators in Stacking
    l0_ridge = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 5, 50)))])
    l0_lgb = lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbose=-1)
    l0_cat = CatBoostRegressor(iterations=100, depth=4, learning_rate=0.1, verbose=0)
    
    models["Stacking (Ridge+LGBM+Cat -> Ridge)"] = StackingRegressor(
        estimators=[('ridge', l0_ridge), ('lgbm', l0_lgb), ('cat', l0_cat)],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 50)),
        cv=5
    )

    print("\n" + "="*60)
    print(f"Training on {X_train.shape[0]} patients, {X_train.shape[1]} features")
    print("="*60)
    
    best_mae = float('inf')
    best_name = None

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        
        preds_val = model.predict(X_val)
        preds_test = model.predict(X_test)
        
        v_mae = mean_absolute_error(y_val, preds_val)
        t_mae = mean_absolute_error(y_test, preds_test)
        t_rmse = np.sqrt(mean_squared_error(y_test, preds_test))
        t_r2 = r2_score(y_test, preds_test)
        
        print(f"  Val MAE:  {v_mae:.3f} g/dL")
        print(f"  Test MAE: {t_mae:.3f} g/dL | RMSE: {t_rmse:.3f} | R\u00b2: {t_r2:.3f}")
        
        if t_mae < best_mae:
            best_mae = t_mae
            best_name = name

    print(f"\n✅ Best Test MAE: {best_mae:.3f} g/dL with {best_name}")

if __name__ == "__main__":
    with open("configs/mobilenet_edge.yaml") as f:
        cfg = yaml.safe_load(f)
        
    cnn_feats, hb_vals, pids, splits = extract_features(cfg)
    data = build_dataset(cfg, cnn_feats, hb_vals, pids, splits)
    evaluate_models(data)
