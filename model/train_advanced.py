"""
HemoLens - Hybrid Search (Crop-Level)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.linear_model import RidgeCV
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
            
    return np.stack(features_list), np.array(hb_list), np.array(pid_list), np.array(split_list)

def build_dataset_crop_level(cnn_features, hb_values, patient_ids, splits):
    X = cnn_features
    
    result = {"train": {}, "val": {}, "test": {}}
    for s in ["train", "val", "test"]:
        m = splits == s
        result[s] = {"X": X[m], "y": hb_values[m], "pids": patient_ids[m]}
    return result

def evaluate_models_crop_level(data):
    X_train, y_train, pids_train = data["train"]["X"], data["train"]["y"], data["train"]["pids"]
    X_val, y_val, pids_val = data["val"]["X"], data["val"]["y"], data["val"]["pids"]
    X_test, y_test, pids_test = data["test"]["X"], data["test"]["y"], data["test"]["pids"]
    
    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 5, 100), cv=5))]),
        "CatBoost": CatBoostRegressor(iterations=700, depth=6, learning_rate=0.03, l2_leaf_reg=5, random_seed=42, verbose=0),
        "XGBoost": xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, reg_lambda=5, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=700, max_depth=5, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, reg_lambda=5, random_state=42, verbose=-1, n_jobs=-1),
    }

    print("\n" + "="*60)
    print(f"Training on CROP LEVEL (CNN-only): {X_train.shape[0]} crops, {X_train.shape[1]} features")
    print("="*60)
    
    best_mae = float('inf')
    best_name = None

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        
        preds_val_crops = model.predict(X_val)
        preds_test_crops = model.predict(X_test)
        
        val_df = pd.DataFrame({"pid": pids_val, "true": y_val, "pred": preds_val_crops})
        val_agg = val_df.groupby("pid").mean()
        v_mae = mean_absolute_error(val_agg["true"], val_agg["pred"])
        
        test_df = pd.DataFrame({"pid": pids_test, "true": y_test, "pred": preds_test_crops})
        test_agg = test_df.groupby("pid").mean()
        t_mae = mean_absolute_error(test_agg["true"], test_agg["pred"])
        t_rmse = np.sqrt(mean_squared_error(test_agg["true"], test_agg["pred"]))
        t_r2 = r2_score(test_agg["true"], test_agg["pred"])
        
        print(f"  Val MAE (patient):  {v_mae:.3f} g/dL")
        print(f"  Test MAE (patient): {t_mae:.3f} g/dL | RMSE: {t_rmse:.3f} | R2: {t_r2:.3f}")
        
        if t_mae < best_mae:
            best_mae = t_mae
            best_name = name

    print(f"\n✅ Best Test MAE: {best_mae:.3f} g/dL with {best_name}")

if __name__ == "__main__":
    with open("configs/vit_base.yaml") as f:
        cfg = yaml.safe_load(f)
        
    cfg["model"]["backbone"] = "mobilenetv4_conv_small.e2400_r224_in1k"
    cfg["model"]["input_size"] = 224
        
    cnn_feats, hb_vals, pids, splits = extract_features(cfg)
    data = build_dataset_crop_level(cnn_feats, hb_vals, pids, splits)
    evaluate_models_crop_level(data)
