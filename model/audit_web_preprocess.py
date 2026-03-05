"""Audit the web-demo preprocessing against the training/val preprocessing.

Goal
----
The deployed web demo does its own resize/normalize logic in JavaScript.
If that differs from the preprocessing used during training / feature extraction,
predictions can drift and look like "hallucinations".

This script:
- Loads the exported ONNX model used by the web demo
- Runs inference on a sample of real processed nail crops
- Compares two preprocessing pipelines:
  A) Training/val: Resize(shorter=256) -> CenterCrop(224) -> ImageNet normalize
  B) Current web: Resize(224x224) (stretched) -> ImageNet normalize

Usage
-----
  python audit_web_preprocess.py --n 60

Notes
-----
- Requires: onnxruntime, torch, torchvision, pillow, pandas, numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


_TF_TRAIN_VAL = None


def _preprocess_train_val(img: Image.Image) -> np.ndarray:
    import torch
    from torchvision import transforms

    global _TF_TRAIN_VAL
    if _TF_TRAIN_VAL is None:
        _TF_TRAIN_VAL = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    x = _TF_TRAIN_VAL(img).unsqueeze(0)  # 1x3x224x224
    return x.detach().cpu().numpy().astype(np.float32)


def _preprocess_web_stretch(img: Image.Image) -> np.ndarray:
    # Old web-demo/app.js behavior: direct stretch resize to 224x224.
    img224 = img.resize((224, 224), resample=Image.BILINEAR)
    arr = np.asarray(img224).astype(np.float32) / 255.0  # HWC, RGB, [0,1]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    chw = np.transpose(arr, (2, 0, 1))  # CHW
    return chw[None, ...].astype(np.float32)


def _preprocess_web_match_training(img: Image.Image) -> np.ndarray:
    # New web-demo behavior (after audit fix):
    # Resize shorter side to 256 (preserve aspect) -> CenterCrop 224 -> normalize.
    return _preprocess_train_val(img)


def _predict(ort_sess, x: np.ndarray) -> float:
    # Robustly select names (export uses image -> hb_prediction)
    input_name = ort_sess.get_inputs()[0].name
    out_name = ort_sess.get_outputs()[0].name
    y = ort_sess.run([out_name], {input_name: x})[0]
    return float(np.asarray(y).reshape(-1)[0])


def _predict_tta(ort_sess, x: np.ndarray) -> float:
    # x: 1x3x224x224. Horizontal flip is last dimension.
    y1 = _predict(ort_sess, x)
    x_flip = x[:, :, :, ::-1].copy()
    y2 = _predict(ort_sess, x_flip)
    return (y1 + y2) / 2.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, default="../web-demo/model/hemolens_hybrid_web.onnx")
    p.add_argument("--data-root", type=str, default="../data/processed")
    p.add_argument("--metadata", type=str, default="../data/processed/metadata_splits.csv")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=60)
    p.add_argument(
        "--aggregate",
        type=str,
        default="crop",
        choices=["crop", "patient_mean"],
        help="Evaluate per crop (web-like) or mean over each patient's 3 crops (training-like).",
    )
    p.add_argument(
        "--web-preprocess",
        type=str,
        default="match_training",
        choices=["match_training", "stretch_224"],
        help="Which web preprocessing to simulate (match current web-demo by default).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tta", action="store_true", help="Average prediction with horizontal flip (TTA).")
    args = p.parse_args()

    onnx_path = Path(args.onnx).resolve()
    data_root = Path(args.data_root).resolve()
    meta_path = Path(args.metadata).resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    try:
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("onnxruntime is required. Try: pip install onnxruntime") from e

    df = pd.read_csv(meta_path)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No rows for split={args.split}")

    rng = np.random.default_rng(args.seed)
    if args.aggregate == "crop":
        take = min(args.n, len(df))
        idxs = rng.choice(len(df), size=take, replace=False)
        df = df.iloc[idxs].reset_index(drop=True)
        n_units = take
    else:
        pids = df["patient_id"].unique()
        take = min(args.n, len(pids))
        chosen = rng.choice(pids, size=take, replace=False)
        df = df[df["patient_id"].isin(chosen)].reset_index(drop=True)
        n_units = take

    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    if args.web_preprocess == "match_training":
        web_preprocess = _preprocess_web_match_training
    else:
        web_preprocess = _preprocess_web_stretch

    pred_fn = _predict_tta if args.tta else _predict

    preds_train = []
    preds_web = []
    ys = []
    abs_diffs = []

    if args.aggregate == "crop":
        it = df.iterrows()
        for _, row in it:
            img_path = data_root / row["image_path"]
            img = Image.open(img_path).convert("RGB")

            x_train = _preprocess_train_val(img)
            x_web = web_preprocess(img)

            y_train = pred_fn(ort_sess, x_train)
            y_web = pred_fn(ort_sess, x_web)
            y_true = float(row["hb_value"])

            preds_train.append(y_train)
            preds_web.append(y_web)
            ys.append(y_true)
            abs_diffs.append(abs(y_train - y_web))
    else:
        for pid, g in df.groupby("patient_id"):
            y_true = float(g["hb_value"].iloc[0])
            y_train_list = []
            y_web_list = []
            for _, row in g.iterrows():
                img_path = data_root / row["image_path"]
                img = Image.open(img_path).convert("RGB")
                x_train = _preprocess_train_val(img)
                x_web = web_preprocess(img)
                y_train_list.append(pred_fn(ort_sess, x_train))
                y_web_list.append(pred_fn(ort_sess, x_web))

            y_train = float(np.mean(y_train_list))
            y_web = float(np.mean(y_web_list))

            preds_train.append(y_train)
            preds_web.append(y_web)
            ys.append(y_true)
            abs_diffs.append(abs(y_train - y_web))

    ys = np.asarray(ys)
    preds_train = np.asarray(preds_train)
    preds_web = np.asarray(preds_web)
    abs_diffs = np.asarray(abs_diffs)

    mae_train = float(np.mean(np.abs(preds_train - ys)))
    mae_web = float(np.mean(np.abs(preds_web - ys)))

    print("=== HemoLens audit: ONNX inference vs preprocessing ===")
    print(f"ONNX:   {onnx_path}")
    print(f"Split:  {args.split} ({args.aggregate}, n={n_units})")
    print(f"TTA:    {'on' if args.tta else 'off'}")
    print("")
    print("MAE (lower is better):")
    print(f"  A) train/val preprocessing: {mae_train:.3f} g/dL")
    print(f"  B) web preprocessing ({args.web_preprocess}): {mae_web:.3f} g/dL")
    print("")
    print("Prediction drift (|A-B|):")
    print(f"  mean: {float(abs_diffs.mean()):.3f} g/dL")
    print(f"  p50:  {float(np.percentile(abs_diffs, 50)):.3f} g/dL")
    print(f"  p90:  {float(np.percentile(abs_diffs, 90)):.3f} g/dL")


if __name__ == "__main__":
    main()
