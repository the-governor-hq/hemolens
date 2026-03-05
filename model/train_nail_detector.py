"""
HemoLens — Train YOLOv8-nano nail detector & export to ONNX.

Trains a tiny object detector to find fingernail bounding boxes in
full-hand photos. The exported ONNX model runs in the browser alongside
the existing Hb regression model.

Pipeline:
  1. Train YOLOv8n on the YOLO-format dataset from prepare_yolo_dataset.py
  2. Evaluate on val + test splits
  3. Export to ONNX (opset 13, static input 320×320 for speed)

Usage:
    python train_nail_detector.py
    python train_nail_detector.py --epochs 100 --imgsz 320 --batch 16
    python train_nail_detector.py --export-only     # skip training, just export best
"""

import argparse
import shutil
from pathlib import Path

import yaml


def train(
    data_yaml: str = "../data/nail_detection/data.yaml",
    epochs: int = 80,
    imgsz: int = 320,
    batch: int = 16,
    model: str = "yolov8n.pt",
    project: str = "checkpoints/nail_detector",
    name: str = "run",
    device: str = "",
    patience: int = 20,
):
    """Train YOLOv8-nano nail detector."""
    from ultralytics import YOLO

    print("=" * 60)
    print("HemoLens — Nail Detector Training (YOLOv8-nano)")
    print("=" * 60)

    yolo = YOLO(model)

    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device or None,
        patience=patience,
        # Augmentation — moderate for small dataset
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,       # no vertical flip for hands
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Training params
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=10,
        # Save
        save=True,
        save_period=-1,    # only save best
        plots=True,
        verbose=True,
    )

    return yolo, results


def evaluate(yolo, data_yaml: str):
    """Evaluate on val and test splits."""
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    # Val
    print("\n--- Validation ---")
    val_metrics = yolo.val(data=data_yaml, split="val")

    # Test
    print("\n--- Test ---")
    test_metrics = yolo.val(data=data_yaml, split="test")

    return val_metrics, test_metrics


def export_onnx(
    weights: str,
    imgsz: int = 320,
    opset: int = 13,
    simplify: bool = True,
    output_dir: str = "checkpoints",
):
    """Export best model to ONNX for browser deployment."""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("ONNX Export")
    print("=" * 60)

    yolo = YOLO(weights)

    onnx_path = yolo.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,     # static shape for WASM performance
        half=False,        # fp32 for browser compatibility
    )

    print(f"Exported: {onnx_path}")

    # Copy to web-demo model dir
    web_model_dir = Path("../web-demo/model")
    web_model_dir.mkdir(parents=True, exist_ok=True)
    dst = web_model_dir / "nail_detector.onnx"
    shutil.copy2(onnx_path, dst)
    print(f"Copied to: {dst}")

    import os
    size_kb = os.path.getsize(dst) / 1024
    print(f"Model size: {size_kb:.0f} KB ({size_kb/1024:.1f} MB)")

    return str(dst)


def find_best_weights(project: str = "checkpoints/nail_detector") -> str:
    """Find the best.pt from the latest training run."""
    # YOLO may save under runs/detect/<project> or directly under <project>
    search_paths = [
        Path(project),
        Path("../runs/detect") / project,
        Path("runs/detect") / project,
    ]

    for project_dir in search_paths:
        if not project_dir.exists():
            continue
        # Find all run directories, pick the latest
        runs = sorted(
            [p for p in project_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        for run_dir in reversed(runs):
            best = run_dir / "weights" / "best.pt"
            if best.exists():
                print(f"Found best weights: {best}")
                return str(best)

    raise FileNotFoundError(
        f"No best.pt found. Searched: {[str(p) for p in search_paths]}"
    )


def main():
    parser = argparse.ArgumentParser(description="HemoLens Nail Detector")
    parser.add_argument("--data", default="../data/nail_detection/data.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=320,
                        help="Input size (320 = fast for browser, 640 = more accurate)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--device", default="", help="cuda device or cpu")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training, export existing best model")
    parser.add_argument("--no-export", action="store_true",
                        help="Train only, skip ONNX export")
    parser.add_argument("--opset", type=int, default=13,
                        help="ONNX opset version (13 for onnxruntime-web)")
    args = parser.parse_args()

    if args.export_only:
        weights = find_best_weights()
        print(f"Using existing weights: {weights}")
        export_onnx(weights, imgsz=args.imgsz, opset=args.opset)
        return

    # Prepare dataset if not done
    data_path = Path(args.data)
    if not data_path.exists():
        print("Dataset not found — running prepare_yolo_dataset.py first...")
        import subprocess, sys
        subprocess.run([sys.executable, "prepare_yolo_dataset.py"], check=True)

    # Train
    yolo, results = train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model=args.model,
        device=args.device,
        patience=args.patience,
    )

    # Evaluate
    evaluate(yolo, args.data)

    # Export
    if not args.no_export:
        weights = find_best_weights()
        export_onnx(weights, imgsz=args.imgsz, opset=args.opset)

    print("\nDone!")


if __name__ == "__main__":
    main()
