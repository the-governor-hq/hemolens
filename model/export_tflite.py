"""
HemoLens — TFLite Export Script

Convert a trained PyTorch HemoLensModel checkpoint to TFLite format
with optional INT8 quantization for on-device inference.

Pipeline: PyTorch → ONNX → TensorFlow SavedModel → TFLite

Usage:
    python export_tflite.py --checkpoint checkpoints/best_model.pth --quantize int8
    python export_tflite.py --checkpoint checkpoints/best_model.pth --quantize float16
    python export_tflite.py --checkpoint checkpoints/best_model.pth --quantize none
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def load_model(checkpoint_path: str) -> tuple:
    """Load a HemoLensModel from checkpoint."""
    from train import HemoLensModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]

    model = HemoLensModel(
        backbone_name=model_cfg["backbone"],
        pretrained=False,
        hidden_dim=model_cfg["head"]["hidden_dim"],
        dropout=model_cfg["head"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    input_size = model_cfg["input_size"]
    print(f"Loaded model from {checkpoint_path} (val_mae={ckpt.get('val_mae', '?')})")
    return model, input_size


def export_to_onnx(model: torch.nn.Module, input_size: int, output_path: str) -> str:
    """Export PyTorch model to ONNX format."""
    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_image"],
        output_names=["hb_prediction"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "hb_prediction": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"ONNX model saved → {output_path}")
    return output_path


def onnx_to_tflite(
    onnx_path: str,
    output_path: str,
    quantize: str = "int8",
    input_size: int = 224,
) -> str:
    """
    Convert ONNX model to TFLite with optional quantization.

    Args:
        quantize: 'none', 'float16', or 'int8'
    """
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # ONNX → TF SavedModel
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    
    saved_model_dir = str(Path(output_path).parent / "saved_model_tmp")
    tf_rep.export_graph(saved_model_dir)
    print(f"TF SavedModel exported → {saved_model_dir}")

    # TF SavedModel → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("Applying float16 quantization...")

    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for calibration
        def representative_dataset():
            for _ in range(100):
                data = np.random.randn(1, input_size, input_size, 3).astype(np.float32)
                # Normalize to ImageNet range
                data = (data * 0.225) + 0.45
                data = np.clip(data, 0.0, 1.0)
                yield [data]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32  # keep output in float for Hb value
        print("Applying INT8 quantization with calibration...")

    else:
        print("No quantization — exporting float32 model...")

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved → {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="HemoLens TFLite Export")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "float16", "int8"],
        default="int8",
        help="Quantization mode (default: int8)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../assets",
        help="Output directory for .tflite file",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, input_size = load_model(args.checkpoint)

    # Step 1: PyTorch → ONNX
    onnx_path = str(output_dir / "hemolens_model.onnx")
    export_to_onnx(model, input_size, onnx_path)

    # Step 2: ONNX → TFLite
    suffix = f"_{args.quantize}" if args.quantize != "none" else ""
    tflite_path = str(output_dir / f"hemolens_model{suffix}.tflite")
    onnx_to_tflite(onnx_path, tflite_path, args.quantize, input_size)

    print(f"\nExport complete! Model ready for deployment: {tflite_path}")


if __name__ == "__main__":
    main()
