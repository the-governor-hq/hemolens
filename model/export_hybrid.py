"""Quick script to export the saved hybrid model to ONNX and verify."""
import os
import torch
import timm
import numpy as np

# Load saved checkpoint
ckpt = torch.load("checkpoints/hemolens_hybrid.pth", map_location="cpu", weights_only=False)
backbone_name = ckpt["backbone"]
cnn_dim = ckpt["cnn_dim"]
print(f"Backbone: {backbone_name}, CNN dim: {cnn_dim}")

# Rebuild model
backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

head = torch.nn.Linear(cnn_dim, 1)

# Build combined module
class HemoLensExport(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(-1)

model = HemoLensExport(backbone, head)

# Load state dict manually
state = ckpt["model_state_dict"]
backbone_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
model.backbone.load_state_dict(backbone_state, strict=False)
model.head.load_state_dict(head_state)
model.eval()

print(f"Model loaded. Head weight shape: {model.head.weight.shape}")

# Export ONNX
dummy = torch.randn(1, 3, 224, 224)
onnx_path = "checkpoints/hemolens_hybrid.onnx"
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["image"],
    output_names=["hb_prediction"],
    opset_version=13,
    dynamic_axes={"image": {0: "batch"}, "hb_prediction": {0: "batch"}},
)
print(f"ONNX exported -> {onnx_path}")

# Verify with onnxruntime
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path)
out = sess.run(None, {"image": dummy.numpy()})
print(f"ONNX inference test: Hb = {out[0][0]:.2f} g/dL")

size_mb = os.path.getsize(onnx_path) / 1024 / 1024
print(f"ONNX model size: {size_mb:.1f} MB")
