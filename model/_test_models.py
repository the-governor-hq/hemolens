"""Quick test of all ONNX models."""
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
from pathlib import Path

models = {
    'web-demo (10MB, 08:19)': '../web-demo/model/hemolens_hybrid_web.onnx',
    'ckpt_web (9.9MB, 09:00)': 'checkpoints/hemolens_hybrid_web.onnx',
    'ckpt_v2 (9.9MB, 09:00)': 'checkpoints/hemolens_web_v2.onnx',
    'ckpt_orig (150KB)': 'checkpoints/hemolens_hybrid.onnx',
}

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

df = pd.read_csv('../data/processed/metadata_splits.csv')
test = df[df['split'] == 'test'].head(3)

for mname, mpath in models.items():
    sess = ort.InferenceSession(mpath, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"\n=== {mname} ===")
    print(f"  Input: {inp.name} {inp.shape}")
    print(f"  Output: {out.name} {out.shape}")
    for _, row in test.iterrows():
        img_path = Path('../data/processed') / row['image_path']
        img = Image.open(img_path).convert("RGB")
        tensor = tf(img).unsqueeze(0).numpy()
        result = float(sess.run(None, {inp.name: tensor})[0].flat[0])
        print(f"  pid={row['patient_id']}: true={row['hb_value']:.1f}, pred={result:.2f}")
