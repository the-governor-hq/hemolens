"""
HemoLens — Dataset module.

Custom PyTorch Dataset for fingernail images with hemoglobin (Hb) regression labels.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FingernailHbDataset(Dataset):
    """
    Dataset for fingernail bed images with Hb (g/dL) labels.

    Expects a CSV with columns:
        - image_path: relative path to the image from `root`
        - hb_value: hemoglobin level in g/dL (float)
        - split: one of 'train', 'val', 'test'

    Args:
        root: Root directory containing images.
        csv_path: Path to the metadata CSV.
        split: Which split to load ('train', 'val', 'test').
        transform: Optional torchvision transform pipeline.
    """

    def __init__(
        self,
        root: str | Path,
        csv_path: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)

        print(f"[FingernailHbDataset] Loaded {len(self.data)} samples for '{split}' split")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        row = self.data.iloc[idx]
        img_path = self.root / row["image_path"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Label
        hb_value = np.float32(row["hb_value"])

        return image, hb_value

    @property
    def hb_values(self) -> np.ndarray:
        """Return all Hb values for this split (useful for statistics)."""
        return self.data["hb_value"].values

    def __repr__(self) -> str:
        return (
            f"FingernailHbDataset(root={self.root}, "
            f"samples={len(self)}, "
            f"hb_range=[{self.data['hb_value'].min():.1f}, "
            f"{self.data['hb_value'].max():.1f}])"
        )
