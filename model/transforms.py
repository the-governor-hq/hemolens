"""
HemoLens — Augmentation / transform pipelines.
"""

from torchvision import transforms


def get_train_transforms(input_size: int = 224, config: dict | None = None) -> transforms.Compose:
    """Training augmentation pipeline with color jitter, flips, and rotation."""
    cfg = config or {}
    cj = cfg.get("color_jitter", {})

    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=cfg.get("horizontal_flip", 0.5)),
        transforms.RandomRotation(degrees=cfg.get("random_rotation", 15)),
        transforms.ColorJitter(
            brightness=cj.get("brightness", 0.2),
            contrast=cj.get("contrast", 0.2),
            saturation=cj.get("saturation", 0.15),
            hue=cj.get("hue", 0.05),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]),
            std=cfg.get("normalize", {}).get("std", [0.229, 0.224, 0.225]),
        ),
    ])


def get_val_transforms(input_size: int = 224, config: dict | None = None) -> transforms.Compose:
    """Validation / test transform — deterministic resize + center crop."""
    cfg = config or {}

    return transforms.Compose([
        transforms.Resize(cfg.get("resize", 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]),
            std=cfg.get("normalize", {}).get("std", [0.229, 0.224, 0.225]),
        ),
    ])
