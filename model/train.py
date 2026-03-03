"""
HemoLens — Model Training Script

Train a backbone + regression head for hemoglobin estimation
from fingernail bed images.  Supports any timm backbone
(MobileNetV3, EfficientNet, ViT, etc.).

Usage:
    python train.py --config configs/mobilenet_edge.yaml
    python train.py --config configs/vit_base.yaml
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("Install timm: pip install timm")

from dataset import FingernailHbDataset
from transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HemoLensModel(nn.Module):
    """
    Backbone-agnostic regression model for Hb estimation.

    Works with any timm backbone (MobileNetV3, EfficientNet, ViT, etc.).
    Architecture:  backbone → global pool → FC → ReLU → Dropout → FC → Hb (scalar)
    """

    def __init__(self, backbone_name: str, pretrained: bool, hidden_dim: int, dropout: float):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # [B, embed_dim]
        return self.head(features).squeeze(-1)  # [B]

    def freeze_backbone(self):
        """Freeze backbone parameters — train head only."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 10,
    label_noise_std: float = 0.0,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Train", leave=False)):
        images = images.to(device)
        targets = targets.to(device)

        # Label noise regularization
        if label_noise_std > 0:
            targets = targets + torch.randn_like(targets) * label_noise_std

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg = running_loss / n_batches
            tqdm.write(f"  Step {batch_idx + 1}: loss={avg:.4f}")

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    n_batches = 0

    for images, targets in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        n_batches += 1

        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="HemoLens ViT Training")
    parser.add_argument("--config", type=str, default="configs/vit_base.yaml")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Transforms
    train_tf = get_train_transforms(cfg["model"]["input_size"], cfg["augmentation"]["train"])
    val_tf = get_val_transforms(cfg["model"]["input_size"], cfg["augmentation"]["val"])

    # Datasets
    data_cfg = cfg["data"]
    train_ds = FingernailHbDataset(data_cfg["root"], data_cfg["metadata_csv"], "train", train_tf)
    val_ds = FingernailHbDataset(data_cfg["root"], data_cfg["metadata_csv"], "val", val_tf)

    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model_cfg = cfg["model"]
    model = HemoLensModel(
        backbone_name=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        hidden_dim=model_cfg["head"]["hidden_dim"],
        dropout=model_cfg["head"]["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_cfg['backbone']}")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Progressive unfreezing: freeze backbone for first N epochs
    freeze_epochs = train_cfg.get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        model.freeze_backbone()
        trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Backbone frozen for {freeze_epochs} epochs ({trainable_after_freeze:,} trainable params)")

    # Label noise for regularization
    label_noise_std = train_cfg.get("label_noise_std", 0.0)
    if label_noise_std > 0:
        print(f"Label noise: σ={label_noise_std:.2f} g/dL")

    # Loss
    loss_cfg = cfg["loss"]
    if loss_cfg["type"] == "huber":
        criterion = nn.HuberLoss(delta=loss_cfg["delta"])
    elif loss_cfg["type"] == "mse":
        criterion = nn.MSELoss()
    elif loss_cfg["type"] == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_cfg['type']}")

    # Optimizer + Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=train_cfg["warmup_epochs"])
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"] - train_cfg["warmup_epochs"]
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[train_cfg["warmup_epochs"]],
    )

    # Checkpoint directory
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        # Progressive unfreezing
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            # Apply lower LR to backbone
            unfreeze_factor = train_cfg.get("unfreeze_lr_factor", 0.1)
            optimizer = AdamW([
                {"params": model.backbone.parameters(), "lr": train_cfg["learning_rate"] * unfreeze_factor},
                {"params": model.head.parameters(), "lr": train_cfg["learning_rate"]},
            ], weight_decay=train_cfg["weight_decay"])
            # Reset scheduler for remaining epochs
            remaining = train_cfg["epochs"] - epoch + 1
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining)
            print(f"\n>>> Backbone unfrozen — backbone LR={train_cfg['learning_rate'] * unfreeze_factor:.2e}")

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{train_cfg['epochs']}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        print(f"{'='*60}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg["logging"]["log_interval"],
            label_noise_std=label_noise_std,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val   Loss: {val_metrics['loss']:.4f} | "
            f"MAE: {val_metrics['mae']:.3f} g/dL | "
            f"RMSE: {val_metrics['rmse']:.3f} | "
            f"R²: {val_metrics['r2']:.4f}"
        )

        # Checkpointing
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            patience_counter = 0
            ckpt_path = save_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": best_mae,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"Saved best model → {ckpt_path} (MAE={best_mae:.3f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{train_cfg['early_stopping_patience']})")

            if patience_counter >= train_cfg["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best validation MAE: {best_mae:.3f} g/dL")


if __name__ == "__main__":
    main()
