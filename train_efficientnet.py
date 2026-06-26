# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.simplecnn_gap9.data import ImageTabularTaskData, select_model_tabular
from src.simplecnn_gap9.utils import append_jsonl, get_device, set_seed


# ============================================================
# 1. 参数设置区
# ============================================================

RUN_NAME = "plan_zichuang_efficientnet"
DATA_ROOT = "plan_1125_dataset"
DEVICE = "cuda"

EPOCHS = 1000
BATCH_SIZE = 64
WORKERS = 16

LR = 1e-4
WEIGHT_DECAY = 1e-3
MOMENTUM = 0.9
OPTIMIZER_NAME = "adamw"       # "adamw" or "sgd"

PRINT_FREQ = 1
PATIENCE_LR = 10
EARLY_STOP_PATIENCE = 100

USE_AMP = True

CROP_SIZE = 224
RESIZE_SIZE = 224

SEED = 3407

# ============================================================
# Resume 设置
# ============================================================

# 首次训练：保持为空字符串
RESUME = "checkpoints/plan_zichuang_efficientnet/EfficientNet_last.pth"

# 继续训练时改成：
# RESUME = "checkpoints/plan_zichuang_efficientnet/EfficientNet_last.pth"
# 或：
# RESUME = "checkpoints/plan_zichuang_efficientnet/EfficientNet_best.pth"

STRICT_LOAD = True
LOAD_OPTIMIZER_STATE = True

# ============================================================
# 数据增强设置
# ============================================================

ZERO_IMAGE_PROB = 0.1

# True 表示图像置零时，target 也置零。
# False 表示只进行图像缺失增强，不改 target。
ZERO_IMAGE_TARGET = True

# ============================================================
# Rollback 设置
# ============================================================

ROLLBACK_PATIENCE = 20
ROLLBACK_LR = 1e-5

CHECKPOINT_ROOT = "checkpoints"

# ============================================================
# EfficientNet 设置
# ============================================================

# 可选：
# efficientnet_b0
# efficientnet_b1
# efficientnet_b2
# efficientnet_v2_s
EFFICIENTNET_NAME = "efficientnet_b0"

# 首次训练建议 False，避免因无法联网下载权重而报错。
PRETRAINED = False

# True 表示冻结 EfficientNet 主干，只训练融合层。
FREEZE_BACKBONE = False

DROPOUT = 0

N_TABULAR = 8
FINAL_OUTPUT_DIM = 1


# ============================================================
# 2. EfficientNet 多模态模型
# ============================================================

def build_efficientnet_backbone(
    name: str = "efficientnet_b0",
    pretrained: bool = False,
) -> Tuple[nn.Module, int]:
    """
    构建 EfficientNet 主干。
    返回：
    backbone, image_feature_dim
    """
    import torchvision.models as models

    name = name.lower()

    try:
        if name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)

        elif name == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b1(weights=weights)

        elif name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b2(weights=weights)

        elif name == "efficientnet_v2_s":
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_v2_s(weights=weights)

        else:
            raise ValueError(f"Unsupported EfficientNet name: {name}")

    except Exception:
        # 兼容旧版本 torchvision
        if name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=pretrained)
        elif name == "efficientnet_b1":
            backbone = models.efficientnet_b1(pretrained=pretrained)
        elif name == "efficientnet_b2":
            backbone = models.efficientnet_b2(pretrained=pretrained)
        elif name == "efficientnet_v2_s":
            backbone = models.efficientnet_v2_s(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported EfficientNet name: {name}")

    if not hasattr(backbone, "classifier"):
        raise ValueError("EfficientNet backbone does not have classifier attribute.")

    if isinstance(backbone.classifier, nn.Sequential):
        image_feature_dim = backbone.classifier[-1].in_features
    else:
        image_feature_dim = backbone.classifier.in_features

    # 去掉原始分类头，只保留图像特征
    backbone.classifier = nn.Identity()

    return backbone, image_feature_dim


class MultimodalEfficientNetRegressor(nn.Module):
    """
    EfficientNet image branch + tabular branch + fusion regressor.
    """

    def __init__(
        self,
        efficientnet_name: str = "efficientnet_b0",
        n_tabular: int = 8,
        final_output_dim: int = 1,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.backbone, image_feature_dim = build_efficientnet_backbone(
            name=efficientnet_name,
            pretrained=pretrained,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        tabular_dim = 64

        self.tabular_net = nn.Sequential(
            nn.Linear(n_tabular, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, tabular_dim),
            nn.BatchNorm1d(tabular_dim),
            nn.ReLU(inplace=True),
        )

        fusion_dim = image_feature_dim + tabular_dim

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, final_output_dim),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        image_feature = self.backbone(image)
        tabular_feature = self.tabular_net(tabular)

        fused = torch.cat([image_feature, tabular_feature], dim=1)
        output = self.regressor(fused)

        return output


# ============================================================
# 3. 优化器
# ============================================================

def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if OPTIMIZER_NAME.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

    return optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )


# ============================================================
# 4. checkpoint 工具函数
# ============================================================

def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    去掉 DataParallel 训练时可能产生的 module. 前缀。
    """
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value

    return cleaned


def save_efficientnet_checkpoint(
    save_dir: Path,
    state: dict,
    is_best: bool,
) -> None:
    """
    保存 EfficientNet checkpoint。
    始终保存 last。
    如果是当前最优模型，则同时保存 best。
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    last_path = save_dir / "EfficientNet_last.pth"
    best_path = save_dir / "EfficientNet_best.pth"

    torch.save(state, last_path)

    if is_best:
        torch.save(state, best_path)


def load_efficientnet_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[ReduceLROnPlateau] = None,
    map_location: torch.device | str = "cpu",
    strict: bool = True,
    load_optimizer_state: bool = True,
) -> dict:
    """
    加载 EfficientNet checkpoint。
    支持从 last 或 best 继续训练。
    """
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        checkpoint = torch.load(
            str(ckpt_path),
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            str(ckpt_path),
            map_location=map_location,
        )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict(state_dict)

    load_info = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        print(f"[Load] {ckpt_path}")
        print(f"Missing keys: {load_info.missing_keys}")
        print(f"Unexpected keys: {load_info.unexpected_keys}")

    if (
        load_optimizer_state
        and optimizer is not None
        and isinstance(checkpoint, dict)
        and "optimizer" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("[Resume] Optimizer state loaded.")

    if (
        load_optimizer_state
        and scheduler is not None
        and isinstance(checkpoint, dict)
        and "scheduler" in checkpoint
    ):
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("[Resume] Scheduler state loaded.")
        except Exception:
            print("[Warning] Scheduler state was not loaded.")

    return checkpoint


# ============================================================
# 5. 训练一个 epoch
# ============================================================

def train_one_epoch(
    loader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    print_freq: int,
    zero_image_prob: float,
    zero_image_target: bool,
) -> float:
    model.train()

    loss_sum = 0.0
    sample_count = 0

    pbar = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc="Train",
        leave=False,
    )

    for step, (images, tabular, target, img_path) in pbar:
        images = images.to(device, non_blocking=True)
        tabular = select_model_tabular(tabular).to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float().view(-1, 1)

        if zero_image_prob > 0:
            batch_size = images.size(0)
            zero_mask = torch.rand(batch_size, device=device) < zero_image_prob

            if zero_mask.any():
                images[zero_mask] = 0.0

                if zero_image_target:
                    target[zero_mask] = 0.0

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp and device.type == "cuda"):
            output = model(images, tabular)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size

        avg_loss = loss_sum / max(sample_count, 1)

        if (step % print_freq == 0) or (step == len(loader)):
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.3e}",
            )

    return avg_loss


# ============================================================
# 6. 验证
# ============================================================

@torch.no_grad()
def validate(
    loader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    print_freq: int,
) -> float:
    model.eval()

    loss_sum = 0.0
    sample_count = 0

    pbar = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc="Val",
        leave=False,
    )

    for step, (images, tabular, target, img_path) in pbar:
        images = images.to(device, non_blocking=True)
        tabular = select_model_tabular(tabular).to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float().view(-1, 1)

        with autocast(enabled=use_amp and device.type == "cuda"):
            output = model(images, tabular)
            loss = criterion(output, target)

        batch_size = images.size(0)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size

        avg_loss = loss_sum / max(sample_count, 1)

        if (step % print_freq == 0) or (step == len(loader)):
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return avg_loss


# ============================================================
# 7. 主函数
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore")

    set_seed(SEED)

    device = get_device(DEVICE)

    if device.type == "cuda":
        cudnn.benchmark = True

    print("=" * 80)
    print("Loading dataset...")
    print("=" * 80)

    data = ImageTabularTaskData(
        image_folder_path=DATA_ROOT,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        crop_size=CROP_SIZE,
        resize_size=RESIZE_SIZE,
    )

    print("=" * 80)
    print("Building MultimodalEfficientNetRegressor...")
    print("=" * 80)

    model = MultimodalEfficientNetRegressor(
        efficientnet_name=EFFICIENTNET_NAME,
        n_tabular=N_TABULAR,
        final_output_dim=FINAL_OUTPUT_DIM,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
        dropout=DROPOUT,
    ).to(device)

    optimizer = build_optimizer(model)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=PATIENCE_LR,
    )

    criterion = nn.MSELoss().to(device)

    scaler = GradScaler(
        enabled=USE_AMP and device.type == "cuda",
    )

    save_dir = Path(CHECKPOINT_ROOT) / RUN_NAME
    log_path = save_dir / "logs.jsonl"
    best_ckpt_path = save_dir / "EfficientNet_best.pth"

    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val = float("inf")

    # ========================================================
    # Resume
    # ========================================================
    if RESUME:
        ckpt = load_efficientnet_checkpoint(
            ckpt_path=RESUME,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            strict=STRICT_LOAD,
            load_optimizer_state=LOAD_OPTIMIZER_STATE,
        )

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_loss", float("inf")))

        print(
            f"[Resume] Loaded checkpoint: {RESUME} | "
            f"epoch={start_epoch} | "
            f"best_val={best_val:.4f}"
        )

    else:
        print("[Start] First training run. No checkpoint is loaded.")

    epochs_since_best = 0
    epochs_since_rollback = 0

    # ========================================================
    # Training loop
    # ========================================================
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            loader=data.trainloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=USE_AMP,
            print_freq=PRINT_FREQ,
            zero_image_prob=ZERO_IMAGE_PROB,
            zero_image_target=ZERO_IMAGE_TARGET,
        )

        val_loss = validate(
            loader=data.testloader,
            model=model,
            criterion=criterion,
            device=device,
            use_amp=USE_AMP,
            print_freq=PRINT_FREQ,
        )

        scheduler.step(val_loss)

        is_best = val_loss < best_val

        if is_best:
            best_val = val_loss
            epochs_since_best = 0
            epochs_since_rollback = 0
        else:
            epochs_since_best += 1
            epochs_since_rollback += 1

        state = {
            "epoch": epoch,
            "arch": "MultimodalEfficientNetRegressor",
            "state_dict": model.state_dict(),
            "best_loss": float(best_val),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "amp": bool(USE_AMP),
            "model_config": {
                "efficientnet_name": EFFICIENTNET_NAME,
                "n_tabular": N_TABULAR,
                "final_output_dim": FINAL_OUTPUT_DIM,
                "pretrained": bool(PRETRAINED),
                "freeze_backbone": bool(FREEZE_BACKBONE),
                "dropout": float(DROPOUT),
            },
        }

        save_efficientnet_checkpoint(
            save_dir=save_dir,
            state=state,
            is_best=is_best,
        )

        # ====================================================
        # Rollback
        # ====================================================
        if epochs_since_rollback >= ROLLBACK_PATIENCE:
            print(f"\n[Rollback] No improvement for {ROLLBACK_PATIENCE} epochs.")

            if best_ckpt_path.exists():
                load_efficientnet_checkpoint(
                    ckpt_path=str(best_ckpt_path),
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    map_location=device,
                    strict=STRICT_LOAD,
                    load_optimizer_state=False,
                )
                print(f"[Rollback] Loaded best checkpoint: {best_ckpt_path}")
            else:
                print("[Rollback] EfficientNet_best.pth not found. Skip loading model.")

            for param_group in optimizer.param_groups:
                param_group["lr"] = ROLLBACK_LR

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=PATIENCE_LR,
            )

            epochs_since_rollback = 0

            print(f"[Rollback] Reset LR to {ROLLBACK_LR:.1e}\n")

        append_jsonl(
            log_path,
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "best_val": float(best_val),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "time_sec": round(time.time() - t0, 2),
                "checkpoint_last": str(save_dir / "EfficientNet_last.pth"),
                "checkpoint_best": str(save_dir / "EfficientNet_best.pth"),
            },
        )

        print(
            f"[{epoch:03d}/{EPOCHS}] "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"best {best_val:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.3e} | "
            f"elapsed {time.time() - t0:.1f}s"
        )

        if epochs_since_best >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best val: {best_val:.4f}"
            )
            break

    print("Training done.")
    print(f"Last checkpoint: {save_dir / 'EfficientNet_last.pth'}")
    print(f"Best checkpoint: {save_dir / 'EfficientNet_best.pth'}")


if __name__ == "__main__":
    main()