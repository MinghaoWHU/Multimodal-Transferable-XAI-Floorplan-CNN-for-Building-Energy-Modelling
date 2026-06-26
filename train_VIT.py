# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.simplecnn_gap9.data import ImageTabularTaskData, select_model_tabular
from src.simplecnn_gap9.utils import append_jsonl, get_device, set_seed
from src.vit_multimodal.model import MultimodalViTRegressor


# ============================================================
# 1. 参数设置区
#    不使用 argparse，所有参数都在这里直接修改
# ============================================================

RUN_NAME = "plan_zichuang_vit"
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

# 如果需要继续训练，就填写 checkpoint 路径
# 例如：RESUME = "checkpoints/plan_zichuang_vit/VIT_best.pth"
RESUME = "checkpoints/plan_zichuang_vit/VIT_last.pth"

ZERO_IMAGE_PROB = 0.1

# True 表示和你原始代码一致：图像置零时，target 也置零
# False 表示只进行图像缺失增强，不改 target
ZERO_IMAGE_TARGET = True

ROLLBACK_PATIENCE = 20
ROLLBACK_LR = 1e-5

CHECKPOINT_ROOT = "checkpoints"

# ViT 设置
PRETRAINED = False             # True 会尝试加载 ImageNet 预训练权重
FREEZE_BACKBONE = False        # True 表示冻结 ViT 主干，只训练融合层
DROPOUT = 0

N_TABULAR = 8
FINAL_OUTPUT_DIM = 1


# ============================================================
# 2. 优化器
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
# 3. checkpoint 保存与读取
# ============================================================

def save_vit_checkpoint(
    save_dir: Path,
    state: dict,
    is_best: bool,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    last_path = save_dir / "VIT_last.pth"
    best_path = save_dir / "VIT_best.pth"

    torch.save(state, last_path)

    if is_best:
        torch.save(state, best_path)


def load_vit_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[ReduceLROnPlateau] = None,
    map_location: torch.device | str = "cpu",
) -> dict:
    try:
        checkpoint = torch.load(
            ckpt_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            ckpt_path,
            map_location=map_location,
        )

    model.load_state_dict(checkpoint["state_dict"], strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception:
            print("[Warning] Scheduler state was not loaded.")

    return checkpoint


# ============================================================
# 4. 训练一个 epoch
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
# 5. 验证
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
# 6. 主函数
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore")

    set_seed(SEED)
    device = get_device(DEVICE)

    if device.type == "cuda":
        cudnn.benchmark = True

    data = ImageTabularTaskData(
        image_folder_path=DATA_ROOT,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        crop_size=CROP_SIZE,
        resize_size=RESIZE_SIZE,
    )

    model = MultimodalViTRegressor(
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
    best_ckpt_path = save_dir / "VIT_best.pth"

    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val = float("inf")

    if RESUME:
        ckpt = load_vit_checkpoint(
            RESUME,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_loss", float("inf")))

        print(
            f"Resumed from {RESUME} | "
            f"epoch={start_epoch} | "
            f"best={best_val:.4f}"
        )

    epochs_since_best = 0
    epochs_since_rollback = 0

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            data.trainloader,
            model,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp=USE_AMP,
            print_freq=PRINT_FREQ,
            zero_image_prob=ZERO_IMAGE_PROB,
            zero_image_target=ZERO_IMAGE_TARGET,
        )

        val_loss = validate(
            data.testloader,
            model,
            criterion,
            device,
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
            "arch": "MultimodalViTRegressor",
            "state_dict": model.state_dict(),
            "best_loss": float(best_val),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "amp": bool(USE_AMP),
            "model_config": {
                "n_tabular": N_TABULAR,
                "final_output_dim": FINAL_OUTPUT_DIM,
                "pretrained": bool(PRETRAINED),
                "freeze_backbone": bool(FREEZE_BACKBONE),
                "dropout": float(DROPOUT),
            },
        }

        save_vit_checkpoint(
            save_dir=save_dir,
            state=state,
            is_best=is_best,
        )

        if epochs_since_rollback >= ROLLBACK_PATIENCE:
            print(f"\n[Rollback] No improvement for {ROLLBACK_PATIENCE} epochs.")

            if best_ckpt_path.exists():
                load_vit_checkpoint(
                    str(best_ckpt_path),
                    model,
                    optimizer=optimizer,
                    scheduler=None,
                    map_location=device,
                )
                print(f"[Rollback] Loaded best checkpoint: {best_ckpt_path}")
            else:
                print("[Rollback] VIT_best.pth not found. Skip loading model.")

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
                "checkpoint_last": str(save_dir / "VIT_last.pth"),
                "checkpoint_best": str(save_dir / "VIT_best.pth"),
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
    print(f"Last checkpoint: {save_dir / 'VIT_last.pth'}")
    print(f"Best checkpoint: {save_dir / 'VIT_best.pth'}")


if __name__ == "__main__":
    main()