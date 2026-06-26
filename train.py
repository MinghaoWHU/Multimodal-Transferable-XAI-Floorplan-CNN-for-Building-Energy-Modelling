# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.simplecnn_gap9.data import ImageTabularTaskData, select_model_tabular
from src.simplecnn_gap9.model import SimpleCNNGAP9
from src.simplecnn_gap9.utils import (
    append_jsonl,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


# ============================================================
# 1. 手动设置训练参数
# ============================================================

@dataclass
class TrainConfig:
    # 实验名称与数据路径
    run_name: str = "plan_zichuang_gap9"
    data_root: str = "plan_1125_dataset"
    device: str = "cuda"

    # 训练参数
    epochs: int = 1000
    batch_size: int = 128
    workers: int = 16

    # 优化器参数
    lr: float = 1e-2
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"      # 可选: "adamw" 或 "sgd"

    # 学习率与早停
    print_freq: int = 1
    patience_lr: int = 20
    early_stop_patience: int = 100

    # AMP 混合精度
    use_amp: bool = True

    # 图像尺寸
    crop_size: int = 224
    resize_size: int = 224

    # 随机种子与继续训练
    seed: int = 3407
    resume: str = ""              # 例如 r"checkpoints/xxx/best.pth"

    # 图像置零增强
    zero_image_prob: float = 0.1

    # 回滚机制
    rollback_patience: int = 20
    rollback_lr: float = 1e-4

    # checkpoint 保存路径
    checkpoint_root: str = "checkpoints"


def get_config() -> TrainConfig:
    """
    手动返回训练配置。
    以后你只需要改 TrainConfig 里的默认值即可。
    """
    args = TrainConfig()

    # 如果你不想改 dataclass 默认值，也可以在这里手动覆盖：
    # args.run_name = "new_experiment_name"
    # args.data_root = r"F:\your_dataset_path"
    # args.device = "cuda"
    # args.epochs = 500
    # args.batch_size = 64
    # args.lr = 1e-3
    # args.optimizer = "adamw"
    # args.use_amp = True
    # args.resume = r""

    return args


# ============================================================
# 2. Optimizer
# ============================================================

def build_optimizer(args: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    optimizer_name = args.optimizer.lower()

    if optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

    else:
        raise ValueError(
            f"未知 optimizer: {args.optimizer}。"
            f"只能设置为 'adamw' 或 'sgd'。"
        )


# ============================================================
# 3. Train one epoch
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
) -> float:
    model.train()

    loss_sum = 0.0
    sample_count = 0

    pbar = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc="Train",
        leave=False
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
                lr=f"{optimizer.param_groups[0]['lr']:.3e}"
            )

    return avg_loss


# ============================================================
# 4. Validate
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
        leave=False
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
# 5. Main
# ============================================================

def main() -> None:
    args = get_config()

    warnings.filterwarnings("ignore")

    set_seed(args.seed)
    device = get_device(args.device)

    if device.type == "cuda":
        cudnn.benchmark = True

    data = ImageTabularTaskData(
        image_folder_path=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        crop_size=args.crop_size,
        resize_size=args.resize_size,
    )

    model = SimpleCNNGAP9(
        n_tabular=8,
        final_output_dim=1
    ).to(device)

    optimizer = build_optimizer(args, model)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=args.patience_lr
    )

    criterion = nn.MSELoss().to(device)

    scaler = GradScaler(
        enabled=args.use_amp and device.type == "cuda"
    )

    save_dir = Path(args.checkpoint_root) / args.run_name
    log_path = save_dir / "logs.jsonl"
    best_ckpt_path = save_dir / "best.pth"

    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        ckpt = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device
        )

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_loss", float("inf")))

        print(
            f"Resumed from {args.resume} | "
            f"epoch={start_epoch} | "
            f"best={best_val:.4f}"
        )

    epochs_since_best = 0
    epochs_since_rollback = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            data.trainloader,
            model,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp=args.use_amp,
            print_freq=args.print_freq,
            zero_image_prob=args.zero_image_prob,
        )

        val_loss = validate(
            data.testloader,
            model,
            criterion,
            device,
            use_amp=args.use_amp,
            print_freq=args.print_freq,
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
            "arch": "SimpleCNNGAP9",
            "state_dict": model.state_dict(),
            "best_loss": float(best_val),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "amp": bool(args.use_amp),
        }

        save_checkpoint(
            save_dir,
            state,
            is_best=is_best
        )

        if epochs_since_rollback >= args.rollback_patience:
            print(f"\n[Rollback] No improvement for {args.rollback_patience} epochs.")

            if best_ckpt_path.exists():
                load_checkpoint(
                    str(best_ckpt_path),
                    model,
                    optimizer=optimizer,
                    map_location=device
                )
                print(f"[Rollback] Loaded best checkpoint: {best_ckpt_path}")
            else:
                print("[Rollback] best.pth not found. Skip loading model.")

            for param_group in optimizer.param_groups:
                param_group["lr"] = args.rollback_lr

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=args.patience_lr
            )

            epochs_since_rollback = 0

            print(f"[Rollback] Reset LR to {args.rollback_lr:.1e}\n")

        append_jsonl(
            log_path,
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "best_val": float(best_val),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "time_sec": round(time.time() - t0, 2),
            },
        )

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"best {best_val:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.3e} | "
            f"elapsed {time.time() - t0:.1f}s"
        )

        if epochs_since_best >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best val: {best_val:.4f}"
            )
            break

    print("Training done.")


if __name__ == "__main__":
    main()