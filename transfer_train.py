# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
# 1. 手动设置迁移学习参数
# ============================================================

@dataclass
class TrainConfig:
    # --------------------------------------------------------
    # 迁移学习任务名称与数据路径
    # --------------------------------------------------------
    run_name: str = "transfer_Birmingham_gap9"
    data_root: str = "plan_Birmingham_dataset"
    device: str = "cuda"

    # --------------------------------------------------------
    # 原始预训练模型权重
    # 这里填原来训练好的 best.pth
    # 例如：
    # pretrained_ckpt: str = r"F:\download plan\download plan\checkpoints\plan_zichuang_gap9\best.pth"
    # --------------------------------------------------------
    pretrained_ckpt: str = r"checkpoints\plan_zichuang_gap9\best.pth"

    # --------------------------------------------------------
    # 是否从迁移学习中断点继续训练
    # resume 是继续迁移学习训练，会加载 optimizer / scheduler。
    # pretrained_ckpt 是只加载原模型权重，不加载 optimizer。
    # --------------------------------------------------------
    resume: str = ""

    # --------------------------------------------------------
    # 冻结设置
    # freeze_image_conv=True 表示冻结所有 Conv2d 图像卷积层
    # freeze_batchnorm=True 表示同时冻结 BatchNorm2d
    # --------------------------------------------------------
    freeze_image_conv: bool = True
    freeze_batchnorm: bool = False

    # --------------------------------------------------------
    # 训练参数
    # 迁移学习建议学习率小一些
    # --------------------------------------------------------
    epochs: int = 500
    batch_size: int = 16
    workers: int = 8

    lr: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"      # 可选: "adamw" 或 "sgd"

    # --------------------------------------------------------
    # 学习率与早停
    # --------------------------------------------------------
    print_freq: int = 1
    patience_lr: int = 20
    early_stop_patience: int = 80

    # --------------------------------------------------------
    # AMP 混合精度
    # --------------------------------------------------------
    use_amp: bool = True

    # --------------------------------------------------------
    # 图像尺寸
    # --------------------------------------------------------
    crop_size: int = 224
    resize_size: int = 224

    # --------------------------------------------------------
    # 随机种子
    # --------------------------------------------------------
    seed: int = 3407

    # --------------------------------------------------------
    # 迁移学习阶段建议先关闭图像置零增强
    # --------------------------------------------------------
    zero_image_prob: float = 0.0

    # --------------------------------------------------------
    # 回滚机制
    # --------------------------------------------------------
    rollback_patience: int = 20
    rollback_lr: float = 1e-5

    # --------------------------------------------------------
    # checkpoint 保存路径
    # 新权重会保存到 checkpoint_root / run_name_时间戳
    # 不会覆盖原来的老权重
    # --------------------------------------------------------
    checkpoint_root: str = "checkpoints"
    create_new_run_folder: bool = True


def get_config() -> TrainConfig:
    """
    手动返回训练配置。
    以后只需要改 TrainConfig 里的参数。
    """
    args = TrainConfig()

    # 也可以在这里手动覆盖参数：
    # args.data_root = r"F:\download plan\download plan\plan_Birmingham_dataset"
    # args.pretrained_ckpt = r"F:\download plan\download plan\checkpoints\plan_zichuang_gap9\best.pth"
    # args.run_name = "transfer_Birmingham_gap9"
    # args.lr = 1e-4
    # args.batch_size = 64
    # args.epochs = 500

    return args


# ============================================================
# 2. 新建保存目录，避免覆盖老权重
# ============================================================

def make_save_dir(args: TrainConfig) -> Path:
    """
    创建 checkpoint 保存文件夹。

    规则：
    1. 如果 resume 不为空，说明是在继续训练，继续保存到 resume 所在文件夹。
    2. 如果 create_new_run_folder=True，每次训练自动创建一个带时间戳的新文件夹。
    3. 如果 create_new_run_folder=False，则使用 checkpoint_root / run_name。
       若文件夹已存在，直接报错，避免覆盖旧权重。
    """
    if args.resume:
        resume_path = Path(args.resume)
        save_dir = resume_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SaveDir] Resume 模式，继续保存到：{save_dir}")
        return save_dir

    base_dir = Path(args.checkpoint_root)

    if args.create_new_run_folder:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = base_dir / f"{args.run_name}_{timestamp}"

        # 防止同一秒内重复运行造成文件夹重名
        idx = 1
        original_save_dir = save_dir

        while save_dir.exists():
            save_dir = Path(f"{original_save_dir}_{idx}")
            idx += 1

    else:
        save_dir = base_dir / args.run_name

        if save_dir.exists():
            raise FileExistsError(
                f"保存文件夹已存在：{save_dir}\n"
                f"为避免覆盖旧权重，请修改 run_name，"
                f"或者设置 create_new_run_folder=True。"
            )

    save_dir.mkdir(parents=True, exist_ok=False)

    print(f"[SaveDir] 新权重将保存到：{save_dir}")

    return save_dir


def save_config(args: TrainConfig, save_dir: Path) -> None:
    """
    保存本次训练参数，便于后续追踪实验设置。
    """
    config_path = save_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            asdict(args),
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"[Config] 参数已保存：{config_path}")


# ============================================================
# 3. 加载预训练模型权重
# ============================================================

def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    去掉 DataParallel 产生的 module. 前缀。
    """
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

    return new_state_dict


def load_pretrained_weights_for_transfer(
    model: nn.Module,
    pretrained_ckpt: str,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    """
    迁移学习加载模型权重。

    只加载模型 state_dict。
    不加载 optimizer。
    不加载 scheduler。

    如果部分层尺寸不一致，会自动跳过。
    """
    if pretrained_ckpt is None or str(pretrained_ckpt).strip() == "":
        print("[Transfer] 未设置 pretrained_ckpt，不加载预训练权重。")
        return [], []

    ckpt_path = Path(pretrained_ckpt)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到预训练权重文件：{ckpt_path}")

    print(f"[Transfer] 加载预训练权重：{ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict_keys(state_dict)

    model_state = model.state_dict()

    matched_state = {}
    loaded_keys = []
    skipped_keys = []

    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            matched_state[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)

    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=True)

    print("=" * 80)
    print("[Transfer] 预训练权重加载完成")
    print(f"[Transfer] 成功加载参数数量：{len(loaded_keys)}")
    print(f"[Transfer] 跳过参数数量：{len(skipped_keys)}")

    if skipped_keys:
        print("[Transfer] 前 20 个被跳过的参数：")
        for name in skipped_keys[:20]:
            print(f"  - {name}")

    print("=" * 80)

    return loaded_keys, skipped_keys


# ============================================================
# 4. 冻结图像卷积层
# ============================================================

def freeze_image_convolution_layers(
    model: nn.Module,
    freeze_batchnorm: bool = True,
) -> List[str]:
    """
    冻结图像卷积相关层。

    这里不依赖具体参数名，而是按模块类型冻结：
    - nn.Conv2d
    - 可选 nn.BatchNorm2d / nn.SyncBatchNorm

    对 SimpleCNNGAP9 来说，这通常对应图像卷积分支。
    """
    frozen_names = []

    for module_name, module in model.named_modules():
        is_conv = isinstance(module, nn.Conv2d)
        is_bn = isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm))

        if is_conv or (freeze_batchnorm and is_bn):
            for param_name, param in module.named_parameters(recurse=False):
                param.requires_grad = False

                if module_name:
                    full_name = f"{module_name}.{param_name}"
                else:
                    full_name = param_name

                frozen_names.append(full_name)

            if freeze_batchnorm and is_bn:
                module.eval()

    print("=" * 80)
    print("[Freeze] 已冻结图像卷积相关参数")
    print(f"[Freeze] 冻结参数数量：{len(frozen_names)}")

    for name in frozen_names[:50]:
        print(f"  - {name}")

    if len(frozen_names) > 50:
        print(f"  ... 其余 {len(frozen_names) - 50} 个参数省略")

    print("=" * 80)

    return frozen_names


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """
    训练过程中保持已经冻结的 BatchNorm 为 eval 状态。
    因为 model.train() 会重新打开 BatchNorm 的训练状态。
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            has_trainable_param = any(
                p.requires_grad for p in module.parameters(recurse=False)
            )

            if not has_trainable_param:
                module.eval()


def print_trainable_parameters(model: nn.Module) -> None:
    """
    打印可训练参数和冻结参数数量。
    """
    total_params = 0
    trainable_params = 0

    print("=" * 80)
    print("[Params] 可训练参数列表")

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n

        if param.requires_grad:
            trainable_params += n
            print(f"  [Trainable] {name} | {tuple(param.shape)} | {n}")

    frozen_params = total_params - trainable_params

    print("-" * 80)
    print(f"[Params] 总参数量：{total_params:,}")
    print(f"[Params] 可训练参数量：{trainable_params:,}")
    print(f"[Params] 冻结参数量：{frozen_params:,}")
    print("=" * 80)


# ============================================================
# 5. Optimizer
# ============================================================

def build_optimizer(args: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    """
    只把 requires_grad=True 的参数交给优化器。
    """
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if len(trainable_params) == 0:
        raise RuntimeError("没有可训练参数。请检查冻结设置。")

    optimizer_name = args.optimizer.lower()

    if optimizer_name == "adamw":
        return optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    elif optimizer_name == "sgd":
        return optim.SGD(
            trainable_params,
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
# 6. Train one epoch
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
    freeze_batchnorm: bool,
) -> float:
    model.train()

    if freeze_batchnorm:
        set_frozen_batchnorm_eval(model)

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
# 7. Validate
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
# 8. Main
# ============================================================

def main() -> None:
    args = get_config()

    warnings.filterwarnings("ignore")

    set_seed(args.seed)
    device = get_device(args.device)

    if device.type == "cuda":
        cudnn.benchmark = True

    print("=" * 80)
    print("[Config] Transfer learning configuration")
    print(f"run_name              : {args.run_name}")
    print(f"data_root             : {args.data_root}")
    print(f"pretrained_ckpt       : {args.pretrained_ckpt}")
    print(f"resume                : {args.resume}")
    print(f"device                : {args.device}")
    print(f"epochs                : {args.epochs}")
    print(f"batch_size            : {args.batch_size}")
    print(f"lr                    : {args.lr}")
    print(f"optimizer             : {args.optimizer}")
    print(f"freeze_image_conv     : {args.freeze_image_conv}")
    print(f"freeze_batchnorm      : {args.freeze_batchnorm}")
    print(f"checkpoint_root       : {args.checkpoint_root}")
    print(f"create_new_run_folder : {args.create_new_run_folder}")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. 创建新的保存目录
    # --------------------------------------------------------
    save_dir = make_save_dir(args)
    save_config(args, save_dir)

    log_path = save_dir / "logs.jsonl"
    best_ckpt_path = save_dir / "best.pth"

    # --------------------------------------------------------
    # 2. Data
    # --------------------------------------------------------
    data = ImageTabularTaskData(
        image_folder_path=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        crop_size=args.crop_size,
        resize_size=args.resize_size,
    )

    # --------------------------------------------------------
    # 3. Model
    # --------------------------------------------------------
    model = SimpleCNNGAP9(
        n_tabular=8,
        final_output_dim=1
    ).to(device)

    # --------------------------------------------------------
    # 4. Load pretrained model weights for transfer
    # --------------------------------------------------------
    if args.resume:
        print("[Transfer] 检测到 resume，将从迁移学习 checkpoint 继续训练。")
        print("[Transfer] 此时不重复加载 pretrained_ckpt。")
    else:
        load_pretrained_weights_for_transfer(
            model=model,
            pretrained_ckpt=args.pretrained_ckpt,
            device=device,
        )

    # --------------------------------------------------------
    # 5. Freeze image convolution layers
    # --------------------------------------------------------
    frozen_names = []

    if args.freeze_image_conv:
        frozen_names = freeze_image_convolution_layers(
            model=model,
            freeze_batchnorm=args.freeze_batchnorm,
        )

    print_trainable_parameters(model)

    # --------------------------------------------------------
    # 6. Optimizer / Scheduler / Loss
    # --------------------------------------------------------
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

    start_epoch = 0
    best_val = float("inf")

    # --------------------------------------------------------
    # 7. Resume transfer training
    # --------------------------------------------------------
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
            f"[Resume] Resumed from {args.resume} | "
            f"epoch={start_epoch} | "
            f"best={best_val:.4f}"
        )

    epochs_since_best = 0
    epochs_since_rollback = 0

    # --------------------------------------------------------
    # 8. Training loop
    # --------------------------------------------------------
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
            freeze_batchnorm=args.freeze_batchnorm,
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
            "transfer_learning": True,
            "pretrained_ckpt": str(args.pretrained_ckpt),
            "data_root": str(args.data_root),
            "frozen_layers": frozen_names,
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

        # ----------------------------------------------------
        # Rollback
        # ----------------------------------------------------
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
                "transfer_learning": True,
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
            f"elapsed {time.time() - t0:.1f}s | "
            f"save_dir {save_dir}"
        )

        if epochs_since_best >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best val: {best_val:.4f}"
            )
            break

    print("=" * 80)
    print("Transfer learning done.")
    print(f"Best checkpoint saved in: {best_ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()