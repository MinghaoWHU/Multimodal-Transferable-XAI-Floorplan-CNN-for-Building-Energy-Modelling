# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.simplecnn_gap9.data import ImageTabularTaskData, select_model_tabular
from src.simplecnn_gap9.utils import get_device, set_seed
from src.vit_multimodal.model import MultimodalViTRegressor


# ============================================================
# 1. 参数设置区
# ============================================================

DATA_ROOT = "plan_1125_dataset"
DEVICE = "cuda"

BATCH_SIZE = 64
WORKERS = 16

CROP_SIZE = 224
RESIZE_SIZE = 224

SEED = 3407
USE_AMP = True

# ViT checkpoint
VIT_CKPT = "checkpoints/plan_zichuang_vit/VIT_best.pth"

# 输出目录
OUTPUT_DIR = "vit_test_results"

# ViT 模型参数，需要和训练时一致
PRETRAINED = False
FREEZE_BACKBONE = False
DROPOUT = 0.20

N_TABULAR = 8
FINAL_OUTPUT_DIM = 1

# 是否严格加载
STRICT_LOAD = True

# CVRMSE 是否乘以 100
CVRMSE_PERCENT = True

# 图像保存设置
FIG_DPI = 600
SAVE_PDF = True
SAVE_PNG = True


# ============================================================
# 2. 工具函数
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


def load_checkpoint_to_model(
    model: nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict: bool = True,
) -> None:
    """
    加载 ViT checkpoint。
    兼容训练脚本中保存的格式：
    {
        "epoch": ...,
        "arch": "MultimodalViTRegressor",
        "state_dict": model.state_dict(),
        ...
    }
    """
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        checkpoint = torch.load(
            str(ckpt_path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            str(ckpt_path),
            map_location=device,
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

    if isinstance(checkpoint, dict):
        epoch = checkpoint.get("epoch", None)
        best_loss = checkpoint.get("best_loss", None)

        if epoch is not None:
            print(f"Loaded checkpoint epoch: {epoch}")

        if best_loss is not None:
            print(f"Loaded checkpoint best_loss: {best_loss}")


def get_model_cost_info(
    model: nn.Module,
    ckpt_path: str,
) -> Dict[str, float]:
    """
    统计模型成本信息。
    不包含训练时间。

    包括：
    1. checkpoint 文件大小
    2. 模型总参数量
    3. 可训练参数量
    """
    ckpt_path = Path(ckpt_path)

    if ckpt_path.exists():
        model_size_byte = ckpt_path.stat().st_size
    else:
        model_size_byte = np.nan

    if np.isnan(model_size_byte):
        model_size_kb = np.nan
        model_size_mb = np.nan
    else:
        model_size_kb = model_size_byte / 1024.0
        model_size_mb = model_size_kb / 1024.0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "Model_Size_Byte": float(model_size_byte),
        "Model_Size_KB": float(model_size_kb),
        "Model_Size_MB": float(model_size_mb),
        "Total_Params": int(total_params),
        "Trainable_Params": int(trainable_params),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cvrmse_percent: bool = True,
) -> Dict[str, float]:
    """
    计算回归评价指标。
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    residual = y_true - y_pred

    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    mean_true = float(np.mean(y_true))

    if abs(mean_true) < 1e-12:
        cvrmse = np.nan
    else:
        cvrmse = rmse / abs(mean_true)

    if cvrmse_percent and not np.isnan(cvrmse):
        cvrmse = cvrmse * 100.0

    return {
        "R2": float(r2),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "CVRMSE": float(cvrmse),
        "MAE": float(mae),
    }


def compute_accuracy_and_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    计算 ±10%, ±20%, ±30% 范围内的样本比例，以及 MAPE。
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    relative_error = np.abs(y_pred - y_true) / denom

    acc_10 = float(np.mean(relative_error <= 0.10) * 100.0)
    acc_20 = float(np.mean(relative_error <= 0.20) * 100.0)
    acc_30 = float(np.mean(relative_error <= 0.30) * 100.0)
    mape = float(np.mean(relative_error) * 100.0)

    return {
        "ACC10": acc_10,
        "ACC20": acc_20,
        "ACC30": acc_30,
        "MAPE": mape,
    }


@torch.no_grad()
def predict_loader(
    loader,
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], float, float]:
    """
    对 dataloader 进行预测，并统计模型调用时间。

    这里统计的是 forward 推理时间。
    不包括数据读取、指标计算和绘图时间。
    """
    model.eval()

    y_true_list = []
    y_pred_list = []
    path_list = []

    total_inference_time = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=desc, leave=False)

    for images, tabular, target, img_path in pbar:
        images = images.to(device, non_blocking=True)
        tabular = select_model_tabular(tabular).to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float().view(-1, 1)

        batch_size = images.shape[0]

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with autocast(enabled=use_amp and device.type == "cuda"):
            output = model(images, tabular)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        total_inference_time += end_time - start_time
        total_samples += batch_size

        y_true_list.append(target.detach().cpu().numpy())
        y_pred_list.append(output.detach().cpu().numpy())

        if isinstance(img_path, (list, tuple)):
            path_list.extend([str(p) for p in img_path])
        else:
            path_list.extend([str(img_path)])

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    if total_samples > 0:
        single_sample_time = total_inference_time / total_samples
    else:
        single_sample_time = np.nan

    return y_true, y_pred, path_list, total_inference_time, single_sample_time


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    output_dir: Path,
    model_name: str = "MultimodalViT",
) -> None:
    """
    绘制预测值-真实值对照散点图。
    包含：
    1. 散点
    2. y = x 完美预测线
    3. ±10%, ±20%, ±30% 预测区间线
    4. 左上角指标文本
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    metrics = compute_metrics(y_true, y_pred, cvrmse_percent=CVRMSE_PERCENT)
    acc = compute_accuracy_and_mape(y_true, y_pred)

    valid_max = max(
        float(np.nanmax(y_true)),
        float(np.nanmax(y_pred)),
    )

    max_lim = np.ceil(valid_max / 50.0) * 50.0
    if max_lim < 100:
        max_lim = 100

    x_line = np.linspace(0, max_lim, 500)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=FIG_DPI)

    # --------------------------------------------------------
    # 预测区间填充
    # --------------------------------------------------------
    ax.fill_between(
        x_line,
        x_line * 0.70,
        x_line * 1.30,
        alpha=0.10,
        color="red",
        linewidth=0,
    )
    ax.fill_between(
        x_line,
        x_line * 0.80,
        x_line * 1.20,
        alpha=0.12,
        color="red",
        linewidth=0,
    )
    ax.fill_between(
        x_line,
        x_line * 0.90,
        x_line * 1.10,
        alpha=0.14,
        color="red",
        linewidth=0,
    )

    # --------------------------------------------------------
    # 散点
    # --------------------------------------------------------
    ax.scatter(
        y_true,
        y_pred,
        s=16,
        alpha=0.10,
        color="#1f77b4",
        edgecolors="none",
        label="Predictions",
        zorder=3,
    )

    # --------------------------------------------------------
    # ±30%, ±20%, ±10% 边界线
    # --------------------------------------------------------
    ax.plot(
        x_line,
        x_line * 1.30,
        color="red",
        linestyle=":",
        linewidth=1.8,
        label="±30% Prediction Bounds",
        zorder=2,
    )
    ax.plot(
        x_line,
        x_line * 0.70,
        color="red",
        linestyle=":",
        linewidth=1.8,
        label="_nolegend_",
        zorder=2,
    )

    ax.plot(
        x_line,
        x_line * 1.20,
        color="red",
        linestyle=(0, (3, 2)),
        linewidth=1.8,
        label="±20% Prediction Bounds",
        zorder=2,
    )
    ax.plot(
        x_line,
        x_line * 0.80,
        color="red",
        linestyle=(0, (3, 2)),
        linewidth=1.8,
        label="_nolegend_",
        zorder=2,
    )

    ax.plot(
        x_line,
        x_line * 1.10,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="±10% Prediction Bounds",
        zorder=2,
    )
    ax.plot(
        x_line,
        x_line * 0.90,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="_nolegend_",
        zorder=2,
    )

    # --------------------------------------------------------
    # 完美预测线
    # --------------------------------------------------------
    ax.plot(
        x_line,
        x_line,
        color="red",
        linestyle="-",
        linewidth=2.0,
        label="Perfect Prediction Line",
        zorder=4,
    )

    # --------------------------------------------------------
    # 指标文本
    # --------------------------------------------------------
    metric_text = (
        rf"$R^2$: {metrics['R2']:.4f}" + "\n"
        f"Accuracy ±30%: {acc['ACC30']:.2f}%\n"
        f"Accuracy ±20%: {acc['ACC20']:.2f}%\n"
        f"Accuracy ±10%: {acc['ACC10']:.2f}%\n"
        f"MAPE: {acc['MAPE']:.2f}%"
    )

    ax.text(
        0.045,
        0.965,
        metric_text,
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="top",
        horizontalalignment="left",
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.65,
            boxstyle="round,pad=0.25",
        ),
    )

    # --------------------------------------------------------
    # 坐标轴与标题
    # --------------------------------------------------------
    ax.set_title(
        f"Prediction vs True Values ({split_name} Set)",
        fontsize=22,
        pad=14,
    )

    ax.set_xlabel(
        r"True EUI (kWh/m$^2$)",
        fontsize=22,
        labelpad=10,
    )

    ax.set_ylabel(
        r"Predicted EUI (kWh/m$^2$)",
        fontsize=22,
        labelpad=10,
    )

    ax.set_xlim(0, max_lim)
    ax.set_ylim(0, max_lim)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=18,
        width=1.2,
        length=6,
    )

    ax.grid(
        True,
        linestyle="-",
        linewidth=0.8,
        alpha=0.45,
    )

    legend = ax.legend(
        loc="lower right",
        fontsize=15,
        frameon=True,
        framealpha=0.88,
        fancybox=True,
        borderpad=0.8,
        labelspacing=0.55,
        handlelength=2.8,
        markerscale=1.8,
    )

    legend.get_frame().set_edgecolor("#bfbfbf")
    legend.get_frame().set_linewidth(0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()

    safe_split = split_name.lower()
    png_path = output_dir / f"{model_name}_{safe_split}_prediction_scatter.png"
    pdf_path = output_dir / f"{model_name}_{safe_split}_prediction_scatter.pdf"

    if SAVE_PNG:
        fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")

    if SAVE_PDF:
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved scatter plot: {png_path}")

    if SAVE_PDF:
        print(f"Saved scatter plot: {pdf_path}")


def evaluate_vit_model(
    model_name: str,
    model: nn.Module,
    ckpt_path: str,
    train_loader,
    test_loader,
    device: torch.device,
    output_dir: Path,
) -> List[Dict[str, float]]:
    """
    评估 ViT 模型。
    输出训练集和测试集指标。
    cost-benefit 表只使用 Test 结果。
    """
    print("\n" + "=" * 80)
    print(f"Evaluating model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 80)

    load_checkpoint_to_model(
        model=model,
        ckpt_path=ckpt_path,
        device=device,
        strict=STRICT_LOAD,
    )

    model = model.to(device)
    model.eval()

    results = []

    for split_name, loader in [
        ("Training", train_loader),
        ("Test", test_loader),
    ]:
        (
            y_true,
            y_pred,
            img_paths,
            inference_total_time,
            inference_single_time,
        ) = predict_loader(
            loader=loader,
            model=model,
            device=device,
            use_amp=USE_AMP,
            desc=f"{model_name} {split_name}",
        )

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            cvrmse_percent=CVRMSE_PERCENT,
        )

        acc = compute_accuracy_and_mape(
            y_true=y_true,
            y_pred=y_pred,
        )

        row = {
            "Model": model_name,
            "Split": split_name,
            "N": int(len(y_true)),

            # Cost 使用 Test 的这两个字段
            "Inference_Total_Time_s": float(inference_total_time),
            "Inference_Single_Time_s": float(inference_single_time),

            # Benefit
            **metrics,
            **acc,
        }

        results.append(row)

        pred_df = pd.DataFrame({
            "img_path": img_paths,
            "y_true": y_true.reshape(-1),
            "y_pred": y_pred.reshape(-1),
            "residual": y_true.reshape(-1) - y_pred.reshape(-1),
            "absolute_error": np.abs(y_true.reshape(-1) - y_pred.reshape(-1)),
        })

        pred_path = output_dir / f"{model_name}_{split_name}_predictions.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        print(
            f"{model_name} | {split_name} | "
            f"N={len(y_true)} | "
            f"Inference_Total_Time={inference_total_time:.6f}s | "
            f"Inference_Single_Time={inference_single_time:.10f}s/sample | "
            f"R2={metrics['R2']:.4f} | "
            f"MSE={metrics['MSE']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"CVRMSE={metrics['CVRMSE']:.4f} | "
            f"MAE={metrics['MAE']:.4f} | "
            f"ACC30={acc['ACC30']:.2f}% | "
            f"ACC20={acc['ACC20']:.2f}% | "
            f"ACC10={acc['ACC10']:.2f}% | "
            f"MAPE={acc['MAPE']:.2f}%"
        )

        plot_prediction_scatter(
            y_true=y_true,
            y_pred=y_pred,
            split_name=split_name,
            output_dir=output_dir,
            model_name=model_name,
        )

    return results


def build_cost_benefit_table(
    result_df: pd.DataFrame,
    model: nn.Module,
    ckpt_path: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    构建最终 cost-benefit 表。
    不包含训练时间。
    只使用 Test split 的结果。
    """
    test_df = result_df[result_df["Split"] == "Test"].copy()

    if test_df.empty:
        raise ValueError("result_df 中没有 Test 结果，无法生成 cost-benefit 表。")

    test_row = test_df.iloc[0]

    cost_info = get_model_cost_info(
        model=model,
        ckpt_path=ckpt_path,
    )

    cost_benefit_df = pd.DataFrame([
        {
            "Model": test_row["Model"],

            # =================================================
            # Cost，不包含训练时间
            # =================================================
            "Cost_Test_Total_Call_Time_s": test_row["Inference_Total_Time_s"],
            "Cost_Test_Single_Call_Time_s": test_row["Inference_Single_Time_s"],
            "Cost_Model_Size_Byte": cost_info["Model_Size_Byte"],
            "Cost_Model_Size_KB": cost_info["Model_Size_KB"],
            "Cost_Model_Size_MB": cost_info["Model_Size_MB"],
            "Cost_Total_Params": cost_info["Total_Params"],
            "Cost_Trainable_Params": cost_info["Trainable_Params"],

            # =================================================
            # Benefit，测试集预测收益
            # =================================================
            "Benefit_Test_MAE": test_row["MAE"],
            "Benefit_Test_MSE": test_row["MSE"],
            "Benefit_Test_RMSE": test_row["RMSE"],
            "Benefit_Test_CVRMSE": test_row["CVRMSE"],
            "Benefit_Test_R2": test_row["R2"],
            "Benefit_ACC30": test_row["ACC30"],
            "Benefit_ACC20": test_row["ACC20"],
            "Benefit_ACC10": test_row["ACC10"],
            "Benefit_MAPE": test_row["MAPE"],
        }
    ])

    cost_benefit_df = cost_benefit_df.round({
        "Cost_Test_Total_Call_Time_s": 6,
        "Cost_Test_Single_Call_Time_s": 10,
        "Cost_Model_Size_KB": 4,
        "Cost_Model_Size_MB": 6,
        "Benefit_Test_MAE": 6,
        "Benefit_Test_MSE": 6,
        "Benefit_Test_RMSE": 6,
        "Benefit_Test_CVRMSE": 6,
        "Benefit_Test_R2": 6,
        "Benefit_ACC30": 6,
        "Benefit_ACC20": 6,
        "Benefit_ACC10": 6,
        "Benefit_MAPE": 6,
    })

    cost_benefit_path = output_dir / "cost_benefit_summary.csv"
    cost_benefit_df.to_csv(cost_benefit_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Final cost-benefit summary")
    print("=" * 80)
    print(cost_benefit_df.to_string(index=False))

    print("\nSaved cost-benefit table:")
    print(f"Cost-benefit summary: {cost_benefit_path}")

    return cost_benefit_df


# ============================================================
# 3. 主函数
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore")

    set_seed(SEED)

    device = get_device(DEVICE)

    if device.type == "cuda":
        cudnn.benchmark = True

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print("Building MultimodalViTRegressor...")
    print("=" * 80)

    vit_model = MultimodalViTRegressor(
        n_tabular=N_TABULAR,
        final_output_dim=FINAL_OUTPUT_DIM,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
        dropout=DROPOUT,
    )

    all_results = evaluate_vit_model(
        model_name="MultimodalViT",
        model=vit_model,
        ckpt_path=VIT_CKPT,
        train_loader=data.trainloader,
        test_loader=data.testloader,
        device=device,
        output_dir=output_dir,
    )

    # --------------------------------------------------------
    # 保存训练集和测试集完整指标
    # --------------------------------------------------------
    result_df = pd.DataFrame(all_results)

    round_cols = [
        "Inference_Total_Time_s",
        "Inference_Single_Time_s",
        "R2",
        "MSE",
        "RMSE",
        "CVRMSE",
        "MAE",
        "ACC30",
        "ACC20",
        "ACC10",
        "MAPE",
    ]

    for col in round_cols:
        if col in result_df.columns:
            if col == "Inference_Single_Time_s":
                result_df[col] = result_df[col].round(10)
            else:
                result_df[col] = result_df[col].round(6)

    summary_path = output_dir / "vit_metrics_summary.csv"
    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Final metrics summary")
    print("=" * 80)
    print(result_df.to_string(index=False))

    # --------------------------------------------------------
    # 保存最终 Cost-Benefit 表
    # 不包含训练时间
    # --------------------------------------------------------
    cost_benefit_df = build_cost_benefit_table(
        result_df=result_df,
        model=vit_model,
        ckpt_path=VIT_CKPT,
        output_dir=output_dir,
    )

    print("\nSaved files:")
    print(f"Metrics summary: {summary_path}")
    print(f"Cost-benefit summary: {output_dir / 'cost_benefit_summary.csv'}")
    print(f"Prediction files and scatter plots are saved in: {output_dir}")


if __name__ == "__main__":
    main()