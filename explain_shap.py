# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.simplecnn_gap9.data import (
    MODEL_FEATURE_NAMES,
    denormalize_model_feature,
    select_model_tabular,
    ImageTabularTaskData,
)
from src.simplecnn_gap9.model import SimpleCNNGAP9, WrappedMLPModel
from src.simplecnn_gap9.utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain SimpleCNNGAP9 using SHAP")

    parser.add_argument("--data-root", type=str, default="plan_1125_dataset")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/shap_gap9")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=224)

    parser.add_argument("--background", type=str, default="mean", choices=["mean", "zero"])
    parser.add_argument("--max-samples", type=int, default=0, help="0 means using all samples")

    return parser.parse_args()


def normalize_shap_values(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if torch.is_tensor(shap_values):
        shap_values = shap_values.detach().cpu().numpy()

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    if shap_values.ndim == 3 and shap_values.shape[0] == 1:
        shap_values = shap_values.squeeze(0)

    if shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    return shap_values


def get_expected_value(explainer) -> float:
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[0]
    expected_value = np.asarray(expected_value).reshape(-1)[0]
    return float(expected_value)


@torch.no_grad()
def extract_fused_features(loader, model: SimpleCNNGAP9, device: torch.device):
    model.eval()

    fused_list = []
    pred_list = []
    target_list = []
    path_list = []

    for images, tabular, target, paths in tqdm(loader, desc="Extract fused features"):
        images = images.to(device, non_blocking=True)
        tabular = select_model_tabular(tabular).to(device, non_blocking=True).float()

        fused = model.forward_features(images, tabular)
        preds = model.mlp(fused)

        fused_list.append(fused.detach().cpu().numpy())
        pred_list.append(preds.detach().cpu().numpy().reshape(-1))
        target_list.append(target.detach().cpu().numpy().reshape(-1))
        path_list.extend(list(paths))

    fused_all = np.concatenate(fused_list, axis=0)
    preds_all = np.concatenate(pred_list, axis=0)
    targets_all = np.concatenate(target_list, axis=0)

    return fused_all, preds_all, targets_all, path_list


def save_summary_plot(shap_values: np.ndarray, fused: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    import shap

    shap.summary_plot(
        shap_values,
        fused,
        feature_names=MODEL_FEATURE_NAMES,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_prediction_shap_plot(preds: np.ndarray, shap_values: np.ndarray, output_dir: Path) -> None:
    shap_abs_sum = np.abs(shap_values).sum(axis=1)

    plt.figure(figsize=(7, 6))
    plt.scatter(preds, shap_abs_sum, alpha=0.6, s=16)
    plt.xlabel("Model prediction")
    plt.ylabel("Total |SHAP| contribution")
    plt.title("Prediction vs. SHAP total contribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_vs_pred.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_dependence_plots(shap_values: np.ndarray, fused: np.ndarray, output_dir: Path) -> None:
    dep_dir = output_dir / "dependence_plots"
    dep_dir.mkdir(parents=True, exist_ok=True)

    for feature_index, feature_name in enumerate(MODEL_FEATURE_NAMES):
        feature_data = denormalize_model_feature(feature_index, fused[:, feature_index])
        shap_data = shap_values[:, feature_index]

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(6, 7),
            gridspec_kw={"height_ratios": [3, 1]},
        )

        axes[0].scatter(feature_data, shap_data, c=feature_data, cmap="viridis", alpha=0.7, s=12)
        axes[0].set_xlabel(f"{feature_name} value")
        axes[0].set_ylabel("SHAP value")
        axes[0].set_title(f"SHAP dependence: {feature_name}")
        axes[0].grid(True, alpha=0.25)

        axes[1].hist(feature_data, bins=25, color="gray", alpha=0.8)
        axes[1].set_xlabel(f"{feature_name} distribution")
        axes[1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(dep_dir / f"{feature_index:02d}_{feature_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    import shap

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print("Using device:", device)

    model = SimpleCNNGAP9(n_tabular=8, final_output_dim=1).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    print("Model loaded:", args.checkpoint)

    data = ImageTabularTaskData(
        image_folder_path=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        crop_size=args.crop_size,
        resize_size=args.resize_size,
    )

    if args.split == "train":
        loader = data.trainloader
    elif args.split == "test":
        loader = data.testloader
    else:
        if not hasattr(data, "allloader"):
            raise FileNotFoundError("split='all' requires merged_data.csv in data-root")
        loader = data.allloader

    fused, preds, targets, img_paths = extract_fused_features(loader, model, device)

    if args.max_samples > 0:
        fused = fused[: args.max_samples]
        preds = preds[: args.max_samples]
        targets = targets[: args.max_samples]
        img_paths = img_paths[: args.max_samples]

    print("Fused shape:", fused.shape)

    if args.background == "zero":
        background = torch.zeros((1, fused.shape[1]), dtype=torch.float32, device=device)
    else:
        background_mean = fused.mean(axis=0, keepdims=True)
        background = torch.tensor(background_mean, dtype=torch.float32, device=device)

    wrapped_model = WrappedMLPModel(model.mlp).to(device)
    wrapped_model.eval()

    explainer = shap.DeepExplainer(wrapped_model, background)

    fused_tensor = torch.tensor(fused, dtype=torch.float32, device=device)
    raw_shap_values = explainer.shap_values(fused_tensor)
    shap_values = normalize_shap_values(raw_shap_values)

    expected_value = get_expected_value(explainer)
    shap_sum = shap_values.sum(axis=1)
    shap_abs_sum = np.abs(shap_values).sum(axis=1)
    reconstructed_pred = expected_value + shap_sum
    reconstruction_error = reconstructed_pred - preds

    print("Mean reconstruction error:", np.mean(np.abs(reconstruction_error)))

    save_summary_plot(shap_values, fused, output_dir)
    save_prediction_shap_plot(preds, shap_values, output_dir)
    save_dependence_plots(shap_values, fused, output_dir)

    df = pd.DataFrame({"img_path": img_paths, "pred": preds, "target": targets})

    for i, feature_name in enumerate(MODEL_FEATURE_NAMES):
        df[f"feature_{feature_name}"] = fused[:, i]
        df[f"SHAP_{feature_name}"] = shap_values[:, i]

    df["shap_sum"] = shap_sum
    df["shap_abs_sum"] = shap_abs_sum
    df["expected_value"] = expected_value
    df["reconstructed_pred"] = reconstructed_pred
    df["reconstruction_error"] = reconstruction_error

    df.to_csv(output_dir / "shap_values.csv", index=False, encoding="utf-8-sig")

    print("Done. Outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
