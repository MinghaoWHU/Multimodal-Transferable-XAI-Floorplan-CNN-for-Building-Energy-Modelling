# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.simplecnn_gap9.data import resolve_image_path
from src.simplecnn_gap9.model import SimpleCNNGAP9
from src.simplecnn_gap9.utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SimpleCNNGAP9 convolution maps")

    parser.add_argument("--data-root", type=str, default="plan_1125_dataset")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, default="merged_data.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/conv_maps")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=224)

    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--every", type=int, default=1, help="process one image every N rows")
    parser.add_argument("--contains", type=str, default="", help="only process paths containing this string")

    return parser.parse_args()


def read_image_paths(csv_path: Path, data_root: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        if "image_path" in reader.fieldnames:
            col = "image_path"
        elif "name" in reader.fieldnames:
            col = "name"
        else:
            col = reader.fieldnames[0]

        paths = [resolve_image_path(row[col], data_root) for row in reader]

    return paths


def build_transform(crop_size: int, resize_size: int):
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )


def load_image(img_path: str, transform) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def visualize_conv_maps(conv_maps: List[np.ndarray], save_path: Path, kernel_sizes: List[int]) -> None:
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12})

    # 每个 kernel 分支内部有多个通道，这里对通道求平均，得到一个空间响应图。
    maps_2d = [x[0].mean(axis=0) for x in conv_maps]

    fig, axes = plt.subplots(2, 7, figsize=(26, 8))
    gs = gridspec.GridSpec(2, 7, figure=fig)

    for r in range(2):
        for c in [5, 6]:
            fig.delaxes(axes[r, c])

    avg_map = np.zeros_like(maps_2d[0])

    for i in range(10):
        r, c = divmod(i, 5)
        ax = axes[r, c]

        fmap = maps_2d[i]
        avg_map += fmap

        im = ax.imshow(fmap, cmap="coolwarm", interpolation="nearest")
        ax.set_title(f"Kernel {kernel_sizes[i]}\nmean={fmap.mean():.4f}")
        ax.axis("off")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
        cbar.formatter = FormatStrFormatter("%.3f")
        cbar.update_ticks()

    avg_map = avg_map / len(maps_2d)

    ax_big = fig.add_subplot(gs[:, 5:7])
    im_big = ax_big.imshow(avg_map, cmap="coolwarm", interpolation="nearest")
    ax_big.set_title(f"Average response\nmean={avg_map.mean():.4f}")
    ax_big.axis("off")

    cbar_big = fig.colorbar(im_big, ax=ax_big, fraction=0.046, pad=0.02)
    cbar_big.formatter = FormatStrFormatter("%.3f")
    cbar_big.update_ticks()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_single_image(img_path: str, model: SimpleCNNGAP9, device: torch.device, transform, output_dir: Path) -> None:
    image = load_image(img_path, transform).to(device)
    conv_maps = model.extract_conv_maps(image)
    conv_maps_np = [x.detach().cpu().numpy() for x in conv_maps]

    img_name = Path(img_path).stem.replace(" ", "_")
    save_path = output_dir / f"{img_name}_conv_maps.png"
    visualize_conv_maps(conv_maps_np, save_path, model.kernel_sizes)


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)
    csv_path = data_root / args.csv

    model = SimpleCNNGAP9(n_tabular=8, final_output_dim=1).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    image_paths = read_image_paths(csv_path, data_root)
    transform = build_transform(args.crop_size, args.resize_size)

    processed = 0

    for idx, img_path in enumerate(tqdm(image_paths, desc="Visualize conv maps")):
        if args.every > 1 and idx % args.every != 0:
            continue

        if args.contains and args.contains not in img_path:
            continue

        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        run_single_image(img_path, model, device, transform, output_dir)
        processed += 1

        if args.limit > 0 and processed >= args.limit:
            break

    print(f"Done. Processed {processed} images. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
