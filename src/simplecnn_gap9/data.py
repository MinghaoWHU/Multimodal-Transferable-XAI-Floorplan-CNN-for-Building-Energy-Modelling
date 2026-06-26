# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


DEFAULT_IMAGE_FOLDER_PATH = "plan_dataset/"
DEFAULT_CENTRE_CROP_SIZE = 224
DEFAULT_RESIZED_IMAGE_SIZE = 224

RAW_TABULAR_FEATURE_NAMES = [
    "NUMBER_HEATED_ROOMS",
    "CURRENT_ENERGY_EFFICIENCY",
    "HOT_WATER_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    "WALLS_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
    "FLOOR_HEIGHT",
    "MAINHEAT_ENERGY_EFF",
]

TABULAR_MEAN = np.array(
    [2.6242, 68.0821, 2.9932, 4.6932, 2.2170, 2.3196, 3.0445, 2.4108, 3.1023],
    dtype=np.float32,
)

TABULAR_STD = np.array(
    [0.8252, 8.7834, 1.4590, 2.0063, 1.7442, 1.1539, 2.0469, 0.0874, 1.4301],
    dtype=np.float32,
)

# 原始训练脚本中使用：torch.cat((tabular[:, :1], tabular[:, 2:]), dim=1)
# 即删除索引 1: CURRENT_ENERGY_EFFICIENCY。
MODEL_TABULAR_INDICES = [0, 2, 3, 4, 5, 6, 7, 8]
MODEL_TABULAR_FEATURE_NAMES = [RAW_TABULAR_FEATURE_NAMES[i] for i in MODEL_TABULAR_INDICES]
MODEL_FEATURE_NAMES = ["Conv"] + MODEL_TABULAR_FEATURE_NAMES


def get_image_transform(crop_size: int = 224, resize_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ]
    )


def resolve_image_path(path_text: str, data_root: Path) -> str:
    path_text = str(path_text).strip()
    candidate = Path(path_text)

    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    if candidate.exists():
        return str(candidate)

    root_candidate = data_root / path_text
    if root_candidate.exists():
        return str(root_candidate)

    return path_text


def load_image_targets_from_csv(
    csv_path: Path,
    image_root: Optional[Path] = None,
    header: bool = True,
    filter_missing: bool = False,
) -> Dict[str, Any]:
    """
    读取图像路径、表格变量和目标值。

    CSV 要求：
        第 1 列：图像路径
        后续列：9 个表格变量 + 1 个目标变量
    """
    csv_path = Path(csv_path)
    image_root = Path(image_root) if image_root is not None else csv_path.parent

    image_targets: Dict[str, Any] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    start_line = 1 if header else 0

    if header:
        print(f"Header line of csv {csv_path}: {rows[0]}")

    for row in tqdm(rows[start_line:], desc=f"Loading {csv_path.name}"):
        if len(row) < 2:
            continue

        img_path = resolve_image_path(row[0], image_root)

        if filter_missing and not os.path.exists(img_path):
            continue

        values = np.array(
            [float(x) if str(x).strip() != "" else 0.0 for x in row[1:]],
            dtype=np.float32,
        )

        image_targets[img_path] = values

    if len(image_targets) == 0:
        raise ValueError(f"No valid samples found in: {csv_path}")

    return image_targets


class ImageTabularDataset(Dataset):
    def __init__(
        self,
        image_data: Dict[str, Any],
        crop_size: int = DEFAULT_CENTRE_CROP_SIZE,
        resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE,
    ) -> None:
        self.image_data = image_data
        self.image_paths = list(image_data.keys())
        self.transform = get_image_transform(crop_size, resize_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        values = np.asarray(self.image_data[img_path], dtype=np.float32)

        if values.shape[0] < len(RAW_TABULAR_FEATURE_NAMES) + 1:
            raise ValueError(
                f"Sample has {values.shape[0]} numeric values, but at least "
                f"{len(RAW_TABULAR_FEATURE_NAMES) + 1} are required: {img_path}"
            )

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tabular_input = values[: len(RAW_TABULAR_FEATURE_NAMES)]
        tabular_input = (tabular_input - TABULAR_MEAN) / TABULAR_STD

        target = np.array(values[-1], dtype=np.float32)

        return image, tabular_input.astype(np.float32), target, img_path


class ImageTabularTaskData:
    def __init__(
        self,
        image_folder_path: str = DEFAULT_IMAGE_FOLDER_PATH,
        crop_size: int = DEFAULT_CENTRE_CROP_SIZE,
        resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE,
        batch_size: int = 16,
        workers: int = 16,
        train_csv: str = "train1.csv",
        test_csv: str = "test1.csv",
        all_csv: str = "merged_data.csv",
        filter_missing: bool = False,
    ) -> None:
        self.image_folder_path = Path(image_folder_path)
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.filter_missing = filter_missing

        ngpus_per_node = max(torch.cuda.device_count(), 1)
        self.batch_size = max(int(batch_size / ngpus_per_node), 1)
        self.workers = max(int((workers + ngpus_per_node - 1) / ngpus_per_node), 0)

        self.train_csv = train_csv
        self.test_csv = test_csv
        self.all_csv = all_csv

        self.trainloader = self.make_loader(train_csv, shuffle=True)
        self.testloader = self.make_loader(test_csv, shuffle=False)

        if (self.image_folder_path / all_csv).exists():
            self.allloader = self.make_loader(all_csv, shuffle=False)

    @property
    def output_image_size(self) -> Tuple[int, int, int]:
        return 3, self.resize_size, self.resize_size

    def make_loader(self, csv_name: str, shuffle: bool) -> DataLoader:
        csv_path = self.image_folder_path / csv_name
        image_data = load_image_targets_from_csv(
            csv_path,
            image_root=self.image_folder_path,
            filter_missing=self.filter_missing,
        )

        dataset = ImageTabularDataset(
            image_data=image_data,
            crop_size=self.crop_size,
            resize_size=self.resize_size,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )


def select_model_tabular(tabular: torch.Tensor) -> torch.Tensor:
    """将 9 维标准化表格变量转换为 SimpleCNNGAP9 实际使用的 8 维变量。"""
    return tabular[:, MODEL_TABULAR_INDICES]


def denormalize_model_feature(feature_index: int, values: np.ndarray) -> np.ndarray:
    """
    将 fused 特征中的表格变量还原到原始尺度。

    feature_index=0 表示 Conv，不做反标准化。
    feature_index>=1 表示第 feature_index-1 个模型表格变量。
    """
    if feature_index == 0:
        return values

    raw_idx = MODEL_TABULAR_INDICES[feature_index - 1]
    return values * TABULAR_STD[raw_idx] + TABULAR_MEAN[raw_idx]
