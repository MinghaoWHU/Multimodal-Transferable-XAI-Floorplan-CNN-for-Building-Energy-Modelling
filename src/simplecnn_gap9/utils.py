# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "cuda") -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(save_dir: Path, state: Dict, is_best: bool) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    last_tmp = save_dir / "last.pth.tmp"
    last_path = save_dir / "last.pth"
    torch.save(state, last_tmp)
    os.replace(last_tmp, last_path)

    if is_best:
        best_tmp = save_dir / "best.pth.tmp"
        best_path = save_dir / "best.pth"
        torch.save(state, best_tmp)
        os.replace(best_tmp, best_path)


def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    map_location: str | torch.device = "cpu",
) -> Dict:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt
