# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNGAP9(nn.Module):
    """
    多尺度卷积 + 全局平均池化 + 图像标量特征 + 表格变量融合的回归模型。

    模型流程：
        image -> multi-scale conv branches -> GAP -> concatenate -> mean -> Conv scalar
        Conv scalar + tabular features -> MLP -> regression output
    """

    def __init__(
        self,
        n_tabular: int = 8,
        in_channels: int = 3,
        conv_channels: int = 8,
        kernel_sizes: Optional[Sequence[int]] = None,
        mlp_hidden_dims: Optional[Sequence[int]] = None,
        final_output_dim: int = 1,
    ) -> None:
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [32, 16, 8]

        self.n_tabular = n_tabular
        self.kernel_sizes = list(kernel_sizes)
        self.conv_channels = conv_channels

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, conv_channels, k, padding="same"),
                    nn.SiLU(inplace=False),
                    nn.Conv2d(conv_channels, conv_channels, k, padding="same"),
                    nn.SiLU(inplace=False),
                )
                for k in self.kernel_sizes
            ]
        )

        self.feat_dim = conv_channels * len(self.kernel_sizes)

        input_dim = 1 + n_tabular
        mlp_layers: List[nn.Module] = []

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(inplace=False),
                ]
            )
            input_dim = hidden_dim

        mlp_layers.append(nn.Linear(input_dim, final_output_dim))
        self.mlp = nn.Sequential(*mlp_layers)

    def extract_conv_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回每个卷积分支的空间响应图。"""
        return [conv(x) for conv in self.convs]

    def forward_image_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """返回单一图像标量特征。"""
        batch_size = x.size(0)
        feats = []

        for fmap in self.extract_conv_maps(x):
            pooled = F.adaptive_avg_pool2d(fmap, 1).view(batch_size, -1)
            feats.append(pooled)

        img_feat = torch.cat(feats, dim=1)
        img_scalar = img_feat.mean(dim=1, keepdim=True)
        return img_scalar

    def forward_features(self, x: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        返回 MLP 输入层之前的融合特征。

        输出维度：
            [batch_size, 1 + n_tabular]
        """
        img_scalar = self.forward_image_scalar(x)

        if tabular.dim() == 1:
            tabular = tabular.unsqueeze(0)

        tabular = tabular.to(img_scalar.dtype)
        fused = torch.cat([img_scalar, tabular], dim=1)
        return fused

    def forward(self, x: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        fused = self.forward_features(x, tabular)
        return self.mlp(fused)


class WrappedMLPModel(nn.Module):
    """用于 SHAP DeepExplainer，仅解释 SimpleCNNGAP9 的 MLP 部分。"""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.mlp(fused)
