# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except Exception as e:
    raise ImportError(
        "当前环境无法导入 torchvision.models.vit_b_16。"
        "请确认 torchvision 版本支持 ViT，例如 torchvision>=0.13。"
    ) from e


class MultimodalViTRegressor(nn.Module):
    """
    Multimodal ViT regression model.

    Image branch:
        Floor-plan image -> ViT-B/16 -> image feature

    Tabular branch:
        Numerical design / energy-efficiency parameters -> MLP -> tabular feature

    Fusion branch:
        concat(image feature, tabular feature) -> MLP -> regression output
    """

    def __init__(
        self,
        n_tabular: int = 8,
        final_output_dim: int = 1,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        image_proj_dim: int = 256,
        tabular_proj_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.image_encoder = vit_b_16(weights=weights)

        try:
            vit_dim = self.image_encoder.heads.head.in_features
        except Exception:
            vit_dim = 768

        self.image_encoder.heads = nn.Identity()

        if freeze_backbone:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_projector = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, image_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tabular_encoder = nn.Sequential(
            nn.Linear(n_tabular, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, tabular_proj_dim),
            nn.LayerNorm(tabular_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fusion_dim = image_proj_dim + tabular_proj_dim

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden_dim // 2, final_output_dim),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        image_feat = self.image_projector(image_feat)

        tabular_feat = self.tabular_encoder(tabular)

        fused_feat = torch.cat([image_feat, tabular_feat], dim=1)
        output = self.fusion_head(fused_feat)

        return output