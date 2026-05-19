import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class BananaViT(nn.Module):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        embedding_dim: int = 128,
    ):
        super().__init__()
        from transformers import ViTModel, ViTConfig

        try:
            self.backbone = ViTModel.from_pretrained(model_name)
        except Exception:
            config = ViTConfig.from_pretrained(model_name)
            self.backbone = ViTModel(config)

        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.GELU(),
        )
        self.regressor = nn.Linear(embedding_dim, 1)
        self.classifier = nn.Linear(embedding_dim, 4)
        self.uncertainty = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        outputs = self.backbone(x)
        cls_token = outputs.last_hidden_state[:, 0]
        emb = self.head(cls_token)
        shelf_life = self.regressor(emb).squeeze(-1)
        stage = self.classifier(emb)
        log_var = self.uncertainty(emb).squeeze(-1)
        return shelf_life, stage, log_var, emb

    @torch.no_grad()
    def predict_with_uncertainty(self, x, mc_samples: int = 30):
        self.train()
        predictions = []
        for _ in range(mc_samples):
            pred, _, _, _ = self.forward(x)
            predictions.append(pred)
        self.eval()
        preds = torch.stack(predictions)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std
