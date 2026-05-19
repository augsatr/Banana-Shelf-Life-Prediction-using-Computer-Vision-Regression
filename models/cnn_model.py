import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


BACKBONE_REGISTRY = {
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_v2_s": models.efficientnet_v2_s,
    "efficientnet_v2_m": models.efficientnet_v2_m,
    "resnet50": models.resnet50,
    "resnext50_32x4d": models.resnext50_32x4d,
    "resnext101_64x4d": models.resnext101_64x4d,
    "convnext_tiny": models.convnext_tiny,
    "convnext_small": models.convnext_small,
    "densenet121": models.densenet121,
    "mobilenet_v3_large": models.mobilenet_v3_large,
}


def get_backbone_in_features(backbone: nn.Module) -> int:
    if hasattr(backbone, "classifier") and hasattr(backbone.classifier, "in_features"):
        if isinstance(backbone.classifier, nn.Sequential):
            last = backbone.classifier[-1]
            return getattr(last, "in_features", 1280)
        return getattr(backbone.classifier, "in_features", 1280)
    if hasattr(backbone, "fc"):
        return backbone.fc.in_features
    if hasattr(backbone, "head"):
        return backbone.head.in_features
    return 1280


class RegressionHead(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
        )
        self.regressor = nn.Linear(embedding_dim, 1)
        self.classifier = nn.Linear(embedding_dim, 4)
        self.uncertainty = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        emb = self.net(x)
        shelf_life = self.regressor(emb).squeeze(-1)
        stage = self.classifier(emb)
        log_var = self.uncertainty(emb).squeeze(-1)
        return shelf_life, stage, log_var, emb


class BananaShelfLifeCNN(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_v2_s",
        pretrained: bool = True,
        dropout: float = 0.3,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name in BACKBONE_REGISTRY:
            weights = "IMAGENET1K_V1" if pretrained else None
            self.backbone = BACKBONE_REGISTRY[backbone_name](weights=weights)
        else:
            self.backbone = models.efficientnet_v2_s(weights="IMAGENET1K_V1" if pretrained else None)

        in_features = get_backbone_in_features(self.backbone)

        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "head"):
            self.backbone.head = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = RegressionHead(in_features, embedding_dim, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = self.pool(features).flatten(1)
        return self.head(features)

    def forward_with_features(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = self.pool(features).flatten(1)
        shelf_life, stage, log_var, emb = self.head(features)
        return shelf_life, stage, log_var, emb, features
