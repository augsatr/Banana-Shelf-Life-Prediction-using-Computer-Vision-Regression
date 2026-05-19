import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from models.cnn_model import BananaShelfLifeCNN
from models.vit_model import BananaViT


class ModelFactory:
    @staticmethod
    def create(model_type: str, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        config = config or {}

        if model_type == "cnn":
            return BananaShelfLifeCNN(
                backbone_name=config.get("backbone", "efficientnet_v2_s"),
                pretrained=config.get("pretrained", True),
                dropout=config.get("dropout", 0.3),
                embedding_dim=config.get("embedding_dim", 128),
            )
        elif model_type == "vit":
            return BananaViT(
                model_name=config.get("model_name", "google/vit-base-patch16-224-in21k"),
                dropout=config.get("dropout", 0.2),
                freeze_backbone=config.get("freeze_backbone", False),
                embedding_dim=config.get("embedding_dim", 128),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_ensemble(model_types: list, configs: Optional[list] = None) -> nn.ModuleList:
        if configs is None:
            configs = [{} for _ in model_types]
        return nn.ModuleList([
            ModelFactory.create(mt, cfg) for mt, cfg in zip(model_types, configs)
        ])

    @staticmethod
    def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
        state = torch.load(path, map_location=device, weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        return model
