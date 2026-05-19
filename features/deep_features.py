import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional


class DeepFeatureExtractor:
    def __init__(self, model_name: str = "efficientnet_v2_s", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.name = model_name

    def _build_model(self, name: str) -> nn.Module:
        import torchvision.models as models
        factory = {
            "efficientnet_v2_s": models.efficientnet_v2_s,
            "efficientnet_b0": models.efficientnet_b0,
            "resnext50_32x4d": models.resnext50_32x4d,
            "convnext_tiny": models.convnext_tiny,
        }
        if name in factory:
            model = factory[name](weights="IMAGENET1K_V1")
            in_features = model.classifier[-1].in_features if hasattr(model, "classifier") else \
                          model.fc.in_features if hasattr(model, "fc") else 2048
            if hasattr(model, "classifier"):
                if isinstance(model.classifier, nn.Sequential):
                    model.classifier = nn.Identity()
                else:
                    model.classifier = nn.Identity()
            elif hasattr(model, "fc"):
                model.fc = nn.Identity()
            return model

        import timm
        model = timm.create_model(name, pretrained=True, num_classes=0)
        return model

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        from torchvision import transforms
        from PIL import Image as PILImage

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        pil = PILImage.fromarray(image_rgb)
        tensor = transform(pil).unsqueeze(0).to(self.device)

        features = self.model(tensor)
        return features.cpu().numpy().flatten()

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        from torchvision import transforms
        from PIL import Image as PILImage

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensors = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
            pil = PILImage.fromarray(rgb)
            tensors.append(transform(pil))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model(batch)
        return features.cpu().numpy()
