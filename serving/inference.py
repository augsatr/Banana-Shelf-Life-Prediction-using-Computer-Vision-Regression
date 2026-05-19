import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import time

from config import cfg
from features.extractor import AdvancedFeatureExtractor
from features.deep_features import DeepFeatureExtractor
from models.ensemble import AdvancedMLEnsemble
from models.cnn_model import BananaShelfLifeCNN
from models.vit_model import BananaViT
from models.factory import ModelFactory
from models.uncertainty import UncertaintyPredictor
from explain.gradcam import GradCAM, visualize_gradcam
from data.preprocessing import BananaPreprocessor


class InferenceEngine:
    def __init__(self, model_type: str = "ensemble", model_path: Optional[str] = None):
        self.model_type = model_type
        self.preprocessor = BananaPreprocessor()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.deep_extractor = None
        self.use_deep = False
        self.metadata = {}

        if model_type == "ensemble":
            self.model = AdvancedMLEnsemble.load(model_path or "advanced_ensemble")
            self.use_cnn = False
            self.metadata["framework"] = "sklearn"
            self.metadata["model_count"] = len(self.model.models)

        elif model_type == "cnn":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = BananaShelfLifeCNN(
                backbone_name=cfg.get("models", "cnn", "backbone", default="efficientnet_v2_s"),
                dropout=0.0,
            )
            path = model_path or str(Path(cfg.get("paths", "models_dir", default="models/saved")) / "best_model.pth")
            state = torch.load(path, map_location=self.device, weights_only=True)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.use_cnn = True
            self.uncertainty = UncertaintyPredictor(self.model, self.device)
            self.gradcam = GradCAM(self.model)
            self.metadata["framework"] = "pytorch"
            self.metadata["device"] = self.device

        elif model_type == "vit":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = BananaViT(
                model_name=cfg.get("models", "vit", "model_name", default="google/vit-base-patch16-224-in21k"),
                dropout=0.0,
                freeze_backbone=True,
            )
            path = model_path or str(Path(cfg.get("paths", "models_dir", default="models/saved")) / "best_model.pth")
            state = torch.load(path, map_location=self.device, weights_only=True)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.use_cnn = True
            self.uncertainty = UncertaintyPredictor(self.model, self.device)
            self.gradcam = GradCAM(self.model)
            self.metadata["framework"] = "huggingface"
            self.metadata["device"] = self.device

        elif model_type == "hybrid":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cnn_model = BananaShelfLifeCNN(backbone_name="efficientnet_v2_s")
            cnn_path = str(Path(cfg.get("paths", "models_dir", default="models/saved")) / "best_model.pth")
            self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device, weights_only=True))
            self.cnn_model.to(self.device)
            self.cnn_model.eval()

            self.ensemble_model = AdvancedMLEnsemble.load("advanced_ensemble")
            self.deep_extractor = DeepFeatureExtractor(device=self.device)
            self.use_cnn = True
            self.use_deep = True
            self.metadata["framework"] = "hybrid"
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, image: np.ndarray) -> Dict:
        start = time.perf_counter()

        if self.use_cnn:
            return self._predict_cnn(image, start)
        else:
            return self._predict_ml(image, start)

    def _predict_cnn(self, image: np.ndarray, start: float) -> Dict:
        from torchvision import transforms
        from PIL import Image as PILImage

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        processed = self.preprocessor(image_rgb)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pil_img = PILImage.fromarray(processed)
        tensor = transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "forward_with_features"):
                shelf_life, stage_logits, log_var, emb, _ = self.model.forward_with_features(tensor)
            else:
                shelf_life, stage_logits, log_var, emb = self.model(tensor)

        pred = float(shelf_life.item())
        stage = int(torch.argmax(stage_logits, dim=1).item())
        aleatoric = float(torch.exp(0.5 * log_var).item())

        mean, std, epistemic = self.uncertainty.mc_dropout_predict(tensor, n_samples=30)
        total_uncertainty = float(std.item())

        cam = self.gradcam(tensor)
        overlay = visualize_gradcam(processed, cam)

        elapsed = time.perf_counter() - start
        return self._format_result(pred, stage, {
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": float(epistemic.item()),
            "total_uncertainty": total_uncertainty,
            "inference_time_ms": round(elapsed * 1000, 2),
            "confidence_95_ci": (round(pred - 1.96 * total_uncertainty, 2),
                                  round(pred + 1.96 * total_uncertainty, 2)),
            "gradcam_overlay": overlay,
        })

    def _predict_ml(self, image: np.ndarray, start: float) -> Dict:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        processed = self.preprocessor(image_rgb)
        features = self.feature_extractor.extract_all(processed).reshape(1, -1)

        pred = float(self.model.predict(features)[0])
        stage = self._shelf_life_to_stage(pred)

        _, components = self.model.predict_with_components(features)
        component_preds = {k: float(v[0]) for k, v in components.items()}
        ensemble_std = float(np.std(list(components.values())))

        if self.use_deep and self.deep_extractor:
            deep_feats = self.deep_extractor.extract(processed)
            shelf_life = (pred + float(self.ensemble_model.predict(deep_feats.reshape(1, -1))[0])) / 2
            stage = self._shelf_life_to_stage(shelf_life)
            pred = shelf_life

        elapsed = time.perf_counter() - start
        return self._format_result(pred, stage, {
            "component_predictions": component_preds,
            "ensemble_std": ensemble_std,
            "inference_time_ms": round(elapsed * 1000, 2),
            "confidence_95_ci": (round(pred - 1.96 * ensemble_std, 2),
                                  round(pred + 1.96 * ensemble_std, 2)),
        })

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        return [self.predict(img) for img in images]

    def _format_result(self, shelf_life: float, stage: int, extra: Dict) -> Dict:
        shelf_life = np.clip(shelf_life, 0, 10)
        stage_name = {
            0: "Green (Unripe)",
            1: "Yellow (Ripe)",
            2: "Yellow with Spots (Overripe)",
            3: "Brown (Spoiled)",
        }.get(stage, f"Stage {stage}")

        return {
            "shelf_life_days": round(shelf_life, 2),
            "ripeness_stage": stage,
            "ripeness_label": stage_name,
            **extra,
        }

    def _shelf_life_to_stage(self, shelf_life: float) -> int:
        if shelf_life > 7:
            return 0
        elif shelf_life > 3:
            return 1
        elif shelf_life > 1:
            return 2
        return 3

    def predict_from_file(self, image_path: str) -> Dict:
        with open(image_path, "rb") as f:
            raw = bytearray(f.read())
        arr = np.asarray(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        result = self.predict(image)
        result["filename"] = Path(image_path).name
        return result
