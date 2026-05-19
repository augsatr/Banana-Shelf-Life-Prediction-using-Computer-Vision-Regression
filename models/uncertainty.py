import torch
import numpy as np
from typing import Tuple, Optional


class UncertaintyPredictor:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def mc_dropout_predict(self, x: torch.Tensor, n_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.train()
        predictions = []
        for _ in range(n_samples):
            pred, _, _, _ = self.model(x)
            predictions.append(pred)
        self.model.eval()
        preds = torch.stack(predictions)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        epistemic = preds.var(dim=0)
        return mean, std, epistemic

    @torch.no_grad()
    def aleatoric_predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        pred, _, log_var, _ = self.model(x)
        aleatoric = torch.exp(0.5 * log_var)
        return pred, aleatoric

    def predict_with_interval(self, x: torch.Tensor, confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
        mean, std, _ = self.mc_dropout_predict(x, n_samples=50)
        z = 1.96 if confidence == 0.95 else 2.576
        lo = mean - z * std
        hi = mean + z * std
        return mean.item(), (lo.item(), hi.item())

    def ensemble_predict(self, models: list, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        for model in models:
            model.eval()
            model.to(self.device)
            with torch.no_grad():
                p, _, _, _ = model(x)
                preds.append(p.cpu().numpy())
        preds = np.array(preds)
        return preds.mean(axis=0), preds.std(axis=0)
