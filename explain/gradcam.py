import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, List, Tuple


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: Optional[str] = None):
        target = self._find_layer(target_layer)
        if target is None:
            layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.ReLU)):
                    layers.append((name, module))
            if layers:
                _, target = layers[-3]

        if target is not None:
            target.register_forward_hook(self._forward_hook)
            target.register_full_backward_hook(self._backward_hook)

    def _find_layer(self, name: Optional[str]):
        if name is None:
            return None
        for n, m in self.model.named_modules():
            if n == name:
                return m
        return None

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        shelf_life, stage_logits, _, _ = self.model(x)

        if target_class is None:
            score = shelf_life.sum()
        else:
            score = stage_logits[:, target_class].sum()

        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((x.shape[2], x.shape[3]))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
        return cam


def visualize_gradcam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + image * (1 - alpha)).astype(np.uint8)
    return overlay
