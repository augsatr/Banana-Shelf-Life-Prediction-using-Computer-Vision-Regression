import numpy as np
import cv2
import random
from typing import List, Tuple, Optional


class MixUp:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, images: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lam = np.random.beta(self.alpha, self.alpha)
        idx = np.random.permutation(len(images))
        mixed_images = lam * images + (1 - lam) * images[idx]
        mixed_targets = lam * targets + (1 - lam) * targets[idx]
        return mixed_images, mixed_targets


class CutMix:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, images: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lam = np.random.beta(self.alpha, self.alpha)
        idx = np.random.permutation(len(images))
        h, w = images.shape[2:]
        cut_h = int(h * np.sqrt(1 - lam))
        cut_w = int(w * np.sqrt(1 - lam))
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2 = min(h, cy + cut_h // 2)
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (w * h)
        mixed_targets = lam * targets + (1 - lam) * targets[idx]
        return images, mixed_targets


class RandAugment:
    def __init__(self, magnitude: int = 12, num_ops: int = 3):
        self.magnitude = magnitude
        self.num_ops = num_ops
        self.ops = [
            self._autocontrast, self._equalize, self._rotate,
            self._solarize, self._color, self._posterize,
            self._contrast, self._brightness, self._sharpness,
            self._shear_x, self._shear_y, self._translate_x, self._translate_y,
        ]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        ops = random.sample(self.ops, min(self.num_ops, len(self.ops)))
        for op in ops:
            image = op(image)
        return image

    def _autocontrast(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lo, hi = gray.min(), gray.max()
        if hi - lo > 0:
            return ((img.astype(np.float32) - lo) * 255 / (hi - lo)).clip(0, 255).astype(np.uint8)
        return img

    def _equalize(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    def _rotate(self, img):
        angle = random.uniform(-self.magnitude, self.magnitude)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _solarize(self, img):
        threshold = random.randint(64, 192)
        img_copy = img.copy()
        img_copy[img_copy > threshold] = 255 - img_copy[img_copy > threshold]
        return img_copy

    def _color(self, img):
        factor = 1.0 + random.uniform(-1, 1) * self.magnitude / 30
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = (hsv[:, :, 1] * factor).clip(0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _posterize(self, img):
        bits = random.randint(3, 6)
        mask = ~(2 ** (8 - bits) - 1)
        return img & mask

    def _contrast(self, img):
        factor = 1.0 + random.uniform(-0.5, 0.5) * self.magnitude / 12
        mean = img.mean(axis=(0, 1), keepdims=True)
        return ((img.astype(np.float32) - mean) * factor + mean).clip(0, 255).astype(np.uint8)

    def _brightness(self, img):
        factor = 1.0 + random.uniform(-0.5, 0.5) * self.magnitude / 12
        return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)

    def _sharpness(self, img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        factor = random.uniform(0, self.magnitude) / 12
        sharpened = cv2.filter2D(img, -1, kernel)
        return cv2.addWeighted(img, 1 - factor, sharpened, factor, 0)

    def _shear_x(self, img):
        strength = random.uniform(-self.magnitude, self.magnitude) / 30
        M = np.float32([[1, strength, 0], [0, 1, 0]])
        h, w = img.shape[:2]
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _shear_y(self, img):
        strength = random.uniform(-self.magnitude, self.magnitude) / 30
        M = np.float32([[1, 0, 0], [strength, 1, 0]])
        h, w = img.shape[:2]
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _translate_x(self, img):
        pixels = int(random.uniform(-self.magnitude, self.magnitude))
        M = np.float32([[1, 0, pixels], [0, 1, 0]])
        h, w = img.shape[:2]
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _translate_y(self, img):
        pixels = int(random.uniform(-self.magnitude, self.magnitude))
        M = np.float32([[1, 0, 0], [0, 1, pixels]])
        h, w = img.shape[:2]
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def get_augmentation_pipeline(mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0,
                               randaug_magnitude: int = 12, randaug_num_ops: int = 3):
    return {
        "randaug": RandAugment(magnitude=randaug_magnitude, num_ops=randaug_num_ops),
        "mixup": MixUp(alpha=mixup_alpha),
        "cutmix": CutMix(alpha=cutmix_alpha),
    }
