import cv2
import numpy as np
from typing import Tuple, Optional


class BananaPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess(image)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = self._remove_background(image)
        image = self._center_crop(image)
        image = self._normalize_illumination(image)
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower = np.array([25, 20, 20])
        upper = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_not(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 2)
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
        return (image * mask_3c).astype(np.uint8)

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is None or len(coords) < 5:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        cx, cy = x + w // 2, y + h // 2
        side = max(w, h) + 40
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(image.shape[1], x1 + side)
        y2 = min(image.shape[0], y1 + side)
        cropped = image[y1:y2, x1:x2]
        padded = np.ones((side, side, 3), dtype=np.uint8) * 255
        offset_x = (side - cropped.shape[1]) // 2
        offset_y = (side - cropped.shape[0]) // 2
        padded[offset_y:offset_y+cropped.shape[0], offset_x:offset_x+cropped.shape[1]] = cropped
        return padded

    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        normalized = cv2.merge([l, a, b])
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)

    def extract_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        return mask

    def extract_contour(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        mask = self.extract_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if not contours:
            return np.zeros_like(mask), 0.0, 0.0
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        return c, area, perimeter
