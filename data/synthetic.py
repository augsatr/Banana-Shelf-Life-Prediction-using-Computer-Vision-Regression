import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter


@dataclass
class RipenessParams:
    h_mean: float
    s_mean: float
    v_mean: float
    brown_spot_density: float
    spot_radius_range: Tuple[float, float]
    curvature: float
    yellowing: float


RIPENESS_MAP = {
    0: RipenessParams(55, 180, 200, 0.0, (0, 0), 1.2, 0.0),     # Green
    1: RipenessParams(30, 200, 220, 0.02, (1, 3), 1.0, 0.3),     # Yellow
    2: RipenessParams(22, 150, 180, 0.08, (2, 6), 0.8, 0.6),    # Spotted
    3: RipenessParams(12, 80, 90, 0.25, (3, 10), 0.6, 0.8),     # Brown
}


class PerlinNoise:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.perm = np.arange(256, dtype=int)
        np.random.RandomState(seed).shuffle(self.perm)
        self.perm = np.stack([self.perm, self.perm]).flatten()

    def _fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a, b, t):
        return a + t * (b - a)

    def _grad(self, hash_val, x, y):
        h = hash_val & 3
        return ((x if h & 1 else -x) + (y if h & 2 else -y))

    def noise(self, x: float, y: float) -> float:
        xi, yi = int(x) & 255, int(y) & 255
        xf, yf = x - int(x), y - int(y)
        u, v = self._fade(xf), self._fade(yf)
        aa = self.perm[self.perm[xi] + yi]
        ab = self.perm[self.perm[xi] + yi + 1]
        ba = self.perm[self.perm[xi + 1] + yi]
        bb = self.perm[self.perm[xi + 1] + yi + 1]
        x1 = self._lerp(self._grad(aa, xf, yf), self._grad(ba, xf - 1, yf), u)
        x2 = self._lerp(self._grad(ab, xf, yf - 1), self._grad(bb, xf - 1, yf - 1), u)
        return (self._lerp(x1, x2, v) + 1) / 2

    def octave_noise(self, x: float, y: float, octaves: int = 4, persistence: float = 0.5) -> float:
        total, amp, freq, max_val = 0.0, 1.0, 1.0, 0.0
        for _ in range(octaves):
            total += self.noise(x * freq, y * freq) * amp
            max_val += amp
            amp *= persistence
            freq *= 2
        return total / max_val


class RealisticBananaGenerator:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.perlin = PerlinNoise()

    def generate(self, shelf_life: float, stage: int = None) -> np.ndarray:
        if stage is None:
            stage = self._shelf_life_to_stage(shelf_life)
        params = RIPENESS_MAP[stage]
        img_size = self.target_size[0] * 2
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        banana = self._draw_banana(img_size, params, stage)
        banana = self._add_texture_noise(banana, stage)
        banana = self._add_brown_spots(banana, params)

        if shelf_life < 4:
            banana = self._add_wrinkle_lines(banana, stage)

        return cv2.resize(banana, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    def _shelf_life_to_stage(self, shelf_life: float) -> int:
        if shelf_life > 7:
            return 0
        elif shelf_life > 3:
            return 1
        elif shelf_life > 1:
            return 2
        return 3

    def _draw_banana(self, size: int, params: RipenessParams, stage: int) -> np.ndarray:
        img = np.ones((size, size, 3), dtype=np.uint8) * 240
        cx, cy = size // 2, size // 2

        w = int(size * 0.22)
        h = int(size * 0.52)
        angle = np.random.uniform(-15, 15)

        y_indices, x_indices = np.ogrid[:size, :size]
        x_shift = params.curvature * (y_indices - cy) ** 2 / (size * 5)

        ellipse_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(ellipse_mask, (cx, cy), (w, h), angle, 0, 360, 255, -1)

        x_off = np.clip(x_shift + x_indices, 0, size - 1).astype(int)
        y_off = np.clip(y_indices, 0, size - 1).astype(int)

        rolled = np.zeros_like(ellipse_mask)
        for i in range(size):
            rolled[y_off[i, 0], x_off[i]] = ellipse_mask[i]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        rolled = cv2.morphologyEx(rolled, cv2.MORPH_CLOSE, kernel)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        banana_hue = int(params.h_mean + np.random.uniform(-5, 5))
        banana_sat = int(params.s_mean + np.random.uniform(-15, 15))
        banana_val = int(params.v_mean + np.random.uniform(-10, 10))
        hsv_img[rolled > 0] = [banana_hue, banana_sat, banana_val]
        img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

        stem_mask = np.zeros((size, size), dtype=np.uint8)
        stem_cx = cx + int(w * 0.6 * np.cos(np.radians(angle)))
        stem_cy = cy - int(h * 0.45 * np.sin(np.radians(angle)))
        cv2.ellipse(stem_mask, (stem_cx, stem_cy), (w // 5, h // 8), angle + 90, 0, 360, 255, -1)
        img[stem_mask > 0] = [90, 120, 80]

        tip_mask = np.zeros((size, size), dtype=np.uint8)
        tip_cx = cx - int(w * 0.6 * np.cos(np.radians(angle)))
        tip_cy = cy + int(h * 0.45 * np.sin(np.radians(angle)))
        cv2.ellipse(tip_mask, (tip_cx, tip_cy), (w // 6, h // 6), angle, 0, 360, 255, -1)
        dark_val = int(params.v_mean * 0.5)
        img[tip_mask > 0] = [params.h_mean, params.s_mean, max(dark_val, 30)]

        return img

    def _add_texture_noise(self, img: np.ndarray, stage: int) -> np.ndarray:
        h, w = img.shape[:2]
        noise = np.zeros((h, w))
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                n = self.perlin.octave_noise(i / 30, j / 30, octaves=3)
                noise[i:i+4, j:j+4] = n

        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        texture_strength = 0.08 + stage * 0.04
        noise_map = (noise * 2 - 1) * texture_strength * 255

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = (hsv[:, :, 2] + noise_map).clip(0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _add_brown_spots(self, img: np.ndarray, params: RipenessParams) -> np.ndarray:
        if params.brown_spot_density < 0.001:
            return img

        h, w = img.shape[:2]
        mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 10
        num_spots = int(mask.sum() * params.brown_spot_density * 0.005)
        num_spots = max(num_spots, 3)

        spot_layer = np.zeros((h, w), dtype=np.uint8)
        spot_y, spot_x = np.where(mask)
        for _ in range(num_spots):
            if len(spot_y) == 0:
                break
            idx = np.random.randint(len(spot_y))
            sy, sx = spot_y[idx], spot_x[idx]
            radius = np.random.uniform(*params.spot_radius_range)
            darkness = np.random.randint(30, 80)
            cv2.circle(spot_layer, (sx, sy), int(radius), darkness, -1)

        blurred = cv2.GaussianBlur(spot_layer, (3, 3), 0.5)
        spot_mask = blurred > 0
        for c in range(3):
            img[:, :, c] = np.where(spot_mask,
                                    np.maximum(img[:, :, c].astype(np.int32) - blurred, 0).astype(np.uint8),
                                    img[:, :, c])

        return img

    def _add_wrinkle_lines(self, img: np.ndarray, stage: int) -> np.ndarray:
        h, w = img.shape[:2]
        mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 10
        wrinkle_img = img.copy()

        for _ in range(stage * 2):
            y = np.random.randint(h // 4, 3 * h // 4)
            x_start = np.random.randint(w // 4, w // 2)
            length = np.random.randint(10, 30)
            pts = np.array([[x_start + i, y + int(np.sin(i * 0.3) * 2)]
                           for i in range(length)], dtype=np.int32)
            pts = pts[(pts[:, 0] < w) & (pts[:, 1] < h)]
            if len(pts) > 1:
                cv2.polylines(wrinkle_img, [pts], False, (50, 50, 50), 1)

        return wrinkle_img

    def generate_batch(self, n: int, output_dir: Path) -> Tuple[List[Path], List[float], List[int]]:
        paths, shelf_lives, stages = [], [], []
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            shelf_life = np.random.uniform(0, 10)
            stage = self._shelf_life_to_stage(shelf_life)
            shelf_life += np.random.normal(0, 0.3)
            shelf_life = np.clip(shelf_life, 0, 10)

            img = self.generate(shelf_life, stage)
            fname = output_dir / f"banana_{i:04d}_s{shelf_life:.2f}_st{stage}.png"
            _, encoded = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            with open(str(fname), "wb") as f:
                f.write(encoded.tobytes())

            paths.append(fname)
            shelf_lives.append(shelf_life)
            stages.append(stage)

        return paths, shelf_lives, stages
