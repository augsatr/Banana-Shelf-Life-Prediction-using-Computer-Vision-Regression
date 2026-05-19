import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")


@dataclass
class FeatureVector:
    raw: np.ndarray = field(default_factory=lambda: np.array([]))
    names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        return dict(zip(self.names, self.raw.tolist()))

    def to_array(self) -> np.ndarray:
        return self.raw


class AdvancedFeatureExtractor:
    def __init__(self):
        self.kmeans = MiniBatchKMeans(n_clusters=6, random_state=42, n_init=5, batch_size=1000)
        self._fitted = False

    def extract_all(self, image: np.ndarray) -> np.ndarray:
        features = FeatureVector()
        self._extract_color_histogram(image, features)
        self._extract_color_stats(image, features)
        self._extract_glcm(image, features)
        self._extract_lbp(image, features)
        self._extract_hog(image, features)
        self._extract_spots(image, features)
        self._extract_dominant_colors(image, features)
        self._extract_morphology(image, features)
        self._extract_curvature(image, features)
        self._extract_fourier_descriptors(image, features)
        return features.to_array()

    def _extract_color_histogram(self, image: np.ndarray, fv: FeatureVector):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i, (name, bins) in enumerate([("H", 16), ("S", 8), ("V", 8)]):
            hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
            hist = hist.flatten() / max(hist.sum(), 1)
            fv.raw = np.concatenate([fv.raw, hist])
            fv.names += [f"hist_{name}_{j}" for j in range(bins)]

    def _extract_color_stats(self, image: np.ndarray, fv: FeatureVector):
        for space_name, space in [("HSV", cv2.COLOR_RGB2HSV), ("LAB", cv2.COLOR_RGB2LAB),
                                    ("YCrCb", cv2.COLOR_RGB2YCrCb)]:
            converted = cv2.cvtColor(image, space)
            for c in range(3):
                channel = converted[:, :, c]
                mask = channel > 0
                if mask.any():
                    vals = channel[mask]
                    stats = [float(vals.mean()), float(vals.std()),
                             float(np.percentile(vals, 10)), float(np.percentile(vals, 90)),
                             float(np.percentile(vals, 50)), float(vals.var())]
                    suffix = f"{space_name}_{['H','S','V'][c] if space_name=='HSV' else ['L','A','B'][c] if space_name=='LAB' else ['Y','Cr','Cb'][c]}"
                    fv.raw = np.concatenate([fv.raw, stats])
                    fv.names += [f"{suffix}_{s}" for s in ["mean","std","p10","p90","p50","var"]]

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        total = max((hsv[:, :, 2] > 0).sum(), 1)
        ranges = [((35, 85), "green"), ((20, 35), "yellow"),
                  ((5, 20), "orange"), ((0, 5), "red_brown")]
        for (lo, hi), name in ranges:
            mask = (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= hi) & (hsv[:, :, 1] > 25)
            fv.raw = np.append(fv.raw, float(mask.sum() / total))
            fv.names.append(f"color_{name}_ratio")

        ripeness = (fv.raw[fv.names.index("color_yellow_ratio")] +
                    2 * fv.raw[fv.names.index("color_orange_ratio")] +
                    3 * fv.raw[fv.names.index("color_red_brown_ratio")] -
                    fv.raw[fv.names.index("color_green_ratio")])
        fv.raw = np.append(fv.raw, ripeness)
        fv.names.append("ripeness_index")

    def _extract_glcm(self, image: np.ndarray, fv: FeatureVector):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = ((gray / gray.max()) * 255).astype(np.uint8) if gray.max() > 0 else gray

        glcm = graycomatrix(gray, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            symmetric=True, normed=True)
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
            vals = graycoprops(glcm, prop)
            fv.raw = np.append(fv.raw, float(vals.mean()))
            fv.raw = np.append(fv.raw, float(vals.std()))
            fv.names += [f"glcm_{prop}_mean", f"glcm_{prop}_std"]

    def _extract_lbp(self, image: np.ndarray, fv: FeatureVector):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = ((gray / gray.max()) * 255).astype(np.uint8) if gray.max() > 0 else gray

        for radius, n_points in [(1, 8), (2, 16), (3, 24)]:
            lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            fv.raw = np.concatenate([fv.raw, hist])
            fv.names += [f"lbp_r{radius}_{i}" for i in range(n_bins)]

    def _extract_hog(self, image: np.ndarray, fv: FeatureVector):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (64, 64))
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        ang = np.rad2deg(ang) % 180

        cell_size = 16
        n_bins = 9
        cells_x, cells_y = gray.shape[1] // cell_size, gray.shape[0] // cell_size
        grad_hist = np.zeros((cells_y, cells_x, n_bins))

        for cy in range(cells_y):
            for cx in range(cells_x):
                y1, y2 = cy * cell_size, (cy + 1) * cell_size
                x1, x2 = cx * cell_size, (cx + 1) * cell_size
                cell_mag = mag[y1:y2, x1:x2]
                cell_ang = ang[y1:y2, x1:x2]
                for b in range(n_bins):
                    lo, hi = b * 20, (b + 1) * 20
                    mask = ((cell_ang >= lo) & (cell_ang < hi)) | \
                           ((cell_ang >= lo + 180) & (cell_ang < hi + 180))
                    grad_hist[cy, cx, b] = cell_mag[mask].sum()

        for b in range(n_bins):
            fv.raw = np.append(fv.raw, float(grad_hist[:, :, b].mean()))
            fv.raw = np.append(fv.raw, float(grad_hist[:, :, b].std()))
            fv.names += [f"hog_mean_{b}", f"hog_std_{b}"]

    def _extract_spots(self, image: np.ndarray, fv: FeatureVector):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        brown_mask = cv2.inRange(hsv, np.array([5, 30, 20]), np.array([25, 255, 130]))
        black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        spots = cv2.bitwise_or(brown_mask, black_mask)

        total = max((gray > 0).sum(), 1)
        fv.raw = np.append(fv.raw, float(spots.sum() / total))
        fv.names.append("spot_ratio")

        contours, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fv.raw = np.append(fv.raw, float(len(contours)))
        fv.names.append("spot_count")

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            fv.raw = np.append(fv.raw, [float(np.mean(areas)), float(np.std(areas)),
                                         float(np.max(areas)), float(np.median(areas))])
            fv.names += ["spot_area_mean", "spot_area_std", "spot_area_max", "spot_area_median"]

            convexities = []
            for c in contours:
                area = cv2.contourArea(c)
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                convexities.append(area / hull_area if hull_area > 0 else 1)
            fv.raw = np.append(fv.raw, [float(np.mean(convexities)), float(np.std(convexities))])
            fv.names += ["spot_convexity_mean", "spot_convexity_std"]
        else:
            fv.raw = np.append(fv.raw, [0, 0, 0, 0, 0, 0])
            fv.names += ["spot_area_mean", "spot_area_std", "spot_area_max",
                          "spot_area_median", "spot_convexity_mean", "spot_convexity_std"]

    def _extract_dominant_colors(self, image: np.ndarray, fv: FeatureVector):
        pixels = image.reshape(-1, 3).astype(np.float32)
        mask = pixels.sum(axis=1) > 0
        if mask.sum() < 30:
            fv.raw = np.append(fv.raw, [0] * 24)
            fv.names += [f"dom_{i}_{c}" for i in range(6) for c in ["r","g","b","prop"]]
            return

        pixels = pixels[mask]
        self.kmeans.fit(pixels)
        labels = self.kmeans.labels_
        colors = self.kmeans.cluster_centers_

        counts = np.bincount(labels, minlength=6)
        props = counts / max(counts.sum(), 1)
        order = np.argsort(-counts)

        for idx in order:
            fv.raw = np.append(fv.raw, [colors[idx, 0], colors[idx, 1], colors[idx, 2], props[idx]])
            fv.names += [f"dom_{idx}_r", f"dom_{idx}_g", f"dom_{idx}_b", f"dom_{idx}_prop"]

    def _extract_morphology(self, image: np.ndarray, fv: FeatureVector):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

        if not contours:
            fv.raw = np.append(fv.raw, [0] * 10)
            fv.names += ["area", "perimeter", "circularity", "aspect_ratio", "extent",
                          "solidity", "equiv_diameter", "compactness", "rectangularity", "elongation"]
            return

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)

        equiv_diameter = np.sqrt(4 * area / np.pi)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        rectangularity = area / (w * h) if w * h > 0 else 0
        elongation = w / h if h > 0 else 0

        morph_feats = [
            float(area), float(perimeter), float(compactness),
            float(w / h) if h > 0 else 0, float(area / (w * h)) if w * h > 0 else 0,
            float(area / hull_area) if hull_area > 0 else 0,
            float(equiv_diameter), float(compactness),
            float(rectangularity), float(elongation),
        ]
        fv.raw = np.concatenate([fv.raw, morph_feats])
        fv.names += ["area", "perimeter", "circularity", "aspect_ratio", "extent",
                      "solidity", "equiv_diameter", "compactness", "rectangularity", "elongation"]

    def _extract_curvature(self, image: np.ndarray, fv: FeatureVector):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

        if not contours:
            fv.raw = np.append(fv.raw, [0, 0, 0, 0, 0])
            fv.names += ["curvature_mean", "curvature_std", "curvature_max",
                          "bending_energy", "skeleton_length"]
            return

        raw_c = max(contours, key=cv2.contourArea)
        c = raw_c.squeeze()
        if len(c.shape) < 2 or c.shape[0] < 5:
            fv.raw = np.append(fv.raw, [0, 0, 0, 0, 0])
            fv.names += ["curvature_mean", "curvature_std", "curvature_max",
                          "bending_energy", "skeleton_length"]
            return

        dx = np.gradient(c[:, 0])
        dy = np.gradient(c[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5 + 1e-8)

        fv.raw = np.append(fv.raw, [
            float(curvature.mean()), float(curvature.std()), float(curvature.max()),
            float((curvature ** 2).sum()), float(cv2.arcLength(raw_c, True)),
        ])
        fv.names += ["curvature_mean", "curvature_std", "curvature_max",
                      "bending_energy", "skeleton_length"]

    def _extract_fourier_descriptors(self, image: np.ndarray, fv: FeatureVector):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            fv.raw = np.append(fv.raw, [0] * 12)
            fv.names += [f"fd_{i}" for i in range(12)]
            return

        c = max(contours, key=cv2.contourArea)
        c = c.squeeze().astype(np.float64)
        if len(c.shape) < 2 or c.shape[0] < 8:
            fv.raw = np.append(fv.raw, [0] * 12)
            fv.names += [f"fd_{i}" for i in range(12)]
            return

        complex_contour = c[:, 0] + 1j * c[:, 1]
        fourier = np.fft.fft(complex_contour)
        descriptors = np.abs(fourier[1:13]) / max(np.abs(fourier[1]).item(), 1e-8)
        fv.raw = np.concatenate([fv.raw, descriptors])
        fv.names += [f"fd_{i}" for i in range(len(descriptors))]
