import cv2
import numpy as np
import pandas as pd

from scipy import ndimage

class SecchiDiskFeatureExtractor:
    """
    Extract visual features from secchi disk region for turbidity assessment.
    Focus: edge clarity, sharpness, contrast.
    """

    def __init__(self):
        self.feature_names = [
            "edge_clarity_canny",
            "edge_clarity_sobel",
            "edge_clarity_multiscale",
            "sharpness_laplacian",
            "sharpness_gradient",
            "gradient_consistency",
            "gradient_peak_ratio",
            "contrast_std",
            "contrast_range",
            "yolo_confidence",
            "disk_area_ratio",
        ]

    def extract_features(self, image, bbox, yolo_confidence):
        """Extract all features from secchi disk region."""
        x1, y1, x2, y2 = map(int, bbox)

        if x2 <= x1 or y2 <= y1:
            return self._default_features()

        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        disk_region = image[y1:y2, x1:x2]
        if disk_region.size == 0:
            return self._default_features()

        gray = cv2.cvtColor(disk_region, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float64)

        features = {}

        edges_canny = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges_canny > 0)
        total_pixels = edges_canny.size

        if edge_pixels > 0:
            edge_density = edge_pixels / total_pixels
            edge_strength = np.mean(edges_canny[edges_canny > 0])
            features["edge_clarity_canny"] = edge_density * (edge_strength / 255.0)
        else:
            features["edge_clarity_canny"] = 0.0

        sobelx = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)

        sobelx = np.clip(sobelx, -1e10, 1e10)
        sobely = np.clip(sobely, -1e10, 1e10)

        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features["edge_clarity_sobel"] = np.mean(sobel_magnitude) / 255.0

        features["edge_clarity_multiscale"] = self._extract_multiscale_edge_clarity(gray)

        laplacian = cv2.Laplacian(gray_float, cv2.CV_64F)
        laplacian = np.clip(laplacian, -1e10, 1e10)
        features["sharpness_laplacian"] = float(np.var(laplacian))

        gradient_x = ndimage.sobel(gray_float, axis=1)
        gradient_y = ndimage.sobel(gray_float, axis=0)

        gradient_x = np.clip(gradient_x, -1e10, 1e10)
        gradient_y = np.clip(gradient_y, -1e10, 1e10)

        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        features["sharpness_gradient"] = float(np.var(gradient_magnitude))

        gradient_angles = np.arctan2(sobely, sobelx)
        angle_std = self._circular_std(gradient_angles.flatten())
        features["gradient_consistency"] = 1.0 / (1.0 + angle_std)

        gradient_threshold = np.percentile(sobel_magnitude, 90)
        gradient_peaks = sobel_magnitude > gradient_threshold
        features["gradient_peak_ratio"] = np.sum(gradient_peaks) / total_pixels

        features["contrast_std"] = float(np.std(gray.astype(np.float64)))
        features["contrast_range"] = float(gray.max() - gray.min())

        disk_area = (x2 - x1) * (y2 - y1)
        image_area = h * w
        features["disk_area_ratio"] = disk_area / image_area

        features["yolo_confidence"] = float(yolo_confidence)

        return features

    def _extract_multiscale_edge_clarity(self, gray):
        """Extract edge clarity at multiple scales."""
        clarities = []

        for kernel_size in [3, 5, 7]:
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            edges = cv2.Canny(blurred, 50, 150)
            clarity = np.sum(edges > 0) / edges.size
            clarities.append(clarity)

        if len(clarities) == 3:
            multiscale_score = (
                0.5 * clarities[0] + 0.3 * clarities[1] + 0.2 * clarities[2]
            )
        else:
            multiscale_score = np.mean(clarities)

        return multiscale_score

    def _circular_std(self, angles):
        """Calculate circular standard deviation for angles."""
        angles = angles[~np.isnan(angles)]

        if len(angles) == 0:
            return np.pi

        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))

        r_value = np.sqrt(sin_mean**2 + cos_mean**2)
        circular_std = np.sqrt(-2 * np.log(r_value + 1e-10))

        return circular_std

    def _default_features(self):
        """Return default features when extraction fails."""
        return {name: 0.0 for name in self.feature_names}

    def normalize_features(self, features, normalization_params=None):
        """Normalize features to 0-1 range for composite scoring."""
        if normalization_params is None:
            normalization_params = {
                "edge_clarity_canny": 0.15,
                "edge_clarity_sobel": 1.0,
                "edge_clarity_multiscale": 0.12,
                "sharpness_laplacian": 500,
                "sharpness_gradient": 1000,
                "gradient_consistency": 1.0,
                "gradient_peak_ratio": 0.20,
                "contrast_std": 60,
                "contrast_range": 200,
                "yolo_confidence": 1.0,
                "disk_area_ratio": 0.3,
            }

        normalized = {}
        for key, value in features.items():
            max_val = normalization_params.get(key, 1.0)
            if np.isfinite(value):
                normalized[key] = min(value / max_val, 1.0)
            else:
                normalized[key] = 0.0

        return normalized

    def calibrate_normalization_params(self, calibration_images, yolo_model):
        """Build normalization parameters from actual data distribution."""
        all_features = []

        print(f"\nCalibrating normalization parameters from {len(calibration_images)} images...")

        for img_path in calibration_images:
            img = cv2.imread(img_path)
            if img is None:
                continue

            results = yolo_model.predict(source=img_path, conf=0.15, verbose=False)
            if len(results[0].boxes) == 0:
                continue

            box = results[0].boxes[0]
            yolo_confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()

            features = self.extract_features(img, bbox, yolo_confidence)
            all_features.append(features)

        if len(all_features) == 0:
            print("WARNING: No features extracted. Using default normalization.")
            return None

        df = pd.DataFrame(all_features)
        normalization_params = {}

        print("\nCalibrated Normalization Parameters (95th percentile):")
        print("-" * 60)

        for col in df.columns:
            p95 = df[col].quantile(0.95)
            if p95 == 0:
                p95 = df[col].max()
            if p95 == 0:
                p95 = 1.0

            normalization_params[col] = p95
            print(f"  {col:30s}: {p95:.4f}")

        print("-" * 60)
        print(f"✓ Calibration complete using {len(all_features)} samples\n")

        return normalization_params
