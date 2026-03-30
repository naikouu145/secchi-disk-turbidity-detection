import cv2
import numpy as np

class TurbiditySourceDetector:
    """
    Detect the primary source of turbidity from visual features.

    Sources:
    1. algal/biological
    2. sediment/mineral
    3. mixed
    """

    def __init__(self):
        self.source_types = ["algal", "sediment", "mixed"]

    def detect_source(self, image, bbox, features):
        """Detect turbidity source from visual characteristics."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]

        margin = int((x2 - x1) * 0.3)
        water_x1 = max(0, x1 - margin)
        water_y1 = max(0, y1 - margin)
        water_x2 = min(w, x2 + margin)
        water_y2 = min(h, y2 + margin)

        water_region = image[water_y1:water_y2, water_x1:water_x2]
        if water_region.size == 0:
            return self._default_source()

        lab = cv2.cvtColor(water_region, cv2.COLOR_BGR2LAB)
        _, a_channel, b_channel = cv2.split(lab)

        mean_a = np.mean(a_channel)
        mean_b = np.mean(b_channel)

        saturation = np.sqrt((mean_a - 128) ** 2 + (mean_b - 128) ** 2)
        greenness = max(0, mean_b - 128) / 128.0
        redness = max(0, mean_a - 128) / 128.0

        algal_color_score = (saturation / 50.0) * (greenness + redness * 0.5)
        algal_color_score = np.clip(algal_color_score, 0, 1)

        sediment_color_score = 1.0 - np.clip(saturation / 30.0, 0, 1)

        gray = cv2.cvtColor(water_region, cv2.COLOR_BGR2GRAY)

        kernel_size = 5
        mean_filtered = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        mean_sq_filtered = cv2.blur((gray.astype(np.float32)) ** 2, (kernel_size, kernel_size))
        local_variance = mean_sq_filtered - mean_filtered**2
        local_variance = np.clip(local_variance, 0, None)

        texture_heterogeneity = np.mean(np.sqrt(local_variance))

        sediment_texture_score = np.clip(texture_heterogeneity / 20.0, 0, 1)
        algal_texture_score = 1.0 - sediment_texture_score

        contrast = features.get("contrast_range", 0)

        if contrast > 150:
            sediment_contrast_score = 1.0
            algal_contrast_score = 0.0
        elif contrast > 100:
            sediment_contrast_score = 0.8
            algal_contrast_score = 0.2
        elif contrast > 75:
            sediment_contrast_score = 0.5
            algal_contrast_score = 0.5
        else:
            sediment_contrast_score = 0.2
            algal_contrast_score = 0.8

        algal_score = (
            0.5 * algal_color_score
            + 0.3 * algal_texture_score
            + 0.2 * algal_contrast_score
        )

        sediment_score = (
            0.5 * sediment_color_score
            + 0.3 * sediment_texture_score
            + 0.2 * sediment_contrast_score
        )

        total = algal_score + sediment_score
        if total > 0:
            algal_score /= total
            sediment_score /= total

        if algal_score > 0.65:
            primary_source = "algal"
            confidence = algal_score
        elif sediment_score > 0.65:
            primary_source = "sediment"
            confidence = sediment_score
        else:
            primary_source = "mixed"
            confidence = 1.0 - abs(algal_score - sediment_score)

        return {
            "primary_source": primary_source,
            "confidence": float(confidence),
            "algal_score": float(algal_score),
            "sediment_score": float(sediment_score),
            "color_indicators": {
                "saturation": float(saturation),
                "greenness": float(greenness),
                "algal_color": float(algal_color_score),
                "sediment_color": float(sediment_color_score),
            },
            "texture_score": float(texture_heterogeneity),
            "contrast_score": float(contrast),
        }

    def _default_source(self):
        """Return default source values when detection fails."""
        return {
            "primary_source": "unknown",
            "confidence": 0.0,
            "algal_score": 0.5,
            "sediment_score": 0.5,
            "color_indicators": {},
            "texture_score": 0.0,
            "contrast_score": 0.0,
        }
