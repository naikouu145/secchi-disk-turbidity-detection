import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge

class VisibilityScoreCalculator:
    """
    Calculate composite visibility score from extracted features.
    Higher score means clearer water and lower turbidity.
    """

    def __init__(self, weights=None, method="balanced"):
        if weights is None:
            self.weights = self._get_default_weights(method)
        else:
            self.weights = weights

        self.method = method

        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            print(f"WARNING: Weights sum to {total_weight:.4f}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] /= total_weight

    def _get_default_weights(self, method):
        if method == "physics":
            return {
                "edge_clarity_canny": 0.30,
                "edge_clarity_sobel": 0.25,
                "edge_clarity_multiscale": 0.15,
                "sharpness_laplacian": 0.10,
                "sharpness_gradient": 0.05,
                "gradient_consistency": 0.05,
                "gradient_peak_ratio": 0.03,
                "contrast_std": 0.03,
                "contrast_range": 0.02,
                "yolo_confidence": 0.02,
                "disk_area_ratio": 0.00,
            }

        if method == "balanced":
            return {
                "edge_clarity_canny": 0.20,
                "edge_clarity_sobel": 0.18,
                "edge_clarity_multiscale": 0.15,
                "sharpness_laplacian": 0.12,
                "sharpness_gradient": 0.10,
                "gradient_consistency": 0.08,
                "gradient_peak_ratio": 0.07,
                "contrast_std": 0.05,
                "contrast_range": 0.03,
                "yolo_confidence": 0.02,
                "disk_area_ratio": 0.00,
            }

        if method == "edge_focused":
            return {
                "edge_clarity_canny": 0.35,
                "edge_clarity_sobel": 0.30,
                "edge_clarity_multiscale": 0.20,
                "sharpness_laplacian": 0.05,
                "sharpness_gradient": 0.03,
                "gradient_consistency": 0.03,
                "gradient_peak_ratio": 0.02,
                "contrast_std": 0.01,
                "contrast_range": 0.01,
                "yolo_confidence": 0.00,
                "disk_area_ratio": 0.00,
            }

        return self._get_default_weights("balanced")

    def calculate_score(self, normalized_features, adaptive=False):
        """Calculate weighted visibility score."""
        if adaptive:
            return self._calculate_adaptive_score(normalized_features)

        score = 0.0
        for feature_name, weight in self.weights.items():
            if feature_name in normalized_features:
                feature_value = normalized_features[feature_name]
                if np.isfinite(feature_value):
                    score += weight * feature_value

        return np.clip(score, 0.0, 1.0)

    def _calculate_adaptive_score(self, normalized_features):
        """Calculate score with adaptive weighting based on detection quality."""
        adjusted_weights = {}
        total_adjusted_weight = 0.0

        yolo_conf = normalized_features.get("yolo_confidence", 0.0)

        for feature_name, base_weight in self.weights.items():
            if feature_name not in normalized_features:
                continue

            feature_value = normalized_features[feature_name]
            if not np.isfinite(feature_value):
                continue

            if feature_name != "yolo_confidence":
                if yolo_conf < 0.3:
                    adjusted_weight = base_weight * (0.5 + 0.5 * yolo_conf / 0.3)
                else:
                    adjusted_weight = base_weight
            else:
                adjusted_weight = base_weight

            adjusted_weights[feature_name] = adjusted_weight
            total_adjusted_weight += adjusted_weight

        if total_adjusted_weight > 0:
            for key in adjusted_weights:
                adjusted_weights[key] /= total_adjusted_weight

        score = 0.0
        for feature_name, weight in adjusted_weights.items():
            score += weight * normalized_features[feature_name]

        return np.clip(score, 0.0, 1.0)

    def get_feature_contributions(self, normalized_features):
        """Get individual feature contribution to final score."""
        contributions = {}

        for feature_name, weight in self.weights.items():
            if feature_name in normalized_features:
                feature_value = normalized_features[feature_name]
                if np.isfinite(feature_value):
                    contributions[feature_name] = weight * feature_value
                else:
                    contributions[feature_name] = 0.0
            else:
                contributions[feature_name] = 0.0

        contributions = dict(
            sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        )
        return contributions

    def learn_weights_from_data(self, features_list, ground_truth_scores):
        """Learn optimal feature weights from labeled samples."""
        df = pd.DataFrame(features_list)

        x_values = df.values
        y_values = np.array(ground_truth_scores)

        model = Ridge(alpha=1.0, fit_intercept=False, positive=True)
        model.fit(x_values, y_values)

        coefficients = model.coef_
        weights_array = coefficients / coefficients.sum()

        learned_weights = {}
        for i, feature_name in enumerate(df.columns):
            learned_weights[feature_name] = max(0.0, weights_array[i])

        print("\nLearned Weights from Data:")
        print("-" * 60)
        for feature, weight in sorted(
            learned_weights.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {feature:30s}: {weight:.4f}")
        print("-" * 60)

        self.weights = learned_weights
        self.method = "learned"

        return learned_weights

    def estimate_ground_truth_from_secchi_depth(self, secchi_depth_m):
        """Convert Secchi depth measurement to visibility score."""
        if secchi_depth_m >= 6.0:
            score = 0.9 + (min(secchi_depth_m, 10) - 6) / 4 * 0.1
        elif secchi_depth_m >= 3.0:
            score = 0.7 + (secchi_depth_m - 3.0) / 3.0 * 0.2
        elif secchi_depth_m >= 1.5:
            score = 0.5 + (secchi_depth_m - 1.5) / 1.5 * 0.2
        elif secchi_depth_m >= 0.5:
            score = 0.2 + (secchi_depth_m - 0.5) / 1.0 * 0.3
        else:
            score = min(secchi_depth_m / 0.5 * 0.2, 0.2)

        return np.clip(score, 0.0, 1.0)

    def compare_weighting_methods(self, normalized_features):
        """Compare output scores across built-in weighting methods."""
        methods = ["physics", "balanced", "edge_focused"]
        scores = {}

        for method in methods:
            temp_weights = self._get_default_weights(method)

            score = 0.0
            for feature_name, weight in temp_weights.items():
                if feature_name in normalized_features:
                    feature_value = normalized_features[feature_name]
                    if np.isfinite(feature_value):
                        score += weight * feature_value

            scores[method] = np.clip(score, 0.0, 1.0)

        scores["current"] = self.calculate_score(normalized_features)
        return scores

    def print_weights(self):
        """Print current weights in a formatted table."""
        print(f"\nCurrent Weighting Method: {self.method.upper()}")
        print("=" * 60)
        print(f"{'Feature':<30s} {'Weight':>10s}")
        print("-" * 60)

        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        for feature, weight in sorted_weights:
            print(f"{feature:<30s} {weight:>10.4f}")

        print("-" * 60)
        print(f"{'TOTAL':<30s} {sum(self.weights.values()):>10.4f}")
        print("=" * 60 + "\n")
