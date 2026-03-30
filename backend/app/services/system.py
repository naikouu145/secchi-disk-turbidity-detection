import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from app.core.config import AppConfig

from .classifier import TurbidityClassifier
from .feature_extraction import SecchiDiskFeatureExtractor
from .score_calculator import VisibilityScoreCalculator


class SecchiTurbiditySystem:
    """End-to-end source-aware turbidity assessment system."""

    def __init__(
        self,
        yolo_model_path=None,
        standard=None,
        weighting_method=None,
        normalization_params=None,
        config=None,
    ):
        self.config = config or AppConfig.from_env()

        resolved_model_path = yolo_model_path or str(self.config.model_path)
        resolved_standard = standard or self.config.default_standard
        resolved_weighting_method = (
            weighting_method or self.config.default_weighting_method
        )

        self.upload_dirs = self.config.ensure_upload_directories()

        if normalization_params is None:
            normalization_params = self.config.load_normalization_parameters()

        self.yolo = YOLO(resolved_model_path)

        self.feature_extractor = SecchiDiskFeatureExtractor()
        self.score_calculator = VisibilityScoreCalculator(method=resolved_weighting_method)
        self.classifier = TurbidityClassifier(standard=resolved_standard)

        self.normalization_params = normalization_params

    def calibrate(self, calibration_images, output_path=None):
        """Calibrate normalization parameters from sample images."""
        norm_params = self.feature_extractor.calibrate_normalization_params(
            calibration_images, self.yolo
        )

        self.normalization_params = norm_params

        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(norm_params, file, indent=2)
        elif norm_params is not None:
            self.config.save_normalization_parameters(norm_params)

        return norm_params

    def assess_single_image(
        self,
        image_path,
        visualize=False,
        verbose=True,
        adaptive_scoring=False,
        override_source=None,
    ):
        """Run full source-aware turbidity assessment for one image."""
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Failed to load image: {image_path}"}

        results = self.yolo.predict(source=image_path, conf=0.15, verbose=False)

        if len(results[0].boxes) == 0:
            result = {
                "image_path": image_path,
                "disk_detected": False,
                "turbidity_category": "Very High Turbidity",
                "visibility_score": 0.0,
                "confidence": "High",
                "confidence_numeric": 1.0,
                "message": "Secchi disk not visible - water is very turbid",
                "turbidity_source": "unknown",
                "standard_used": "N/A",
            }

            if verbose:
                self._print_result(result)

            return result

        box = results[0].boxes[0]
        yolo_confidence = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy()

        features = self.feature_extractor.extract_features(img, bbox, yolo_confidence)
        normalized_features = self.feature_extractor.normalize_features(
            features, self.normalization_params
        )

        visibility_score = self.score_calculator.calculate_score(
            normalized_features, adaptive=adaptive_scoring
        )

        if override_source:
            turbidity_source = {
                "primary_source": override_source,
                "confidence": 1.0,
                "note": "User override",
            }
        elif self.classifier.auto_mode or verbose:
            turbidity_source = self.classifier.source_detector.detect_source(
                img, bbox, features
            )
        else:
            turbidity_source = None

        classification_result = self.classifier.classify(
            visibility_score,
            turbidity_source=turbidity_source,
            image=img,
            bbox=bbox,
            features=features,
        )

        turbidity_category = classification_result["category"]
        standard_used = classification_result["standard_used"]

        classification_confidence = self.classifier.get_confidence(visibility_score)
        confidence_numeric = self.classifier.get_confidence_numeric(visibility_score)

        prob_classification = self.classifier.get_probabilistic_classification(
            visibility_score
        )
        contributions = self.score_calculator.get_feature_contributions(normalized_features)

        category_info = self.classifier.get_category_info(
            turbidity_category, turbidity_source
        )

        equivalent_metrics = self.classifier.get_equivalent_metrics(
            visibility_score, turbidity_source
        )

        method_comparison = self.score_calculator.compare_weighting_methods(
            normalized_features
        )

        result = {
            "image_path": image_path,
            "disk_detected": True,
            "turbidity_source": turbidity_source,
            "standard_used": standard_used,
            "turbidity_category": turbidity_category,
            "visibility_score": visibility_score,
            "confidence": classification_confidence,
            "confidence_numeric": confidence_numeric,
            "yolo_confidence": yolo_confidence,
            "bbox": bbox.tolist(),
            "edge_clarity": features["edge_clarity_canny"],
            "edge_clarity_multiscale": features["edge_clarity_multiscale"],
            "sharpness": features["sharpness_laplacian"],
            "contrast": features["contrast_std"],
            "gradient_consistency": features["gradient_consistency"],
            "features": features,
            "normalized_features": normalized_features,
            "feature_contributions": contributions,
            "category_info": category_info,
            "equivalent_metrics": equivalent_metrics,
            "probabilistic_classification": prob_classification,
            "method_comparison": method_comparison,
        }

        if verbose:
            self._print_result(result)

        if visualize:
            self._visualize_result(img, result)

        return result

    def _print_result(self, result):
        """Print source-aware assessment results."""
        print("\n" + "=" * 70)
        print("SOURCE-AWARE TURBIDITY ASSESSMENT")
        print("=" * 70)

        if not result["disk_detected"]:
            print(f"Status: {result['message']}")
            print(f"Category: {result['turbidity_category']}")
        else:
            if result.get("turbidity_source"):
                source_info = result["turbidity_source"]
                print("\nTURBIDITY SOURCE DETECTED:")
                print(
                    f"  Primary Source: {source_info.get('primary_source', 'unknown').upper()}"
                )
                print(f"  Detection Confidence: {source_info.get('confidence', 0):.2f}")

                if "algal_score" in source_info:
                    print(f"  Algal Score: {source_info['algal_score']:.2f}")
                    print(f"  Sediment Score: {source_info['sediment_score']:.2f}")

                print(f"\nStandard Selected: {result.get('standard_used', 'N/A').upper()}")

            print("\nCLASSIFICATION:")
            print(f"  Category: {result['turbidity_category']}")
            print(f"  Visibility Score: {result['visibility_score']:.3f}")
            print(
                f"  Confidence: {result['confidence']} ({result['confidence_numeric']:.2f})"
            )

            if "probabilistic_classification" in result:
                prob_class = result["probabilistic_classification"]
                if len([p for p in prob_class.values() if p > 0.01]) > 1:
                    print("\n  Probabilistic Breakdown:")
                    for cat, prob in sorted(
                        prob_class.items(), key=lambda x: x[1], reverse=True
                    ):
                        if prob > 0.01:
                            print(f"    {cat}: {prob * 100:.1f}%")

            print("\nDETECTION:")
            print(f"  YOLO Confidence: {result['yolo_confidence']:.3f}")

            print("\nKEY FEATURES:")
            print(f"  Edge Clarity (Single): {result['edge_clarity']:.4f}")
            print(
                f"  Edge Clarity (Multi-scale): {result['edge_clarity_multiscale']:.4f}"
            )
            print(f"  Gradient Consistency: {result['gradient_consistency']:.4f}")
            print(f"  Sharpness: {result['sharpness']:.2f}")
            print(f"  Contrast: {result['contrast']:.2f}")

            print("\nTOP CONTRIBUTING FEATURES:")
            for i, (feat, contrib) in enumerate(
                list(result["feature_contributions"].items())[:3], 1
            ):
                print(f"  {i}. {feat}: {contrib:.4f}")

            if "equivalent_metrics" in result:
                metrics = result["equivalent_metrics"]
                print("\nESTIMATED WATER QUALITY METRICS:")

                if "ntu_range" in metrics:
                    print(
                        f"  Turbidity (NTU): {metrics['estimated_ntu']} NTU "
                        f"[{metrics['ntu_range'][0]}-{metrics['ntu_range'][1]}]"
                    )
                    print(
                        f"  Secchi Depth: {metrics['estimated_secchi_depth_m']} m "
                        f"[{metrics['secchi_depth_range'][0]}-{metrics['secchi_depth_range'][1]}]"
                    )

                    if metrics.get("tsi_applicable", False):
                        print(
                            f"  Carlson TSI: {metrics['estimated_carlson_tsi']} "
                            f"[{metrics['tsi_range'][0]}-{metrics['tsi_range'][1]}]"
                        )
                    else:
                        print(
                            f"  Carlson TSI: N/A ({metrics.get('tsi_note', 'Not applicable')})"
                        )

                    print(f"  Estimate Confidence: {metrics['confidence']:.2f}")

            if "category_info" in result and result["category_info"]:
                info = result["category_info"]
                print("\nCATEGORY INFORMATION:")
                print(f"  Description: {info.get('description', 'N/A')}")
                print(f"  Visibility: {info.get('visibility', 'N/A')}")
                print(f"  Typical Causes: {info.get('typical_causes', 'N/A')}")
                print(f"  NTU Range: {info.get('ntu_range', 'N/A')}")
                print(f"  Secchi Depth: {info.get('secchi_depth_equivalent', 'N/A')}")

                if "trophic_state" in info and info["trophic_state"] != "N/A":
                    print(f"  Trophic State: {info['trophic_state']}")

                if "management_action" in info:
                    print("\nMANAGEMENT RECOMMENDATION:")
                    print(f"  {info['management_action']}")

                if "ecological_impact" in info:
                    print("\nECOLOGICAL IMPACT:")
                    print(f"  {info['ecological_impact']}")

        print("=" * 70 + "\n")

    def _visualize_result(self, img, result):
        """Visualize detection and assessment details."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        img_annotated = img.copy()

        if result["disk_detected"]:
            bbox = result["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            color = self._get_category_color(result["turbidity_category"])
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)

            label = f"{result['turbidity_category']}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(
                img_annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                img_annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            if result.get("turbidity_source"):
                source = result["turbidity_source"].get("primary_source", "unknown")
                source_label = f"Source: {source.capitalize()}"
                cv2.putText(
                    img_annotated,
                    source_label,
                    (x1, y1 - label_size[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

            score_label = f"Score: {result['visibility_score']:.2f}"
            score_size = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                img_annotated,
                (x1, y2 + 5),
                (x1 + score_size[0], y2 + score_size[1] + 10),
                color,
                -1,
            )
            cv2.putText(
                img_annotated,
                score_label,
                (x1, y2 + score_size[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        axes[0].imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
        axes[0].axis("off")

        source_str = ""
        if result.get("turbidity_source"):
            source = result["turbidity_source"].get("primary_source", "unknown")
            source_str = f" ({source.capitalize()})"
        axes[0].set_title(f"Detection Result{source_str}", fontsize=14, fontweight="bold")

        if result["disk_detected"]:
            contributions = result["feature_contributions"]
            top_features = dict(list(contributions.items())[:8])

            features = list(top_features.keys())
            values = list(top_features.values())

            colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
            axes[1].barh(
                features, values, color=colors_bar, edgecolor="black", linewidth=1.5
            )
            axes[1].set_xlabel("Contribution", fontsize=12, fontweight="bold")
            axes[1].set_title("Feature Contributions", fontsize=14, fontweight="bold")
            axes[1].set_xlim([0, max(values) * 1.2] if values else [0, 1])
            axes[1].grid(True, alpha=0.3, axis="x")

            for i, (_, value) in enumerate(zip(features, values)):
                axes[1].text(value + 0.005, i, f"{value:.3f}", va="center", fontweight="bold", fontsize=9)
        else:
            axes[1].text(
                0.5,
                0.5,
                "Disk Not Detected",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
            )
            axes[1].axis("off")

        if result["disk_detected"] and result.get("turbidity_source"):
            source_info = result["turbidity_source"]

            if "algal_score" in source_info and "sediment_score" in source_info:
                sources = ["Algal", "Sediment"]
                scores = [
                    source_info["algal_score"] * 100,
                    source_info["sediment_score"] * 100,
                ]

                colors_source = ["#2ecc71", "#95a5a6"]

                axes[2].barh(
                    sources,
                    scores,
                    color=colors_source,
                    edgecolor="black",
                    linewidth=1.5,
                    alpha=0.8,
                )
                axes[2].set_xlabel("Source Score (%)", fontsize=12, fontweight="bold")
                axes[2].set_title("Turbidity Source Analysis", fontsize=14, fontweight="bold")
                axes[2].set_xlim([0, 105])
                axes[2].grid(True, alpha=0.3, axis="x")

                for i, (_, score) in enumerate(zip(sources, scores)):
                    axes[2].text(
                        score + 2,
                        i,
                        f"{score:.1f}%",
                        va="center",
                        fontweight="bold",
                        fontsize=10,
                    )

                primary = source_info.get("primary_source", "mixed").capitalize()
                axes[2].text(
                    0.5,
                    -0.15,
                    f"Primary: {primary}",
                    transform=axes[2].transAxes,
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                    bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
                )

        plt.tight_layout()
        plt.show()

    def _get_category_color(self, category):
        """Get BGR color by category."""
        colors = {
            "Clear Water": (0, 255, 0),
            "Slightly Turbid": (0, 255, 255),
            "Moderately Turbid": (0, 165, 255),
            "High Turbidity": (0, 0, 255),
            "Very High Turbidity": (128, 0, 128),
        }
        return colors.get(category, (255, 255, 255))

    def assess_batch(
        self,
        image_paths,
        save_results=True,
        output_path=None,
        adaptive_scoring=False,
        show_progress=True,
    ):
        """Batch processing with source detection."""
        results = []

        print(f"\nProcessing {len(image_paths)} images (SOURCE-AWARE MODE)...")
        print("-" * 70)

        for i, img_path in enumerate(image_paths, 1):
            if show_progress:
                print(f"\n[{i}/{len(image_paths)}] {Path(img_path).name}")

            result = self.assess_single_image(
                img_path,
                visualize=False,
                verbose=False,
                adaptive_scoring=adaptive_scoring,
            )
            results.append(result)

            if show_progress and result["disk_detected"]:
                source = result.get("turbidity_source", {}).get("primary_source", "unknown")
                std = result.get("standard_used", "N/A")
                print(
                    f"  -> {result['turbidity_category']} "
                    f"(score: {result['visibility_score']:.3f}, "
                    f"source: {source}, std: {std})"
                )
            elif show_progress:
                print(f"  -> {result['turbidity_category']}")

        df = self._results_to_dataframe(results)

        if save_results:
            if output_path is None:
                output_path = "turbidity_assessment_source_aware.csv"
            df.to_csv(output_path, index=False)
            print(f"\n✓ Results saved to: {output_path}")

        self._print_batch_summary(df)

        return df

    def _results_to_dataframe(self, results):
        """Convert results to a DataFrame."""
        df_data = []

        for item in results:
            row = {
                "image_path": Path(item["image_path"]).name,
                "disk_detected": item["disk_detected"],
                "turbidity_source": item.get("turbidity_source", {}).get(
                    "primary_source", "unknown"
                ),
                "source_confidence": item.get("turbidity_source", {}).get(
                    "confidence", 0.0
                ),
                "standard_used": item.get("standard_used", "N/A"),
                "turbidity_category": item["turbidity_category"],
                "visibility_score": item.get("visibility_score", 0.0),
                "confidence": item.get("confidence", "N/A"),
                "confidence_numeric": item.get("confidence_numeric", 0.0),
                "yolo_confidence": item.get("yolo_confidence", 0.0),
                "edge_clarity": item.get("edge_clarity", 0.0),
                "edge_clarity_multiscale": item.get("edge_clarity_multiscale", 0.0),
                "gradient_consistency": item.get("gradient_consistency", 0.0),
                "sharpness": item.get("sharpness", 0.0),
                "contrast": item.get("contrast", 0.0),
            }

            if "equivalent_metrics" in item:
                metrics = item["equivalent_metrics"]
                row["estimated_ntu"] = metrics.get("estimated_ntu", None)
                row["estimated_secchi_depth_m"] = metrics.get(
                    "estimated_secchi_depth_m", None
                )
                row["estimated_carlson_tsi"] = metrics.get("estimated_carlson_tsi", None)
                row["tsi_applicable"] = metrics.get("tsi_applicable", False)

            df_data.append(row)

        return pd.DataFrame(df_data)

    def _print_batch_summary(self, df):
        """Print batch summary with source distribution."""
        print("\n" + "=" * 70)
        print("BATCH PROCESSING SUMMARY (SOURCE-AWARE)")
        print("=" * 70)

        print(f"\nTotal images: {len(df)}")
        print(
            f"Disks detected: {df['disk_detected'].sum()} "
            f"({df['disk_detected'].sum() / len(df) * 100:.1f}%)"
        )

        if "turbidity_source" in df.columns:
            print("\nTurbidity Source Distribution:")
            source_counts = df[df["disk_detected"]]["turbidity_source"].value_counts()
            for source, count in source_counts.items():
                print(
                    f"  {source.capitalize()}: {count} "
                    f"({count / df['disk_detected'].sum() * 100:.1f}%)"
                )

        if "standard_used" in df.columns:
            print("\nStandards Applied:")
            std_counts = df[df["disk_detected"]]["standard_used"].value_counts()
            for std, count in std_counts.items():
                print(f"  {std.upper()}: {count}")

        print("\nTurbidity Category Distribution:")
        category_counts = df["turbidity_category"].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} ({count / len(df) * 100:.1f}%)")

        detected_df = df[df["disk_detected"]]

        if len(detected_df) > 0:
            print("\nVisibility Score Statistics:")
            print(f"  Mean: {detected_df['visibility_score'].mean():.3f}")
            print(f"  Std: {detected_df['visibility_score'].std():.3f}")
            print(
                f"  Range: [{detected_df['visibility_score'].min():.3f}, "
                f"{detected_df['visibility_score'].max():.3f}]"
            )

        print("=" * 70 + "\n")

    def export_config(self, output_path):
        """Export current source-aware configuration to JSON."""
        config = {
            "mode": "source_aware" if self.classifier.auto_mode else "fixed",
            "standard": self.classifier.standard,
            "weighting_method": self.score_calculator.method,
            "weights": self.score_calculator.weights,
            "normalization_params": self.normalization_params,
            "source_detection_enabled": self.classifier.auto_mode,
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
