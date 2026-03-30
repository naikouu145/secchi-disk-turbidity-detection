import math
import numpy as np

from .source_detector import TurbiditySourceDetector

class TurbidityClassifier:
    """Source-aware turbidity classifier."""

    def __init__(self, standard="auto", thresholds=None, categories=None):
        self.standards = {
            "carlson": {
                "thresholds": [0.25, 0.50, 0.75],
                "categories": [
                    "High Turbidity",
                    "Moderately Turbid",
                    "Slightly Turbid",
                    "Clear Water",
                ],
                "reference": "Carlson Trophic State Index (1977)",
                "note": "Assumes algal/biological turbidity. May overestimate for sediment.",
                "application": "Algal-dominated lakes and reservoirs",
                "best_for": "algal",
            },
            "sediment": {
                "thresholds": [0.20, 0.45, 0.70],
                "categories": [
                    "High Turbidity",
                    "Moderately Turbid",
                    "Slightly Turbid",
                    "Clear Water",
                ],
                "reference": "Sediment-based turbidity standards (USGS)",
                "note": "Adjusted for mineral/sediment turbidity. Stricter than Carlson.",
                "application": "Sediment-dominated rivers, streams, post-storm events",
                "best_for": "sediment",
            },
            "epa": {
                "thresholds": [0.20, 0.40, 0.60],
                "categories": [
                    "High Turbidity",
                    "Moderately Turbid",
                    "Slightly Turbid",
                    "Clear Water",
                ],
                "reference": "EPA Water Quality Standards",
                "note": "General purpose, balanced for mixed sources",
                "application": "General freshwater bodies, regulatory compliance",
                "best_for": "mixed",
            },
            "marine": {
                "thresholds": [0.15, 0.35, 0.65],
                "categories": [
                    "High Turbidity",
                    "Moderately Turbid",
                    "Slightly Turbid",
                    "Clear Water",
                ],
                "reference": "Oceanographic standards",
                "note": "Stricter thresholds for coastal/marine environments",
                "application": "Coastal waters, estuaries, marine environments",
                "best_for": "mixed",
            },
            "freshwater": {
                "thresholds": [0.30, 0.55, 0.80],
                "categories": [
                    "High Turbidity",
                    "Moderately Turbid",
                    "Slightly Turbid",
                    "Clear Water",
                ],
                "reference": "Freshwater lake standards",
                "note": "Based on typical freshwater clarity ranges",
                "application": "Rivers, streams, freshwater lakes",
                "best_for": "mixed",
            },
        }

        self.source_detector = TurbiditySourceDetector()
        self.auto_mode = standard == "auto"

        if standard == "auto":
            self._set_standard("epa")
        elif standard in self.standards:
            self._set_standard(standard)
        elif standard == "custom":
            if thresholds is None or categories is None:
                raise ValueError("Must provide thresholds and categories for custom standard")
            self.standard = "custom"
            self.thresholds = sorted(thresholds)
            self.categories = categories
            self.reference = "Custom user-defined"
            self.note = "User-defined thresholds"
            self.application = "Custom application"
            self.best_for = "custom"
        else:
            raise ValueError(f"Unknown standard: {standard}")

        if len(self.categories) != len(self.thresholds) + 1:
            raise ValueError(
                f"Need {len(self.thresholds) + 1} categories for {len(self.thresholds)} thresholds"
            )

        self.threshold_spacing = self._calculate_threshold_spacing()

    def _set_standard(self, standard_name):
        """Set an internal standard by name."""
        std_info = self.standards[standard_name]
        self.standard = standard_name
        self.thresholds = std_info["thresholds"]
        self.categories = std_info["categories"]
        self.reference = std_info["reference"]
        self.note = std_info["note"]
        self.application = std_info["application"]
        self.best_for = std_info["best_for"]

    def _calculate_threshold_spacing(self):
        """Calculate average spacing between thresholds."""
        if len(self.thresholds) < 2:
            return 0.25
        spacings = np.diff(sorted(self.thresholds))
        return float(np.mean(spacings))

    def classify(
        self,
        visibility_score,
        turbidity_source=None,
        image=None,
        bbox=None,
        features=None,
    ):
        """Classify turbidity with source awareness."""
        result = {}

        if self.auto_mode:
            if turbidity_source is None:
                if image is not None and bbox is not None and features is not None:
                    turbidity_source = self.source_detector.detect_source(image, bbox, features)
                else:
                    turbidity_source = {"primary_source": "mixed", "confidence": 0.5}

            source = turbidity_source["primary_source"]
            if source == "algal":
                self._set_standard("carlson")
            elif source == "sediment":
                self._set_standard("sediment")
            else:
                self._set_standard("epa")

            self.threshold_spacing = self._calculate_threshold_spacing()
            result["source_info"] = turbidity_source
            result["standard_used"] = self.standard
        else:
            result["standard_used"] = self.standard
            if turbidity_source is not None:
                result["source_info"] = turbidity_source

        for i, threshold in enumerate(self.thresholds):
            if visibility_score < threshold:
                result["category"] = self.categories[i]
                return result

        result["category"] = self.categories[-1]
        return result

    def get_confidence(self, visibility_score):
        """Adaptive confidence label."""
        distances = [abs(visibility_score - t) for t in self.thresholds]
        min_distance = min(distances) if distances else 1.0

        high_threshold = self.threshold_spacing * 0.40
        medium_threshold = self.threshold_spacing * 0.20

        if min_distance > high_threshold:
            return "High"
        if min_distance > medium_threshold:
            return "Medium"
        return "Low"

    def get_confidence_numeric(self, visibility_score):
        """Numeric confidence score in range 0..1."""
        distances = [abs(visibility_score - t) for t in self.thresholds]
        min_distance = min(distances) if distances else 1.0

        if self.threshold_spacing > 0:
            confidence = min(min_distance / self.threshold_spacing, 1.0)
        else:
            confidence = 0.5

        return confidence

    def get_probabilistic_classification(self, visibility_score):
        """Probabilistic classification for borderline cases."""
        distances = [abs(visibility_score - t) for t in self.thresholds]
        probabilities = {cat: 0.0 for cat in self.categories}

        primary_idx = 0
        for i, threshold in enumerate(self.thresholds):
            if visibility_score >= threshold:
                primary_idx = i + 1

        min_distance = min(distances) if distances else 1.0

        if min_distance > self.threshold_spacing * 0.5:
            probabilities[self.categories[primary_idx]] = 1.0
            return probabilities

        nearest_threshold_idx = distances.index(min(distances))
        nearest_threshold = self.thresholds[nearest_threshold_idx]

        sigma = self.threshold_spacing * 0.2

        if visibility_score < nearest_threshold:
            primary_idx = nearest_threshold_idx
            secondary_idx = max(0, nearest_threshold_idx - 1)
        else:
            primary_idx = nearest_threshold_idx + 1
            secondary_idx = min(len(self.categories) - 1, nearest_threshold_idx + 2)

        prob_primary = 0.5 + 0.5 * min_distance / (sigma + 1e-10)
        prob_primary = np.clip(prob_primary, 0.5, 1.0)

        probabilities[self.categories[primary_idx]] = prob_primary
        probabilities[self.categories[secondary_idx]] = 1.0 - prob_primary

        return probabilities

    def get_category_info(self, category, turbidity_source=None):
        """Get source-aware category details."""
        info_base = {
            "High Turbidity": {
                "description": "Water is very murky, disk barely or not visible",
                "visibility": "Poor",
                "secchi_depth_equivalent": "< 0.5 meters",
                "water_quality": "Poor",
            },
            "Moderately Turbid": {
                "description": "Water has noticeable cloudiness, disk visible but unclear",
                "visibility": "Fair",
                "secchi_depth_equivalent": "0.5 - 1.5 meters",
                "water_quality": "Fair",
            },
            "Slightly Turbid": {
                "description": "Water is mostly clear with slight haze",
                "visibility": "Good",
                "secchi_depth_equivalent": "1.5 - 4.0 meters",
                "water_quality": "Good",
            },
            "Clear Water": {
                "description": "Water is very clear, disk sharply visible",
                "visibility": "Excellent",
                "secchi_depth_equivalent": "> 4.0 meters",
                "water_quality": "Excellent",
            },
        }

        info = info_base.get(category, {}).copy()

        if turbidity_source and turbidity_source.get("primary_source"):
            source = turbidity_source["primary_source"]

            if source == "algal":
                info["typical_causes"] = "Algal bloom, high nutrients, eutrophication"
                info["ntu_range"] = self._get_ntu_range_algal(category)
                info["trophic_state"] = self._get_trophic_state(category)
                info["management_action"] = self._get_management_algal(category)
                info["ecological_impact"] = self._get_ecological_algal(category)
            elif source == "sediment":
                info["typical_causes"] = "Sediment, silt, clay, erosion, storm runoff"
                info["ntu_range"] = self._get_ntu_range_sediment(category)
                info["trophic_state"] = "N/A (sediment-based, not biological)"
                info["management_action"] = self._get_management_sediment(category)
                info["ecological_impact"] = self._get_ecological_sediment(category)
            else:
                info["typical_causes"] = "Mixed sources: algae, sediment, organic matter"
                info["ntu_range"] = self._get_ntu_range_mixed(category)
                info["trophic_state"] = self._get_trophic_state(category) + " (if algal component)"
                info["management_action"] = self._get_management_mixed(category)
                info["ecological_impact"] = self._get_ecological_mixed(category)

        return info

    def _get_ntu_range_algal(self, category):
        ranges = {
            "High Turbidity": "> 100 NTU",
            "Moderately Turbid": "25 - 100 NTU",
            "Slightly Turbid": "5 - 25 NTU",
            "Clear Water": "< 5 NTU",
        }
        return ranges.get(category, "N/A")

    def _get_ntu_range_sediment(self, category):
        ranges = {
            "High Turbidity": "> 150 NTU",
            "Moderately Turbid": "40 - 150 NTU",
            "Slightly Turbid": "10 - 40 NTU",
            "Clear Water": "< 10 NTU",
        }
        return ranges.get(category, "N/A")

    def _get_ntu_range_mixed(self, category):
        ranges = {
            "High Turbidity": "> 120 NTU",
            "Moderately Turbid": "30 - 120 NTU",
            "Slightly Turbid": "7 - 30 NTU",
            "Clear Water": "< 7 NTU",
        }
        return ranges.get(category, "N/A")

    def _get_trophic_state(self, category):
        states = {
            "High Turbidity": "Hypereutrophic (Carlson TSI > 60)",
            "Moderately Turbid": "Eutrophic (Carlson TSI 50-60)",
            "Slightly Turbid": "Mesotrophic (Carlson TSI 40-50)",
            "Clear Water": "Oligotrophic (Carlson TSI < 40)",
        }
        return states.get(category, "N/A")

    def _get_management_algal(self, category):
        actions = {
            "High Turbidity": "Reduce nutrient inputs (P, N), investigate pollution sources, consider algaecide if HAB",
            "Moderately Turbid": "Monitor nutrient levels, implement BMPs, reduce agricultural runoff",
            "Slightly Turbid": "Maintain current nutrient management, monitor trends",
            "Clear Water": "Preserve low-nutrient conditions, prevent development impacts",
        }
        return actions.get(category, "Monitor water quality")

    def _get_management_sediment(self, category):
        actions = {
            "High Turbidity": "Implement erosion control, stabilize stream banks, reduce construction impacts",
            "Moderately Turbid": "Monitor erosion sources, install sediment traps, vegetate bare soil",
            "Slightly Turbid": "Maintain riparian buffers, monitor during storm events",
            "Clear Water": "Preserve watershed vegetation, prevent soil disturbance",
        }
        return actions.get(category, "Monitor water quality")

    def _get_management_mixed(self, category):
        actions = {
            "High Turbidity": "Address both nutrient and sediment sources, watershed assessment needed",
            "Moderately Turbid": "Monitor nutrients and sediment, implement combined BMPs",
            "Slightly Turbid": "Maintain current controls, periodic water quality testing",
            "Clear Water": "Preserve existing conditions, protect watershed",
        }
        return actions.get(category, "Monitor water quality")

    def _get_ecological_algal(self, category):
        impacts = {
            "High Turbidity": "Severe light limitation, potential HABs, oxygen depletion risk",
            "Moderately Turbid": "Moderate productivity, some light limitation, fish habitat degraded",
            "Slightly Turbid": "Balanced productivity, adequate light penetration",
            "Clear Water": "Low productivity, high light penetration, pristine conditions",
        }
        return impacts.get(category, "N/A")

    def _get_ecological_sediment(self, category):
        impacts = {
            "High Turbidity": "Severe light limitation, sediment smothering, gill damage to fish",
            "Moderately Turbid": "Reduced visibility, habitat degradation, spawning impacts",
            "Slightly Turbid": "Minor impacts, temporary after storm events",
            "Clear Water": "Minimal sediment stress, good aquatic habitat",
        }
        return impacts.get(category, "N/A")

    def _get_ecological_mixed(self, category):
        impacts = {
            "High Turbidity": "Combined light limitation and sediment stress, degraded habitat",
            "Moderately Turbid": "Moderate impacts from both sources, ecosystem stress",
            "Slightly Turbid": "Minor combined effects, generally acceptable",
            "Clear Water": "Minimal stress, healthy aquatic ecosystem",
        }
        return impacts.get(category, "N/A")

    def get_equivalent_metrics(self, visibility_score, turbidity_source=None):
        """Estimate equivalent water quality metrics with source-aware logic."""
        if visibility_score > 0.85:
            secchi_depth = 5.0 + (visibility_score - 0.85) * 20.0
        elif visibility_score > 0.70:
            secchi_depth = 3.0 + (visibility_score - 0.70) * 13.33
        elif visibility_score > 0.50:
            secchi_depth = 1.5 + (visibility_score - 0.50) * 7.5
        elif visibility_score > 0.25:
            secchi_depth = 0.5 + (visibility_score - 0.25) * 4.0
        else:
            secchi_depth = 0.1 + visibility_score * 1.6

        if turbidity_source and turbidity_source.get("primary_source"):
            source = turbidity_source["primary_source"]

            if source == "algal":
                k_factor = 1.7
                ntu_estimate = k_factor / max(secchi_depth, 0.1)
            elif source == "sediment":
                k_factor = 2.5
                ntu_estimate = k_factor / max(secchi_depth, 0.1)
            else:
                k_factor = 2.0
                ntu_estimate = k_factor / max(secchi_depth, 0.1)
        else:
            k_factor = 2.0
            ntu_estimate = k_factor / max(secchi_depth, 0.1)

        if visibility_score < 0.3:
            ntu_estimate *= 1.5 - visibility_score

        if turbidity_source and turbidity_source.get("primary_source") == "algal":
            tsi_estimate = 60 - 14.41 * math.log(max(secchi_depth, 0.1))
            tsi_applicable = True
        else:
            tsi_estimate = None
            tsi_applicable = False

        confidence = self.get_confidence_numeric(visibility_score)
        uncertainty_factor = 1.0 - confidence

        secchi_uncertainty = secchi_depth * 0.2 * (1 + uncertainty_factor)
        ntu_uncertainty = ntu_estimate * 0.3 * (1 + uncertainty_factor)

        result = {
            "estimated_ntu": round(ntu_estimate, 1),
            "ntu_range": (
                round(max(0, ntu_estimate - ntu_uncertainty), 1),
                round(ntu_estimate + ntu_uncertainty, 1),
            ),
            "estimated_secchi_depth_m": round(secchi_depth, 2),
            "secchi_depth_range": (
                round(max(0.1, secchi_depth - secchi_uncertainty), 2),
                round(secchi_depth + secchi_uncertainty, 2),
            ),
            "confidence": confidence,
        }

        if tsi_applicable and tsi_estimate is not None:
            tsi_uncertainty = 5.0 * (1 + uncertainty_factor)
            result["estimated_carlson_tsi"] = round(tsi_estimate, 1)
            result["tsi_range"] = (
                round(tsi_estimate - tsi_uncertainty, 1),
                round(min(100, tsi_estimate + tsi_uncertainty), 1),
            )
            result["tsi_applicable"] = True
        else:
            result["estimated_carlson_tsi"] = None
            result["tsi_range"] = None
            result["tsi_applicable"] = False
            result["tsi_note"] = (
                "TSI not applicable - turbidity is sediment-based, not algal"
            )

        return result

    def print_standard_info(self):
        """Print information about all available standards."""
        print("\n" + "=" * 70)
        print("AVAILABLE TURBIDITY CLASSIFICATION STANDARDS")
        print("=" * 70)

        for std_name, std_info in self.standards.items():
            print(f"\n{std_name.upper()}:")
            print(f"  Reference: {std_info['reference']}")
            print(f"  Thresholds: {std_info['thresholds']}")
            print(f"  Note: {std_info['note']}")
            print(f"  Best for: {std_info['best_for']} turbidity")
            print(f"  Application: {std_info['application']}")

        print("\n" + "=" * 70)
        print(
            "CURRENT MODE: "
            + ("AUTO (source-aware)" if self.auto_mode else self.standard.upper())
        )
        print("=" * 70 + "\n")
