from app.services.classifier import TurbidityClassifier


def test_fixed_standard_classification_boundaries():
    classifier = TurbidityClassifier(standard="epa")

    low = classifier.classify(0.10)
    high = classifier.classify(0.90)

    assert low["category"] == "High Turbidity"
    assert high["category"] == "Clear Water"


def test_auto_mode_switches_standard_from_source():
    classifier = TurbidityClassifier(standard="auto")

    result = classifier.classify(
        visibility_score=0.5,
        turbidity_source={"primary_source": "sediment", "confidence": 0.9},
    )

    assert result["standard_used"] == "sediment"


def test_equivalent_metrics_tsi_applicability():
    classifier = TurbidityClassifier(standard="epa")

    algal = classifier.get_equivalent_metrics(0.7, {"primary_source": "algal"})
    sediment = classifier.get_equivalent_metrics(0.7, {"primary_source": "sediment"})

    assert algal["tsi_applicable"] is True
    assert sediment["tsi_applicable"] is False
