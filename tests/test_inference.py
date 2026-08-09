from pathlib import Path

import pytest

from ml.predictor import ModelPredictor


def test_inference_output_shape_and_range(trained_model: Path):
    result = ModelPredictor(trained_model).predict(
        "selenium playwright automation api testing regression"
    )
    assert result["predicted_role"] in {
        "Data Scientist",
        "Data Engineer",
        "Backend Developer",
        "QA Engineer",
        "Frontend Developer",
        "DevOps Engineer",
    }
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["role_ranking"]) == 6
    total = sum(item["probability"] for item in result["role_ranking"])
    assert abs(total - 1.0) < 1e-6


def test_directional_behavior_for_qa_keywords(trained_model: Path):
    result = ModelPredictor(trained_model).predict(
        "selenium playwright testing automation regression"
    )
    assert result["predicted_role"] == "QA Engineer"


@pytest.mark.parametrize(
    ("text", "expected_role"),
    [
        ("docker kubernetes terraform aws cloud automation", "DevOps Engineer"),
        ("react angular frontend typescript css accessibility", "Frontend Developer"),
    ],
)
def test_directional_behavior_for_multiple_roles(
    trained_model: Path,
    text: str,
    expected_role: str,
):
    result = ModelPredictor(trained_model).predict(text)
    assert result["predicted_role"] == expected_role


def test_inference_is_invariant_to_case_and_spacing(trained_model: Path):
    predictor = ModelPredictor(trained_model)
    baseline = predictor.predict("python machine learning statistics nlp")
    variant = predictor.predict("  PYTHON   Machine Learning  STATISTICS   NLP ")
    assert baseline["predicted_role"] == variant["predicted_role"]
    assert baseline["role_ranking"] == variant["role_ranking"]


def test_inference_rejects_empty_input(trained_model: Path):
    with pytest.raises(ValueError, match="must not be empty"):
        ModelPredictor(trained_model).predict("   ")
