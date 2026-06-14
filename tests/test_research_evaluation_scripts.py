from pathlib import Path

import pandas as pd

from src.evaluate_ablation_study import exact_mcnemar_p_value
from src.evaluate_entailment_layer import normalize_entailment_label
from src.error_analysis_benchmark import build_error_analysis
from src.plot_confidence_calibration import build_calibration_table
from src.source_ranker import blend_relevance_and_trust


def test_exact_mcnemar_p_value_returns_valid_probability():
    stats = exact_mcnemar_p_value([True, True, False, False], [True, False, True, False])
    assert 0.0 <= stats["p_value"] <= 1.0
    assert stats["a_correct_b_wrong"] == 1
    assert stats["a_wrong_b_correct"] == 1


def test_entailment_label_normalization():
    assert normalize_entailment_label("Supports") == "entailment"
    assert normalize_entailment_label("Refuted") == "contradiction"
    assert normalize_entailment_label("NEI") == "neutral"


def test_source_trust_weight_can_be_disabled():
    relevance = 0.40
    trust = 1.00
    assert blend_relevance_and_trust(relevance, trust, trust_weight=0.0) == relevance
    assert blend_relevance_and_trust(relevance, trust, trust_weight=1.0) == trust


def test_calibration_table_from_predictions(tmp_path: Path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"confidence": 0.9, "correct": True},
            {"confidence": 0.8, "correct": False},
            {"confidence": 0.2, "correct": False},
        ]
    ).to_csv(path, index=False)
    table = build_calibration_table(path, output_dir=tmp_path, bins=2)
    assert not table.empty
    assert (tmp_path / "confidence_calibration.csv").exists()


def test_benchmark_error_analysis_output(tmp_path: Path):
    input_path = tmp_path / "benchmark_predictions.csv"
    output_path = tmp_path / "benchmark_error_analysis.csv"
    pd.DataFrame(
        [
            {
                "claim": "Mars is blue.",
                "true_label": "Refuted",
                "predicted_label": "Supported",
                "correct": False,
                "top_source": "Example",
                "top_url": "https://example.com",
                "explanation": "Related source was found.",
            }
        ]
    ).to_csv(input_path, index=False)
    output = build_error_analysis(input_path, output_path)
    assert len(output) == 1
    assert output_path.exists()
    assert "likely_error_type" in output.columns
