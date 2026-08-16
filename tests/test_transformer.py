"""
Unit tests for ai/transformer_analyser.py using mocks to keep tests fast and CI-friendly.
"""

from unittest.mock import MagicMock, patch
import pytest

from ai.transformer_analyser import TransformerPolicyAnalyser


@patch("ai.transformer_analyser.PolicyAnalyser")
def test_transformer_fallback_when_import_fails(mock_lexicon_class):
    """Verifies fallback to PolicyAnalyser when transformers pipeline fails to initialize."""
    mock_lexicon_instance = MagicMock()
    mock_lexicon_instance.analyze_statement.return_value = {
        "stance_label": "Neutral",
        "net_stance_score": 50.0,
    }
    mock_lexicon_class.return_value = mock_lexicon_instance

    with patch("transformers.pipeline", side_effect=ImportError("No transformers package")):
        analyser = TransformerPolicyAnalyser()
        res = analyser.analyze_statement("Inflation remains elevated.")

        assert res["stance_label"] == "Neutral"
        assert res["net_stance_score"] == 50.0
        mock_lexicon_instance.analyze_statement.assert_called_once_with(
            "Inflation remains elevated."
        )


@patch("transformers.pipeline")
def test_transformer_hawkish_inference(mock_pipeline):
    """Verifies Hawkish mapping for negative FinBERT sentiment."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "negative", "score": 0.90}]
    mock_pipeline.return_value = mock_pipe

    analyser = TransformerPolicyAnalyser()
    res = analyser.analyze_statement("Central bank pledges to hike rates aggressively.")

    assert res["stance_label"] == "Hawkish"
    assert res["net_stance_score"] == 95.0
    assert res["confidence_score"] == 0.90
    assert res["model_used"] == "ProsusAI/finbert"


@patch("transformers.pipeline")
def test_transformer_dovish_inference(mock_pipeline):
    """Verifies Dovish mapping for positive FinBERT sentiment."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "positive", "score": 0.80}]
    mock_pipeline.return_value = mock_pipe

    analyser = TransformerPolicyAnalyser()
    res = analyser.analyze_statement("Economic outlook is weakening, rate cuts expected.")

    assert res["stance_label"] == "Dovish"
    assert res["net_stance_score"] == 10.0
    assert res["confidence_score"] == 0.80
