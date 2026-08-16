"""
Integration Tests for Unified Scenario Analysis Pipeline.

Verifies end-to-end flow connecting NLP policy stance scoring (with negation and weights),
interest rate transmission, and household/MSME resilience calculations.
"""

from analysis.scenario_analysis import ScenarioEngine


def test_integrated_scenario_pipeline_hawkish():
    engine = ScenarioEngine()
    statement = "Inflation risk is high and price stability requires a rate hike and withdrawal of accommodation."

    results = engine.run_scenario(statement_text=statement, repo_rate_change_bps=50.0)

    # Scenario Summary Verification
    assert "scenario_summary" in results
    assert results["scenario_summary"]["policy_stance_label"] == "Hawkish"
    assert results["scenario_summary"]["policy_stance_score"] > 50.0

    # Policy Analysis Payload Verification (Weighted & Signal Fields)
    policy = results["policy_analysis"]
    assert "weighted_hawkish_score" in policy
    assert "weighted_dovish_score" in policy
    assert policy["weighted_hawkish_score"] > 0.0
    assert "rate hike" in policy["hawkish_signals_found"]

    # Profile Impact Verification
    assert "priya_homeowner" in results["profile_impacts"]
    assert results["profile_impacts"]["priya_homeowner"]["monthly_emi_change"] > 0.0


def test_integrated_scenario_pipeline_negated_dovish():
    engine = ScenarioEngine()
    statement = "There is no inflation risk and no rate hike planned. We observe growth slowdown."

    results = engine.run_scenario(statement_text=statement, repo_rate_change_bps=-25.0)

    # Verify Negation Propagation to Scenario Summary
    assert results["scenario_summary"]["policy_stance_label"] == "Dovish"
    assert results["scenario_summary"]["policy_stance_score"] < 50.0

    # Verify Flipped Signals in Policy Output
    policy = results["policy_analysis"]
    assert policy["weighted_dovish_score"] > 0.0
    assert "NOT inflation risk (flipped dovish)" in policy["dovish_signals_found"]

    # Verify Negative EMI Delta for Rate Cuts
    assert results["profile_impacts"]["priya_homeowner"]["monthly_emi_change"] < 0.0
