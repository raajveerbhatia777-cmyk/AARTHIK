"""
Unified Scenario Analysis Pipeline.

Connects NLP policy stance inputs, interest rate transmission dynamics,
and household/MSME financial resilience calculations into a single scenario run.
"""

from typing import Dict, Any, Optional
from ai.policy_analyser import PolicyAnalyser
from models.resilience_model import ResilienceEngine, FinancialProfile


class ScenarioEngine:
    """
    Executes end-to-end macro-financial stress tests and policy transmission scenarios.
    """

    def __init__(self) -> None:
        self.policy_analyser = PolicyAnalyser()
        self.resilience_engine = ResilienceEngine()

    def run_scenario(
        self,
        statement_text: str,
        repo_rate_change_bps: float,
        borrowing_transmission_factor: float = 0.75,
        deposit_transmission_factor: float = 0.68,
        custom_profiles: Optional[Dict[str, FinancialProfile]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a complete scenario simulation.

        Parameters
        ----------
        statement_text : str
            Text of the RBI MPC policy statement or speech.
        repo_rate_change_bps : float
            Policy rate shift in basis points (e.g., +50.0 or -25.0).
        borrowing_transmission_factor : float, optional
            Pass-through rate to lending products (default 0.75).
        deposit_transmission_factor : float, optional
            Pass-through rate to deposit products (default 0.68).
        custom_profiles : dict, optional
            Optional dict of FinancialProfile objects to override defaults.

        Returns
        -------
        Dict[str, Any]
            Aggregated output containing policy stance analysis and 
            per-profile resilience impact metrics.
        """
        # 1. Analyze Policy Stance via NLP Analyser
        policy_results = self.policy_analyser.analyze_statement(statement_text)

        # 2. Convert Rate Shocks into Decimals
        borrowing_shock = (repo_rate_change_bps / 10000.0) * borrowing_transmission_factor
        deposit_shock = (repo_rate_change_bps / 10000.0) * deposit_transmission_factor

        # 3. Evaluate Resilience Across Profiles
        profiles = custom_profiles or self.resilience_engine.default_profiles
        profile_evaluations = {}

        for profile_key, profile in profiles.items():
            profile_evaluations[profile_key] = self.resilience_engine.evaluate_profile(
                profile,
                borrowing_rate_shock_decimal=borrowing_shock,
                deposit_rate_shock_decimal=deposit_shock,
            )

        return {
            "scenario_summary": {
                "repo_rate_change_bps": repo_rate_change_bps,
                "borrowing_rate_shock_bps": round(borrowing_shock * 10000, 1),
                "deposit_rate_shock_bps": round(deposit_shock * 10000, 1),
                "policy_stance_label": policy_results["stance_label"],
                "policy_stance_score": policy_results["net_stance_score"],
            },
            "policy_analysis": policy_results,
            "profile_impacts": profile_evaluations,
        }


if __name__ == "__main__":
    engine = ScenarioEngine()
    
    sample_statement = """
    The Monetary Policy Committee noted that inflation risk remains elevated due to food price volatility, 
    necessitating sustained focus on price stability. The committee reiterated its stance on withdrawal 
    of accommodation to align inflation with target.
    """

    # Run a +50 bps Repo Hike scenario
    output = engine.run_scenario(
        statement_text=sample_statement,
        repo_rate_change_bps=50.0
    )

    print("=== Integrated Scenario Analysis Output ===")
    print(f"Policy Stance     : {output['scenario_summary']['policy_stance_label']} ({output['scenario_summary']['policy_stance_score']}/100)")
    print(f"Repo Rate Delta   : {output['scenario_summary']['repo_rate_change_bps']:+} bps")
    print("\nProfile Resilience Scores:")
    for key, eval_data in output["profile_impacts"].items():
        print(f"  - {key:22s}: Resilience Index = {eval_data['resilience_index']}/100 | EMI Delta = ₹{eval_data['monthly_emi_change']:+,.2f}")
