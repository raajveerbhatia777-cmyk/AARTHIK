"""
Financial Resilience Engine Implementation.

Evaluates household/business balance sheet sensitivity to interest rate shocks
and calculates a 0-100 Financial Resilience Index score.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FinancialProfile:
    name: str
    category: str
    monthly_income: float
    liquid_savings: float
    floating_debt_principal: float
    fixed_income_assets: float
    debt_tenure_years: float = 15.0
    current_borrowing_rate: float = 0.085
    current_deposit_rate: float = 0.068
    essential_monthly_expenses: float = 0.0


class ResilienceEngine:
    """
    Simulates rate shocks on household and MSME economic profiles
    and calculates a composite 0-100 Resilience Score.
    """

    def __init__(self) -> None:
        self.default_profiles = self._initialize_default_profiles()

    def _initialize_default_profiles(self) -> Dict[str, FinancialProfile]:
        return {
            "aarav_student": FinancialProfile(
                name="Aarav (Young Professional / Student)",
                category="Student",
                monthly_income=45000.0,
                liquid_savings=60000.0,
                floating_debt_principal=400000.0,  # Education Loan
                fixed_income_assets=25000.0,
                debt_tenure_years=7.0,
                current_borrowing_rate=0.095,
                current_deposit_rate=0.065,
                essential_monthly_expenses=25000.0,
            ),
            "priya_homeowner": FinancialProfile(
                name="Priya (First-Time Homeowner)",
                category="Homeowner",
                monthly_income=120000.0,
                liquid_savings=250000.0,
                floating_debt_principal=5000000.0,  # Housing Loan
                fixed_income_assets=300000.0,
                debt_tenure_years=20.0,
                current_borrowing_rate=0.085,
                current_deposit_rate=0.068,
                essential_monthly_expenses=55000.0,
            ),
            "rajesh_msme": FinancialProfile(
                name="Rajesh (Small Business Owner)",
                category="MSME",
                monthly_income=250000.0,  # Business Revenue / Income
                liquid_savings=400000.0,
                floating_debt_principal=3000000.0,  # Working Capital Loan
                fixed_income_assets=200000.0,
                debt_tenure_years=5.0,
                current_borrowing_rate=0.110,
                current_deposit_rate=0.065,
                essential_monthly_expenses=140000.0,
            ),
            "sunita_saver": FinancialProfile(
                name="Sunita (Senior Citizen / Saver)",
                category="Saver",
                monthly_income=35000.0,  # Pension
                liquid_savings=1500000.0,
                floating_debt_principal=0.0,
                fixed_income_assets=2500000.0,  # Senior Citizen Term Deposits
                debt_tenure_years=0.0,
                current_borrowing_rate=0.0,
                current_deposit_rate=0.075,
                essential_monthly_expenses=22000.0,
            ),
        }

    @staticmethod
    def calculate_emi(principal: float, annual_rate: float, tenure_years: float) -> float:
        """Calculates monthly Equated Monthly Installment (EMI)."""
        if principal <= 0 or tenure_years <= 0 or annual_rate <= 0:
            return 0.0
        r = annual_rate / 12.0
        n = tenure_years * 12.0
        emi = principal * r * ((1.0 + r) ** n) / (((1.0 + r) ** n) - 1.0)
        return float(emi)

    def evaluate_profile(
        self,
        profile: FinancialProfile,
        borrowing_rate_shock_decimal: float = 0.0,
        deposit_rate_shock_decimal: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculates financial impact metrics and 0-100 Resilience Score under rate shocks.
        """
        new_borrowing_rate = max(0.001, profile.current_borrowing_rate + borrowing_rate_shock_decimal)
        new_deposit_rate = max(0.001, profile.current_deposit_rate + deposit_rate_shock_decimal)

        # Baseline vs Post-Shock Monthly EMI
        base_emi = self.calculate_emi(
            profile.floating_debt_principal, profile.current_borrowing_rate, profile.debt_tenure_years
        )
        new_emi = self.calculate_emi(
            profile.floating_debt_principal, new_borrowing_rate, profile.debt_tenure_years
        )
        monthly_emi_change = new_emi - base_emi

        # Baseline vs Post-Shock Monthly Deposit Interest Income
        base_interest_inc = (profile.fixed_income_assets * profile.current_deposit_rate) / 12.0
        new_interest_inc = (profile.fixed_income_assets * new_deposit_rate) / 12.0
        monthly_interest_change = new_interest_inc - base_interest_inc

        # Net Disposable Cash Flow Impact
        net_monthly_cashflow_change = monthly_interest_change - monthly_emi_change
        post_shock_disposable_income = (
            profile.monthly_income + new_interest_inc - profile.essential_monthly_expenses - new_emi
        )

        # ------------------ Resilience Index Scoring (0-100 Scale) ------------------
        # 1. Debt Burden Score (Lower DTI ratio -> Higher Score)
        dti_ratio = (new_emi / profile.monthly_income) if profile.monthly_income > 0 else 0.0
        debt_score = max(0.0, min(100.0, (1.0 - dti_ratio / 0.60) * 100.0))

        # 2. Liquidity Runway Score (Months of expenses covered by savings)
        monthly_burn = profile.essential_monthly_expenses + new_emi
        runway_months = profile.liquid_savings / monthly_burn if monthly_burn > 0 else 12.0
        liquidity_score = max(0.0, min(100.0, (runway_months / 6.0) * 100.0))  # 6 months = 100/100

        # 3. Interest Rate Sensitivity Score
        rate_sensitivity = abs(net_monthly_cashflow_change) / profile.monthly_income if profile.monthly_income > 0 else 0.0
        sensitivity_score = max(0.0, min(100.0, (1.0 - rate_sensitivity / 0.10) * 100.0))

        # Overall Weighted Resilience Score
        resilience_index = round(
            0.45 * debt_score + 0.35 * liquidity_score + 0.20 * sensitivity_score, 1
        )

        return {
            "base_emi": round(base_emi, 2),
            "new_emi": round(new_emi, 2),
            "monthly_emi_change": round(monthly_emi_change, 2),
            "monthly_interest_change": round(monthly_interest_change, 2),
            "net_monthly_cashflow_change": round(net_monthly_cashflow_change, 2),
            "post_shock_disposable_income": round(post_shock_disposable_income, 2),
            "debt_score": round(debt_score, 1),
            "liquidity_score": round(liquidity_score, 1),
            "resilience_index": max(0.0, min(100.0, resilience_index)),
        }


if __name__ == "__main__":
    engine = ResilienceEngine()
    print("=== Financial Resilience Shock Engine (+100 bps Transmission Shock) ===")
    
    # Simulate +100 bps Repo Hike transmission: +75 bps to Borrowing, +68 bps to Deposits
    borrowing_shock = 0.0075
    deposit_shock = 0.0068

    for key, prof in engine.default_profiles.items():
        res = engine.evaluate_profile(prof, borrowing_shock, deposit_shock)
        print(f"\nProfile: {prof.name}")
        print(f"  Monthly EMI: ₹{res['base_emi']:,.2f} -> ₹{res['new_emi']:,.2f} (Δ ₹{res['monthly_emi_change']:+,.2f})")
        print(f"  Net Cash Flow Impact: ₹{res['net_monthly_cashflow_change']:+,.2f}/month")
        print(f"  Resilience Index: {res['resilience_index']}/100")
