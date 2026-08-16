"""
Monetary Policy Transmission Engine Implementation.

Models transmission lags and pass-through elasticities across:
Repo Rate -> WACR -> T-Bills -> G-Secs -> MCLR -> Deposit Rates
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class TransmissionModel:
    """
    Monetary Policy Transmission Model using Auto-Regressive Distributed Lag (ARDL) dynamics.

    For each target rate Y_t and anchor policy rate X_t (repo rate):
    Y_t = alpha + beta_0 * X_t + beta_1 * X_{t-1} + phi * Y_{t-1} + epsilon_t

    Calculates:
    - Short-Run Elasticity (Immediate Pass-Through): beta_0
    - Long-Run Pass-Through Elasticity: (beta_0 + beta_1) / (1 - phi)
    - Half-Life Lag (Months): log(0.5) / log(phi)
    """

    TARGET_COLUMNS = ["wacr", "tbill_91d", "gsec_10y", "mclr_1y", "deposit_rate_1y"]

    def __init__(self, anchor_col: str = "repo_rate") -> None:
        self.anchor_col = anchor_col
        self.elasticities: Dict[str, Dict[str, float]] = {}
        self.fitted: bool = False

    def fit(self, df: pd.DataFrame, max_lag: int = 2) -> "TransmissionModel":
        """
        Fits ARDL regressions for all target rates relative to the policy anchor rate.

        Parameters
        ----------
        df : pd.DataFrame
            Historical rate data containing anchor_col and TARGET_COLUMNS.
        max_lag : int, default=2
            Maximum lag order for policy and target series in months.

        Returns
        -------
        TransmissionModel
            The fitted instance of TransmissionModel.
        """
        if self.anchor_col not in df.columns:
            raise KeyError(f"Anchor column '{self.anchor_col}' missing from input DataFrame.")

        X_repo = df[self.anchor_col].values

        for target in self.TARGET_COLUMNS:
            if target not in df.columns:
                continue

            Y = df[target].values
            
            # Construct lagged features: [1, X_t, X_{t-1}, Y_{t-1}]
            Y_t = Y[max_lag:]
            X_t = X_repo[max_lag:]
            X_lag1 = X_repo[max_lag - 1 : -1]
            Y_lag1 = Y[max_lag - 1 : -1]

            # Matrix design matrix
            X_matrix = np.column_stack([np.ones_like(Y_t), X_t, X_lag1, Y_lag1])

            # OLS Estimation
            coeffs, _, _, _ = np.linalg.lstsq(X_matrix, Y_t, rcond=None)
            alpha, beta_0, beta_1, phi = coeffs

            # Ensure stability constraint on AR component
            phi = float(np.clip(phi, 0.001, 0.95))

            # Long-run cumulative elasticity
            long_run_elasticity = float((beta_0 + beta_1) / (1.0 - phi))
            long_run_elasticity = float(np.clip(long_run_elasticity, 0.05, 1.5))

            # Half-life of transmission in months
            half_life_months = float(np.log(0.5) / np.log(phi)) if phi > 0 else 0.0

            # Calculate R-squared
            y_pred = X_matrix @ coeffs
            ss_res = np.sum((Y_t - y_pred) ** 2)
            ss_tot = np.sum((Y_t - np.mean(Y_t)) ** 2)
            r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            self.elasticities[target] = {
                "short_run_elasticity": float(beta_0),
                "long_run_elasticity": long_run_elasticity,
                "ar_coefficient": phi,
                "half_life_months": half_life_months,
                "r_squared": max(0.0, r_squared),
            }

        self.fitted = True
        return self

    def propagate(
        self,
        repo_shock_decimal: float,
        current_rates: Optional[Dict[str, float]] = None,
        horizon: str = "long_run",
    ) -> Dict[str, float]:
        """
        Propagates a policy rate shock through the financial transmission chain.

        Parameters
        ----------
        repo_shock_decimal : float
            Interest rate change in decimal (e.g., +0.0100 for +100 bps, -0.0050 for -50 bps).
        current_rates : Dict[str, float], optional
            Base rates to apply transmission shocks to. If None, default baseline rates are used.
        horizon : str, default="long_run"
            Either 'short_run' (immediate 1-month impact) or 'long_run' (full equilibrium impact).

        Returns
        -------
        Dict[str, float]
            New post-shock interest rates for all target instruments.
        """
        if not self.fitted:
            # Fallback theoretical Indian transmission elasticity parameters if unfitted
            self._apply_fallback_parameters()

        if current_rates is None:
            current_rates = {
                "repo_rate": 0.065,
                "wacr": 0.065,
                "tbill_91d": 0.067,
                "gsec_10y": 0.071,
                "mclr_1y": 0.085,
                "deposit_rate_1y": 0.068,
            }

        new_rates = current_rates.copy()
        new_rates["repo_rate"] = current_rates.get("repo_rate", 0.065) + repo_shock_decimal

        elasticity_key = "short_run_elasticity" if horizon == "short_run" else "long_run_elasticity"

        for target, stats in self.elasticities.items():
            elasticity = stats[elasticity_key]
            current_val = current_rates.get(target, 0.065)
            # Propagate rate change
            target_change = repo_shock_decimal * elasticity
            new_rates[target] = max(0.0, current_val + target_change)

        return new_rates

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Returns full transmission diagnostics, elasticities, and lags.
        """
        if not self.fitted:
            self._apply_fallback_parameters()
        return self.elasticities

    def _apply_fallback_parameters(self) -> None:
        """
        Stylized empirical RBI transmission elasticities based on Indian market research.
        """
        self.elasticities = {
            "wacr": {
                "short_run_elasticity": 0.92,
                "long_run_elasticity": 1.00,
                "ar_coefficient": 0.15,
                "half_life_months": 0.3,
                "r_squared": 0.95,
            },
            "tbill_91d": {
                "short_run_elasticity": 0.78,
                "long_run_elasticity": 0.95,
                "ar_coefficient": 0.35,
                "half_life_months": 0.6,
                "r_squared": 0.88,
            },
            "gsec_10y": {
                "short_run_elasticity": 0.35,
                "long_run_elasticity": 0.60,
                "ar_coefficient": 0.65,
                "half_life_months": 1.6,
                "r_squared": 0.72,
            },
            "mclr_1y": {
                "short_run_elasticity": 0.28,
                "long_run_elasticity": 0.75,
                "ar_coefficient": 0.72,
                "half_life_months": 2.1,
                "r_squared": 0.85,
            },
            "deposit_rate_1y": {
                "short_run_elasticity": 0.22,
                "long_run_elasticity": 0.68,
                "ar_coefficient": 0.78,
                "half_life_months": 2.8,
                "r_squared": 0.81,
            },
        }
        self.fitted = True


if __name__ == "__main__":
    try:
        df = pd.read_csv("data/processed/rbi_rates_monthly.csv")
        model = TransmissionModel()
        model.fit(df)

        print("=== Monetary Transmission Elasticity Summary ===")
        summary_df = pd.DataFrame(model.get_summary()).T
        print(summary_df[["short_run_elasticity", "long_run_elasticity", "half_life_months", "r_squared"]])

        print("\n=== Scenario Simulation: RBI +100 bps Rate Hike (+1.00%) ===")
        shock_100bps = 0.0100
        baseline = {
            "repo_rate": 0.065,
            "wacr": 0.065,
            "tbill_91d": 0.067,
            "gsec_10y": 0.071,
            "mclr_1y": 0.085,
            "deposit_rate_1y": 0.068,
        }
        post_shock = model.propagate(shock_100bps, baseline, horizon="long_run")

        for k in baseline:
            initial = baseline[k] * 100
            updated = post_shock[k] * 100
            diff = updated - initial
            print(f"{k:17s}: {initial:.2f}% -> {updated:.2f}% ({diff:+.2f}% / {diff*100:+.0f} bps)")

    except FileNotFoundError:
        print("Processed dataset missing. Run 'python data/ingest_rbi_data.py' first.")
