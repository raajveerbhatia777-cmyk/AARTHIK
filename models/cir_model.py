"""
Cox-Ingersoll-Ross (CIR) Interest Rate Model Implementation.

SDE: dr_t = kappa * (theta - r_t) * dt + sigma * sqrt(r_t) * dW_t
"""

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


class CIRModel:
    """
    Cox-Ingersoll-Ross (CIR) stochastic interest rate model.

    Parameters
    ----------
    kappa : float, optional
        Speed of mean reversion (must be > 0).
    theta : float, optional
        Long-run equilibrium interest rate (must be > 0).
    sigma : float, optional
        Volatility parameter (must be > 0).
    r0 : float, optional
        Initial interest rate value.
    """

    def __init__(
        self,
        kappa: Optional[float] = None,
        theta: Optional[float] = None,
        sigma: Optional[float] = None,
        r0: Optional[float] = None,
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0

    def _feller_condition(self) -> bool:
        """
        Check if the Feller condition (2 * kappa * theta > sigma^2) holds.
        When satisfied, the process is strictly positive and guaranteed not to hit zero.
        """
        if self.kappa is None or self.theta is None or self.sigma is None:
            return False
        return (2.0 * self.kappa * self.theta) > (self.sigma**2)

    def fit(self, rates: Union[np.ndarray, pd.Series], dt: float = 1.0 / 12.0) -> "CIRModel":
        """
        Calibrate CIR model parameters (kappa, theta, sigma) from historical time-series data
        using discretized weighted ordinary least squares (OLS).

        Parameters
        ----------
        rates : np.ndarray or pd.Series
            Array of historical interest rates in decimal form (e.g., 0.065 for 6.5%).
        dt : float, default=1/12
            Time increment between observations in years (1/12 for monthly).

        Returns
        -------
        CIRModel
            The fitted instance of CIRModel.
        """
        r = np.asarray(rates, dtype=float)
        if len(r) < 3:
            raise ValueError("Rate series must contain at least 3 data points for estimation.")

        self.r0 = float(r[-1])

        # Discretized CIR model regression:
        # (r_{t+1} - r_t) / sqrt(r_t) = alpha / sqrt(r_t) + beta * sqrt(r_t) + error
        # where alpha = kappa * theta * dt, beta = -kappa * dt
        r_t = np.maximum(r[:-1], 1e-8)
        r_next = np.maximum(r[1:], 1e-8)

        sqrt_r_t = np.sqrt(r_t)
        y = (r_next - r_t) / sqrt_r_t
        x1 = 1.0 / sqrt_r_t
        x2 = sqrt_r_t

        X = np.column_stack([x1, x2])
        
        # OLS estimation: beta_hat = (X^T X)^(-1) X^T y
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta = coeffs[0], coeffs[1]

        # Extract kappa and theta
        kappa_est = -beta / dt
        theta_est = alpha / (kappa_est * dt) if kappa_est > 0 else np.mean(r)

        # Estimate residual volatility sigma
        residuals = y - (alpha * x1 + beta * x2)
        res_var = np.var(residuals, ddof=2)
        sigma_est = np.sqrt(max(res_var / dt, 1e-8))

        # Enforce positivity constraints
        self.kappa = max(float(kappa_est), 1e-4)
        self.theta = max(float(theta_est), 1e-4)
        self.sigma = max(float(sigma_est), 1e-4)

        if not self._feller_condition():
            warnings.warn(
                f"Feller condition violated (2*kappa*theta = {2 * self.kappa * self.theta:.6f} "
                f"<= sigma^2 = {self.sigma**2:.6f}). Simulated paths may reach zero.",
                UserWarning,
            )

        return self

    def simulate_paths(
        self,
        n_paths: int = 100,
        n_steps: int = 12,
        dt: float = 1.0 / 12.0,
        r0: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate future interest rate paths using Euler-Maruyama discretization
        with full truncation to enforce non-negativity.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths to generate.
        n_steps : int
            Number of forward time steps.
        dt : float
            Step size in years (1/12 for monthly).
        r0 : float, optional
            Starting rate. Defaults to self.r0 if not provided.
        random_state : int, optional
            Seed for random number generator reproducibility.

        Returns
        -------
        np.ndarray
            Simulated paths array of shape (n_paths, n_steps + 1).
        """
        if self.kappa is None or self.theta is None or self.sigma is None:
            raise ValueError("Model must be fitted or initialized with parameters prior to simulation.")

        start_r0 = float(r0 if r0 is not None else (self.r0 if self.r0 is not None else self.theta))
        rng = np.random.default_rng(random_state)

        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = start_r0
        sqrt_dt = np.sqrt(dt)

        for t in range(n_steps):
            r_curr = np.maximum(paths[:, t], 0.0)
            dW = rng.standard_normal(n_paths) * sqrt_dt
            drift = self.kappa * (self.theta - r_curr) * dt
            diffusion = self.sigma * np.sqrt(r_curr) * dW
            paths[:, t + 1] = np.maximum(0.0, r_curr + drift + diffusion)

        return paths

    def get_summary(self) -> Dict[str, Union[float, bool]]:
        """
        Return summary statistics and diagnostic metrics of the fitted CIR model.
        """
        half_life = np.log(2) / self.kappa if self.kappa and self.kappa > 0 else np.nan
        return {
            "kappa": float(self.kappa) if self.kappa is not None else np.nan,
            "theta": float(self.theta) if self.theta is not None else np.nan,
            "sigma": float(self.sigma) if self.sigma is not None else np.nan,
            "r0": float(self.r0) if self.r0 is not None else np.nan,
            "half_life_years": float(half_life),
            "feller_satisfied": self._feller_condition(),
        }


if __name__ == "__main__":
    try:
        df = pd.read_csv("data/processed/rbi_rates_monthly.csv")
        rates = df["repo_rate"].values

        model = CIRModel()
        model.fit(rates, dt=1.0 / 12.0)

        print("=== CIR Model Calibration Summary ===")
        for key, val in model.get_summary().items():
            print(f"{key}: {val}")

        paths = model.simulate_paths(n_paths=5, n_steps=12, dt=1.0 / 12.0, random_state=42)
        print(f"\nSimulated Paths Shape: {paths.shape}")
        print("First 12-month simulated path (%):", np.round(paths[0] * 100, 2))
    except FileNotFoundError:
        print("Dataset not found. Run 'python data/ingest_rbi_data.py' first.")
