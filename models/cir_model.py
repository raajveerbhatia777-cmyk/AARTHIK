"""CIR (Cox–Ingersoll–Ross) model implementation.

Provides a production-grade estimator and simulator for the CIR process:

    dr_t = kappa*(theta - r_t) dt + sigma * sqrt(r_t) dW_t

Estimation is performed by discretizing the SDE and fitting an OLS regression
on the Euler-Maruyama discretization. Volatility is estimated from residuals
using the conditional variance structure Var(epsilon_t | r_t) = sigma^2 r_t dt.

The simulator uses an Euler-Maruyama step with full truncation at zero to
ensure non-negative rates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import log
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class CIRModel:
    """Cox–Ingersoll–Ross model estimator and simulator.

    Attributes
    ----------
    kappa: Optional[float]
        Speed of mean reversion (per year).
    theta: Optional[float]
        Long-run mean level of the rate (annualized units).
    sigma: Optional[float]
        Volatility parameter (annualized).
    r0: Optional[float]
        Initial rate used for simulation when not provided.
    eps: float
        Small epsilon used to stabilize divisions.
    """

    kappa: Optional[float] = None
    theta: Optional[float] = None
    sigma: Optional[float] = None
    r0: Optional[float] = None
    eps: float = field(default=1e-8)

    def fit(self, rates: np.ndarray, dt: float = 1 / 12) -> None:
        """Calibrate kappa, theta and sigma from historical rates.

        Parameters
        ----------
        rates
            1-D array of historical short rates in consistent time units
            (e.g., monthly rates expressed as decimals if dt=1/12).
        dt
            Time step between observations in years (default monthly = 1/12).

        Notes
        -----
        The discretized Euler-Maruyama reads
            r_{t+1} = r_t + kappa*(theta - r_t)*dt + sigma*sqrt(r_t)*sqrt(dt)*z_t
        which can be rearranged to an approximate AR(1):
            r_{t+1} = a + b * r_t + residual
        with a = kappa*theta*dt and b = 1 - kappa*dt.

        We fit a and b by OLS, recover kappa and theta, and estimate sigma
        from the conditional variance of residuals: Var(residual | r_t) = sigma^2 r_t dt.
        """
        rates = np.asarray(rates, dtype=float)
        if rates.ndim != 1 or rates.size < 3:
            raise ValueError("rates must be a 1-D array with at least 3 observations")

        y = rates[1:]
        x = rates[:-1]

        # Build design matrix [1, x]
        X = np.column_stack((np.ones_like(x), x))

        # OLS using least squares (stable for small problems)
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        a_hat, b_hat = float(coeffs[0]), float(coeffs[1])

        # Recover kappa and theta from discretization
        kappa_hat = (1.0 - b_hat) / dt
        # Protect against very small/negative kappa estimates
        if kappa_hat <= 0:
            logger.warning("Estimated non-positive kappa (%.6g). Forcing to small positive value.", kappa_hat)
            kappa_hat = max(kappa_hat, self.eps)

        theta_hat = a_hat / (kappa_hat * dt + self.eps)
        if theta_hat <= 0:
            logger.warning("Estimated non-positive theta (%.6g). Forcing to small positive value.", theta_hat)
            theta_hat = max(theta_hat, self.eps)

        # Residuals and sigma estimation
        residuals = y - (a_hat + b_hat * x)
        # Avoid division by zero when r_t is tiny; add eps
        denom = np.maximum(x * dt, self.eps)
        sigma2_hat = np.mean(residuals ** 2 / denom)
        sigma_hat = float(np.sqrt(max(0.0, sigma2_hat)))

        # Ensure positivity of sigma
        if sigma_hat <= 0:
            logger.warning("Estimated non-positive sigma (%.6g). Forcing to small positive value.", sigma_hat)
            sigma_hat = max(sigma_hat, self.eps)

        # Assign estimates
        self.kappa = float(kappa_hat)
        self.theta = float(theta_hat)
        self.sigma = float(sigma_hat)

        # Set r0 if not present
        if self.r0 is None:
            self.r0 = float(rates[-1])

        # Check Feller condition and log
        if not self._feller_condition():
            logger.warning(
                "Feller condition violated: 2*kappa*theta = %.6g <= sigma^2 = %.6g",
                2 * self.kappa * self.theta,
                self.sigma ** 2,
            )
        else:
            logger.info("Feller condition satisfied: 2*kappa*theta = %.6g > sigma^2 = %.6g",
                        2 * self.kappa * self.theta, self.sigma ** 2)

    def _feller_condition(self) -> bool:
        """Return True if Feller condition holds (2*kappa*theta > sigma^2)."""
        if self.kappa is None or self.theta is None or self.sigma is None:
            raise ValueError("Model parameters are not all set. Run fit() first.")
        return 2.0 * self.kappa * self.theta > (self.sigma ** 2)

    def simulate_paths(
        self,
        n_paths: int = 100,
        n_steps: int = 12,
        dt: float = 1 / 12,
        r0: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate Monte Carlo paths using Euler-Maruyama with full truncation.

        Parameters
        ----------
        n_paths
            Number of Monte Carlo paths to simulate.
        n_steps
            Number of time steps to simulate (horizon in steps).
        dt
            Time increment per step (in years).
        r0
            Initial rate. If None, uses fitted self.r0 or raises if missing.
        random_state
            Optional integer seed for reproducible simulations.

        Returns
        -------
        np.ndarray
            Simulated rates with shape (n_paths, n_steps + 1).
        """
        if self.kappa is None or self.theta is None or self.sigma is None:
            raise ValueError("Model parameters are not set. Call fit() before simulate_paths().")

        if r0 is None:
            if self.r0 is None:
                raise ValueError("Initial rate r0 must be provided (either as argument or by fit()).")
            r0 = float(self.r0)

        rng = np.random.default_rng(random_state)
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = max(0.0, float(r0))

        sqrt_dt = np.sqrt(dt)
        kappa, theta, sigma = self.kappa, self.theta, self.sigma

        for t in range(n_steps):
            r_t = paths[:, t]
            # full truncation: use sqrt(max(r_t, 0)) for diffusion term
            sqrt_rt = np.sqrt(np.maximum(r_t, 0.0))
            dW = rng.standard_normal(size=n_paths) * sqrt_dt
            dr_det = kappa * (theta - r_t) * dt
            dr_stoch = sigma * sqrt_rt * dW
            r_next = r_t + dr_det + dr_stoch
            # reflect/absorption at zero: ensure non-negative
            r_next = np.maximum(r_next, 0.0)
            paths[:, t + 1] = r_next

        return paths

    def get_summary(self) -> Dict[str, float]:
        """Return a dictionary summarizing fitted parameters and diagnostics."""
        if self.kappa is None or self.theta is None or self.sigma is None:
            raise ValueError("Model is not fitted yet")

        half_life = float(log(2.0) / self.kappa) if self.kappa > 0 else float('inf')
        feller = bool(self._feller_condition())

        return {
            "kappa": float(self.kappa),
            "theta": float(self.theta),
            "sigma": float(self.sigma),
            "half_life_years": half_life,
            "feller_satisfied": feller,
        }


def _load_processed_rates(path: str = "data/processed/rbi_rates_monthly.csv") -> np.ndarray:
    """Load processed monthly RBI rates from CSV. Expects a column 'repo_rate'.

    Returns a 1-D ndarray of rates.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        logger.error("Processed rates file not found at %s", path)
        raise

    if "repo_rate" not in df.columns:
        raise ValueError("CSV must contain a 'repo_rate' column")

    return df["repo_rate"].astype(float).to_numpy()


if __name__ == "__main__":
    # Driver for quick experimentation
    try:
        rates = _load_processed_rates()
    except Exception as exc:  # pragma: no cover - manual execution path
        logger.error("Unable to load rates: %s", exc)
        raise SystemExit(1)

    model = CIRModel()
    model.fit(rates, dt=1 / 12)
    summary = model.get_summary()
    logger.info("Fitted model summary: %s", summary)

    paths = model.simulate_paths(n_paths=50, n_steps=12, dt=1 / 12, random_state=42)
    logger.info("Simulated %d paths for %d steps (showing first path):", paths.shape[0], paths.shape[1] - 1)
    logger.info(paths[0])
