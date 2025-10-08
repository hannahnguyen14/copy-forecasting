from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom

from tsfb.base.models.base_model import BatchMaker
from tsfb.base.models.base_statistic import BaseStatistic


class NegativeBinomialGLM(BaseStatistic):
    """
    Negative Binomial GLM for univariate time series forecasting.

    - Overdispersed counts: Var[Y] = mu + (mu^2)/r
    - In statsmodels, NB uses parameter alpha where Var[Y] = mu + alpha * mu^2,
      so alpha = 1/r.
    """

    def __init__(self, alpha: float = 0.05, r: float = 10.0):
        """
        Parameters
        ----------
        alpha : float
            Significance level for intervals.
        r : float
            Dispersion parameter (>0). Larger r -> closer to Poisson.
        """
        super().__init__(alpha=alpha)
        if r <= 0:
            raise ValueError("r must be > 0 for Negative Binomial.")
        self._r: float = float(r)

    @staticmethod
    def required_hyper_params() -> dict:
        """Expose defaults for hyper-params."""
        return {"alpha": 0.05, "r": 10.0}

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "NegativeBinomialGLM":
        target_col, train_with_t = self._init_train_state(train_data)
        design_matrix, target = self._build_design_and_target(
            train_with_t, target_col, add_intercept=True
        )

        nb_alpha = 1.0 / self._r  # Var = mu + nb_alpha * mu^2
        model = sm.GLM(
            target, design_matrix, family=sm.families.NegativeBinomial(alpha=nb_alpha)
        )
        self._results = model.fit()
        return self

    def _prediction_intervals(
        self, mu_hat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prediction intervals via Negative Binomial quantiles.

        Parameterization for scipy.stats.nbinom:
        - nbinom(r, p) counts #failures before r successes, with success prob p.
        - mean = r*(1-p)/p  => p = r / (r + mu)
        """
        r = self._r
        mu = np.asarray(mu_hat, dtype=float)
        p = r / (r + mu)

        alpha = self._alpha
        pred_lo = nbinom.ppf(alpha / 2.0, r, p)
        pred_hi = nbinom.ppf(1.0 - alpha / 2.0, r, p)

        # Mode for NB(r, p): floor((r - 1) * (1 - p) / p)  if r > 1, else 0
        y_mode = np.where(r > 1.0, np.floor((r - 1.0) * (1.0 - p) / p), 0.0)

        return pred_lo, pred_hi, y_mode

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """Not implemented for this model."""
        raise NotImplementedError("Not implemented batch forecasting!")
