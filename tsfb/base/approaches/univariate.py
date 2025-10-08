from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.models.base_model import BatchMaker, ModelBase


class UnivariateToMultivariate(ForecastingApproach):
    """A univariate forecasting approach that
    processes each time series column independently.

    This class implements the `ForecastingApproach`
    interface for univariate forecasting,
    where each column of the input time series is
    modeled separately using individual models.
    It supports fitting models and generating forecasts or batch forecasts with
    optional covariates.

    Attributes:
        models (Dict[str, ModelBase]): A dictionary mapping column names to
        fitted models.
    """

    def __init__(self, model_factory: Callable[[], ModelBase]):
        super().__init__()
        self.model_factory = model_factory
        self.models: Dict[str, ModelBase] = {}

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> "UnivariateToMultivariate":
        for col in train_data.columns:
            model = self.model_factory()
            model.forecast_fit(train_data[[col]], covariates=covariates, **kwargs)
            self.models[col] = model
        return self

    def forecast(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Model not fitted")

        preds: Dict[str, np.ndarray] = {}
        for col in lookback_data.columns:
            if col not in self.models:
                raise RuntimeError(f"Model not fitted for column {col}")

            model = self.models[col]
            arr = model.forecast(
                horizon, lookback_data[[col]], covariates=covariates, **kwargs
            )
            if hasattr(arr, "values"):
                arr = arr.values
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            preds[col] = arr

        last_ts = lookback_data.index[-1]
        freq = pd.infer_freq(lookback_data.index) or "H"
        idx = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]
        return pd.DataFrame(preds, index=idx)

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Model not fitted")

        eval_orig = getattr(batch_maker, "_batch_maker", None)
        if not isinstance(eval_orig, RollingForecastEvalBatchMaker):
            raise ValueError("batch_maker must wrap RollingForecastEvalBatchMaker")

        try:
            full_index = pd.DatetimeIndex(eval_orig.series.index)
        except Exception:
            full_index = pd.RangeIndex(0, len(eval_orig.series), 1)

        full_np = (
            eval_orig.series
            if isinstance(eval_orig.series, np.ndarray)
            else eval_orig.series.to_numpy()
        )

        def make_col_df(col_idx, col_name):
            col_np = full_np[:, col_idx : col_idx + 1]
            cov_np = self._process_covariates(covariates, col_idx)
            col_df = pd.DataFrame(col_np, full_index, [col_name])
            sub_eval = RollingForecastEvalBatchMaker(
                series=col_df,
                index_list=eval_orig.index_list,
                covariates=cov_np,
            )
            sub_pred = RollingForecastPredictBatchMaker(sub_eval)
            arr = self.models[col_name].batch_forecast(horizon, sub_pred, **kwargs)
            return pd.concat(
                [
                    pd.DataFrame(
                        arr[win_i, :, 0],
                        index=self._create_window_index(full_index, start, horizon),
                        columns=[col_name],
                    )
                    for win_i, start in enumerate(eval_orig.index_list)
                ],
                axis=0,
            )

        all_col_dfs = [
            make_col_df(col_idx, col_name)
            for col_idx, col_name in enumerate(self.models.keys())
        ]
        return pd.concat(all_col_dfs, axis=1)

    def _process_covariates(
        self, covariates: Optional[Dict[str, pd.DataFrame]], col_idx: int
    ) -> Dict[str, pd.DataFrame]:
        """Process covariates for a specific column index into DataFrames.

        Args:
            covariates: Dictionary of covariate DataFrames or arrays.
            col_idx: Column index to extract from each covariate.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of covariate names to
                single-column DataFrames for the specified column.
        """
        cov_df: Dict[str, pd.DataFrame] = {}
        if covariates:
            for k, df in covariates.items():
                if isinstance(df, pd.DataFrame):
                    cov_df[k] = df.iloc[:, col_idx : col_idx + 1]
                else:
                    cov_df[k] = pd.DataFrame(df[:, col_idx : col_idx + 1])
        return cov_df

    def _create_window_index(
        self, full_index: pd.Index, start: int, horizon: int
    ) -> pd.Index:
        """Create an index for a forecast window.

        Args:
            full_index: Full index of the time series.
            start: Starting index for the forecast window.
            horizon: Number of time steps to forecast.

        Returns:
            pd.Index: A pandas Index (DatetimeIndex or RangeIndex) for the
                forecast window.
        """
        try:
            win_idx = full_index[start : start + horizon]
        except Exception:
            win_idx = pd.RangeIndex(start, start + horizon)
        return win_idx
