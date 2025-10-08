from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.approaches.default import DefaultApproach
from tsfb.base.approaches.univariate import UnivariateToMultivariate
from tsfb.base.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.models.base_model import BatchMaker
from tsfb.base.models.model_loader import get_single_model


class HybridApproach(ForecastingApproach):
    """
    A hybrid forecasting approach that combines multiple sub-approaches (univariate or
    multivariate) applied to specified groups of time series columns.

    This class allows different forecasting strategies to be applied to different
    subsets of series. Each group is associated with a sub-approach and a model.

    Attributes:
        config (Dict[str, Any]): Configuration containing group info and model settings.
        feature_names (List[str]): Names of time series features in the training data.
        models (Dict[str, Any]): Mapping of series group to trained forecasting
        approach.

    Methods:
        forecast_fit(train_data, covariates, **kwargs): Fit forecasting models for each
            series group as defined in the config.
        forecast(horizon, lookback_data, covariates, **kwargs): Generate forecast
            results for each group and combine them into a full output.
        batch_forecast(horizon, batch_maker, covariates, **kwargs): Perform
            rolling batch forecasting using stored models for each group.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HybridApproach with a configuration of forecasting groups.

        Each group in the config specifies:
            - The series (columns) it applies to
            - The sub-approach to use ("univariate_per_series" or "multivariate")
            - The model name and its hyperparameters

        Args:
            config (Dict[str, Any]): Configuration for defining model groups and
                settings.
        """
        super().__init__()
        self.config: Dict[str, Any] = config
        self.feature_names: List[str] = []
        self.models: Dict[str, Any] = {}

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> "HybridApproach":
        """
        Fit forecasting models for each group of time series defined in the config.

        For each group, creates a forecasting approach (univariate or multivariate)
        and trains it using the specified model.

        Args:
            train_data (pd.DataFrame): Input training data with multiple time series.
            covariates (Optional[pd.DataFrame]): Optional external covariates.
            **kwargs: Additional arguments passed to the sub-approach fit method.

        Returns:
            HybridApproach: The fitted instance.

        Raises:
            KeyError: If config does not contain 'group' key.
            ValueError: If sub_approach is not one of
            "univariate_per_series" or "multivariate".
        """
        if "group" not in self.config:
            raise KeyError("Required key 'group' not found in config")

        self.feature_names = train_data.columns.tolist()

        for grp in self.config["group"]:
            cols = grp["series"]
            sub = grp["sub_approach"]

            # Validate sub_approach before trying to load model
            if sub not in ["univariate_per_series", "multivariate"]:
                raise ValueError(f"Unknown sub_approach: {sub}")

            base = get_single_model(
                grp["model_name"], grp.get("model_hyper_params", {})
            )

            def make_factory(base_model):
                return lambda: base_model

            approach: ForecastingApproach
            if sub == "univariate_per_series":
                approach = UnivariateToMultivariate(model_factory=make_factory(base))
            else:  # sub == "multivariate"
                approach = DefaultApproach(model_factory=make_factory(base))

            subset_df = train_data[cols]
            approach.forecast_fit(subset_df, covariates=covariates, **kwargs)
            self.models[",".join(cols)] = (approach, cols)

        return self

    def forecast(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate forecasts for all groups and combine results into a single DataFrame.

        Args:
            horizon (int): Forecast horizon.
            lookback_data (pd.DataFrame): Recent time series values used for
                forecasting.
            covariates (Optional[pd.DataFrame]): Optional future covariates.
            **kwargs: Additional arguments passed to the sub-approach forecast method.

        Returns:
            pd.DataFrame: Combined forecast results for all series.

        Raises:
            RuntimeError: If models have not been fitted before calling this method.
        """
        if not self.models:
            raise RuntimeError("Must call forecast_fit() before forecast()")

        outputs = []
        for approach, cols in self.models.values():
            outputs.append(
                approach.forecast(
                    horizon, lookback_data[cols], covariates=covariates, **kwargs
                )
            )
        return pd.concat(outputs, axis=1)

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Perform batch (rolling) forecast for each group of time series.

        This method splits the input batch maker by series group, performs rolling
        forecasting using each group's trained approach, and combines the results.

        Args:
            horizon (int): Number of future steps to forecast.
            batch_maker (BatchMaker): Wrapper containing historical data and rolling
                evaluation setup.
            covariates (Optional[Dict[str, pd.DataFrame]]): Optional dict of covariates
                for each forecast time step.
            **kwargs: Additional arguments passed to the sub-approach batch_forecast.

        Returns:
            pd.DataFrame: Concatenated batch forecast results across all series groups.

        Raises:
            RuntimeError: If models have not been fitted before calling this method.
            ValueError: If batch_maker is not wrapping a RollingForecastEvalBatchMaker.
        """
        if not self.models:
            raise RuntimeError("Must call forecast_fit() before batch_forecast()")

        # unwrap batch_maker
        inner = getattr(batch_maker, "_batch_maker", None)
        if inner is None:
            raise ValueError("batch_maker must wrap RollingForecastEvalBatchMaker")

        if isinstance(inner.series, np.ndarray):
            full_df = getattr(
                inner,
                "series_df",
                pd.DataFrame(inner.series, columns=self.feature_names),
            )
        else:
            full_df = inner.series

        results: List[pd.DataFrame] = []

        for approach, target_cols in self.models.values():
            sub_series = full_df[target_cols]

            sub_eval = RollingForecastEvalBatchMaker(
                series=sub_series,
                index_list=inner.index_list,
                covariates=inner.covariates,
            )

            sub_pred = RollingForecastPredictBatchMaker(sub_eval)
            df_pred = approach.batch_forecast(horizon, sub_pred, **kwargs)
            results.append(df_pred)

        return pd.concat(results, axis=1)
