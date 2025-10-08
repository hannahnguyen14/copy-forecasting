# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.evaluation.metrics.metrics import get_instantiated_metric_dict
from tsfb.base.evaluation.strategy.constants import FieldNames
from tsfb.base.evaluation.strategy.forecasting import ForecastingStrategy
from tsfb.base.evaluation.strategy.metric_utils import evaluate_metrics
from tsfb.base.utils.data_processing import split_time


@dataclass
class ForecastConfig:
    """Configuration for forecasting.

    A dataclass containing configuration parameters for the forecasting process.

    Attributes:
        horizon (int): The forecast horizon (number of future time steps to predict)
        save_tp (bool): Whether to save true and predicted values
        train_ratio (float): Ratio of training data in train+validation split
        train_len (int): Length of training data
        is_multi (bool): Whether this is a multivariate forecast
    """

    horizon: int
    save_tp: bool
    train_ratio: float
    train_len: int
    is_multi: bool


@dataclass
class CovariateData:
    """Container for covariate data.

    Holds training and prediction covariates data.

    Attributes:
        train (Dict[str, pd.DataFrame]): Covariates for training
        pred (Dict[str, pd.DataFrame]): Covariates for prediction
    """

    train: Dict[str, pd.DataFrame]
    pred: Dict[str, pd.DataFrame]

    @classmethod
    def from_raw_data(
        cls, covariates: Optional[Dict[str, pd.DataFrame]], train_len: int
    ) -> "CovariateData":
        """Create CovariateData from raw covariates dictionary.

        Args:
            covariates: Dictionary of covariate DataFrames by type
            train_len: Length of training data for splitting past covariates

        Returns:
            CovariateData instance with processed covariates
        """
        cov_train: Dict[str, pd.DataFrame] = {}
        cov_pred: Dict[str, pd.DataFrame] = {}

        if covariates:
            raw_past = covariates.get("past_covariates")
            raw_future = covariates.get("future_covariates")
            raw_static = covariates.get("static_covariates")

            if raw_past is not None:
                cov_train["past_covariates"] = raw_past.iloc[:train_len]
                cov_pred["past_covariates"] = raw_past.iloc[train_len:]
            if raw_future is not None:
                cov_train["future_covariates"] = raw_future
                cov_pred["future_covariates"] = raw_future
            if raw_static is not None:
                cov_train["static_covariates"] = raw_static
                cov_pred["static_covariates"] = raw_static

        return cls(train=cov_train, pred=cov_pred)


@dataclass
class TrainingData:
    """Container for training and test data.

    Holds the different versions of training data and test data.

    Attributes:
        train_df (pd.DataFrame): Original training data
        test_df (pd.DataFrame): Test data for evaluation
        norm_train_df (pd.DataFrame): Normalized training data
        train_len (int): Length of training data
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    norm_train_df: pd.DataFrame
    train_len: int


@dataclass
class ModelOutput:
    """Container for model training outputs.

    Holds the predictions and timing information from model training.

    Attributes:
        predictions (pd.DataFrame): Model predictions
        fit_time (float): Time taken for model fitting
        predict_time (float): Time taken for prediction
    """

    predictions: pd.DataFrame
    fit_time: float
    predict_time: float


class FixedForecast(ForecastingStrategy):
    """Fixed-window forecasting strategy.

    Implements a forecasting strategy with fixed training and test windows.
    Handles data preparation, model training, prediction and evaluation.

    Attributes:
        REQUIRED_CONFIGS (List[str]): Required configuration parameters
    """

    REQUIRED_CONFIGS = ForecastingStrategy.REQUIRED_CONFIGS + [
        "horizon",
        "save_true_pred",
        "target_channel",
    ]

    @staticmethod
    def _evaluate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scaler: Any,
        metric_names: List[str],
    ) -> List[float]:
        """Compute evaluation metrics."""
        return evaluate_metrics(y_true, y_pred, scaler, metric_names)

    def _get_forecast_config(
        self, series: pd.DataFrame, series_name: str
    ) -> ForecastConfig:
        """Get forecast configuration from parameters."""
        horizon = self._get_scalar_config_value("horizon", series_name)
        save_tp = self._get_scalar_config_value("save_true_pred", series_name)
        loader_cfg = self._get_scalar_config_value("data_loader_config", series_name)

        sr = loader_cfg.get("split_ratio", {})
        train_frac = sr.get("train")
        val_frac = sr.get("val")
        test_frac = sr.get("test")
        if None in [train_frac, val_frac, test_frac]:
            raise ValueError(
                "data_loader_config.split_ratio must define 'train', 'val', 'test'"
            )

        tv_ratio = train_frac + val_frac
        if not 0 < tv_ratio < 1:
            raise ValueError(f"Invalid train+val ratio: {tv_ratio}")
        train_ratio = train_frac / tv_ratio

        total_len = len(series)
        train_len = total_len - horizon
        if train_len <= 0:
            raise ValueError("Prediction length exceeds data length")

        is_multi = series_name == "__default__" or series.shape[1] > 1

        return ForecastConfig(
            horizon=horizon,
            save_tp=save_tp,
            train_ratio=train_ratio,
            train_len=train_len,
            is_multi=is_multi,
        )

    def _normalize_data(self, train_df: pd.DataFrame, loader: Any) -> pd.DataFrame:
        """Apply normalization to training data if scaler exists."""
        if (
            loader is not None
            and hasattr(loader, "target_scalers")
            and loader.target_scalers
        ):
            scaler = loader.target_scalers
            return pd.DataFrame(
                scaler.transform(train_df),
                columns=train_df.columns,
                index=train_df.index,
            )
        return train_df

    def _fit_and_predict(
        self,
        approach: ForecastingApproach,
        config: ForecastConfig,
        norm_train_df: pd.DataFrame,
        covariate_data: CovariateData,
    ) -> Tuple[pd.DataFrame, float, float]:
        """Fit model and make predictions."""
        # Fit
        t0 = time.time()
        approach.forecast_fit(
            train_data=norm_train_df,
            covariates=covariate_data.train,
            train_ratio_in_tv=config.train_ratio,
        )
        fit_time = time.time() - t0

        # Forecast
        t2 = time.time()
        pred_df = approach.forecast(
            horizon=config.horizon,
            lookback_data=norm_train_df,
            covariates=covariate_data.pred,
        )
        predict_time = time.time() - t2

        return pred_df, fit_time, predict_time

    # pylint: disable=too-many-arguments
    def _prepare_prediction(
        self,
        pred_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: ForecastConfig,
        series_name: str,
        loader: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare prediction and truth data for evaluation."""
        # Ensure prediction is a DataFrame
        if not isinstance(pred_df, pd.DataFrame):
            pred_df = pd.DataFrame(
                pred_df,
                index=test_df.index,
                columns=test_df.columns if config.is_multi else [series_name],
            )

        # Inverse transform prediction if scaler is available
        if (
            loader is not None
            and hasattr(loader, "target_scalers")
            and loader.target_scalers
        ):
            scaler = loader.target_scalers
            y_pred = scaler.inverse_transform(pred_df)
        else:
            y_pred = pred_df.to_numpy()

        y_true = test_df.to_numpy()

        if not config.is_multi:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

        return y_true, y_pred

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        loader: Any = None,
    ) -> List[Any]:
        """Execute the fixed forecast strategy.

        Main method that orchestrates the forecasting process:
        1. Gets configurations
        2. Prepares data
        3. Trains model and generates predictions
        4. Evaluates results

        Args:
            series: Input time series data
            meta_info: Optional metadata about the series
            approach: Forecasting approach to use
            series_name: Name of the series being forecast
            covariates: Optional covariate data
            loader: Data loader with optional scaler

        Returns:
            List containing metric values and metadata
        """
        # Get configuration
        config = self._get_forecast_config(series, series_name)

        # Split and normalize data
        train_df, test_df = split_time(series, config.train_len)
        norm_train_df = self._normalize_data(train_df, loader)

        # Handle covariates
        covariate_data = CovariateData.from_raw_data(covariates, config.train_len)

        # Fit and predict
        pred_df, fit_time, predict_time = self._fit_and_predict(
            approach, config, norm_train_df, covariate_data
        )

        # Prepare data for evaluation
        y_true, y_pred = self._prepare_prediction(
            pred_df, test_df, config, series_name, loader
        )

        # Calculate metrics
        metric_vals = self._evaluate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            scaler=loader.target_scalers
            if loader and hasattr(loader, "target_scalers")
            else None,
            metric_names=self.evaluator.metric_names,
        )

        # Encode results if needed
        actual_encoded = self._encode_data(test_df) if config.save_tp else np.nan
        pred_encoded = self._encode_data(pred_df) if config.save_tp else np.nan

        return metric_vals + [
            series_name,
            fit_time,
            predict_time,
            actual_encoded,
            pred_encoded,
            "",
        ]

    @staticmethod
    def accepted_metrics() -> List[str]:
        """Get list of accepted metric names.

        Returns:
            List of supported metric names
        """
        return list(get_instantiated_metric_dict().keys())

    @property
    # pylint: disable=duplicate-code
    def field_names(self) -> List[str]:
        """Get names of result fields.

        Returns:
            List of field names for results DataFrame
        """
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]
