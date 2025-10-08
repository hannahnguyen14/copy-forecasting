# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.evaluation.metrics.metrics import get_instantiated_metric_dict
from tsfb.base.evaluation.strategy.constants import FieldNames
from tsfb.base.evaluation.strategy.forecasting import ForecastingStrategy
from tsfb.base.evaluation.strategy.metric_utils import evaluate_metrics
from tsfb.base.evaluation.strategy.rolling_batch_maker import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.utils.data_processing import split_time


class RollingForecast(ForecastingStrategy):
    """Rolling forecast evaluation strategy.

    This strategy evaluates forecasting models using a rolling window approach,
    where predictions are made repeatedly by moving forward in time.
    """

    REQUIRED_CONFIGS = ForecastingStrategy.REQUIRED_CONFIGS + [
        "horizon",
        "stride",
        "num_rollings",
        "save_true_pred",
        "target_channel",
    ]

    @staticmethod
    def _get_index(
        train_length: int, test_length: int, horizon: int, stride: int
    ) -> List[int]:
        """Calculate indices for rolling windows.

        Args:
            train_length: Length of training data
            test_length: Length of test data
            horizon: Forecast horizon
            stride: Step size between windows

        Returns:
            List of starting indices for rolling windows
        """
        data_len = train_length + test_length
        idxs = list(range(train_length, data_len - horizon + 1, stride))
        if (test_length - horizon) % stride != 0:
            idxs.append(data_len - horizon)
        return idxs

    def _get_split_lens(
        self, series: pd.DataFrame, meta_info: Optional[pd.Series], tv_ratio: float
    ) -> Tuple[int, int]:
        """Calculate lengths for train and test splits.

        Args:
            series: Input time series data
            meta_info: Optional metadata about the series
            tv_ratio: Train+validation ratio

        Returns:
            Tuple of (train_length, test_length)

        Raises:
            ValueError: If resulting train or test length is <= 0
        """
        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = int(tv_ratio * data_len)
        test_length = data_len - train_length
        if train_length <= 0 or test_length <= 0:
            raise ValueError("Length of train/test must be > 0")
        return train_length, test_length

    # pylint: disable=too-many-arguments
    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        loader: Optional[Any] = None,
    ) -> List[Any]:
        # if approach.batch_forecast.__annotations__.get("not_implemented_batch"):
        #     return self._eval_sample(
        #       series, meta_info, approach, series_name, covariates, loader
        #     )
        # else:
        #     return self._eval_batch(
        #       series, meta_info, approach, series_name, covariates, loader
        #     )
        return self._eval_sample(
            series, meta_info, approach, series_name, covariates, loader
        )

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    def _eval_sample(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]],
        loader: Optional[Any],
    ) -> List[Any]:
        # 1. Đọc config
        horizon = self._get_scalar_config_value("horizon", series_name)
        stride = self._get_scalar_config_value("stride", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        save_tp = self._get_scalar_config_value("save_true_pred", series_name)
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        loader_cfg = self._get_scalar_config_value("data_loader_config", series_name)

        # 2. Tách train/test
        (
            train_frac,
            val_frac,
            _,
            train_ratio_in_tv,
        ) = self._extract_loader_config(loader_cfg)
        tv_ratio = train_frac + val_frac
        train_len, test_len = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid, _ = split_time(series, train_len)

        # 3. Lấy target series (1 cột hoặc all)
        target_train_valid = (
            train_valid if target_channel is None else train_valid[[target_channel]]
        )

        # 4. Khởi tạo scaler
        scaler = self._get_eval_scaler(loader, target_train_valid, train_ratio_in_tv)

        # 5. Transform trước khi fit
        if scaler is not None:
            norm_target = pd.DataFrame(
                scaler.transform(target_train_valid),
                index=target_train_valid.index,
                columns=target_train_valid.columns,
            )
        else:
            norm_target = target_train_valid

        # 6. covariates for sample
        cov_train = self._extract_covariates(covariates, train_len)

        # 7. Fit model
        t0 = time.time()
        approach.forecast_fit(
            train_data=norm_target,
            covariates=cov_train,
            train_ratio_in_tv=train_ratio_in_tv,
        )
        t1 = time.time()

        # 8. Rolling sample
        idxs = self._get_index(train_len, test_len, horizon, stride)[:num_rollings]
        all_scores, all_true, all_pred = [], [], []
        total_inf_time = 0.0

        for start in idxs:
            train_i, rest = split_time(series, start)
            test_i = rest.head(horizon)
            y_true_df = test_i if target_channel is None else test_i[[target_channel]]

            lookback = train_i if target_channel is None else train_i[[target_channel]]
            if scaler is not None:
                lookback = pd.DataFrame(
                    scaler.transform(lookback),
                    index=lookback.index,
                    columns=lookback.columns,
                )

            t2 = time.time()
            pred_norm = approach.forecast(
                horizon=horizon,
                lookback_data=lookback,
                covariates=self._extract_covariates_for_prediction(covariates, start),
            )
            t3 = time.time()
            total_inf_time += t3 - t2

            if scaler is not None:
                pred_df = pd.DataFrame(
                    scaler.inverse_transform(pred_norm),
                    index=y_true_df.index,
                    columns=y_true_df.columns,
                )
            else:
                pred_df = (
                    pred_norm
                    if isinstance(pred_norm, pd.DataFrame)
                    else pd.DataFrame(
                        pred_norm, index=y_true_df.index, columns=y_true_df.columns
                    )
                )

            y_true = y_true_df.to_numpy().flatten()
            y_pred = pred_df.to_numpy().flatten()
            score = self.evaluator.evaluate(
                actual=y_true, predicted=y_pred, scaler=None
            )

            all_scores.append(score)
            all_true.append(y_true_df)
            all_pred.append(pred_df)

        mean_scores = np.mean(np.stack(all_scores), axis=0).tolist()
        avg_inf_time = total_inf_time / len(all_scores)
        actual_enc = self._encode_data(all_true) if save_tp else np.nan
        pred_enc = self._encode_data(all_pred) if save_tp else np.nan

        return mean_scores + [
            series_name,
            t1 - t0,
            avg_inf_time,
            actual_enc,
            pred_enc,
            "",
        ]

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    def _eval_batch(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]],
        loader: Optional[Any],
    ) -> List[Any]:
        horizon = self._get_scalar_config_value("horizon", series_name)
        stride = self._get_scalar_config_value("stride", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        save_tp = self._get_scalar_config_value("save_true_pred", series_name)
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        loader_cfg = self._get_scalar_config_value("data_loader_config", series_name)

        (
            train_frac,
            val_frac,
            _,
            train_ratio_in_tv,
        ) = self._extract_loader_config(loader_cfg)
        tv_ratio = train_frac + val_frac
        train_len, test_len = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid, _ = split_time(series, train_len)

        target_train_valid = (
            train_valid if target_channel is None else train_valid[[target_channel]]
        )
        scaler = self._get_eval_scaler(loader, target_train_valid, train_ratio_in_tv)

        series_target = series if target_channel is None else series[[target_channel]]
        if scaler is not None:
            norm_series = pd.DataFrame(
                scaler.transform(series_target),
                index=series_target.index,
                columns=series_target.columns,
            )
        else:
            norm_series = series_target

        cov_train = self._extract_covariates(covariates, train_len)
        t0 = time.time()
        approach.forecast_fit(
            train_data=norm_series.iloc[:train_len],
            covariates=cov_train,
            train_ratio_in_tv=train_ratio_in_tv,
        )
        t1 = time.time()

        idxs = self._get_index(train_len, test_len, horizon, stride)[:num_rollings]
        batch_maker = RollingForecastEvalBatchMaker(norm_series, idxs, covariates)
        predict_maker = RollingForecastPredictBatchMaker(batch_maker)

        all_preds, total_inf_time = [], 0.0
        while predict_maker.has_more_batches():
            t2 = time.time()
            preds_norm = approach.batch_forecast(horizon, predict_maker)
            t3 = time.time()
            total_inf_time += t3 - t2
            all_preds.append(preds_norm)

        all_preds = np.concatenate(all_preds, axis=0)
        targets = batch_maker.make_batch_eval(horizon)["target"]

        if scaler is not None:
            # flatten → (n_windows*horizon, n_series)
            assert isinstance(all_preds, np.ndarray)
            flat = all_preds.reshape(
                -1, all_preds.shape[-1] if all_preds.ndim == 3 else 1
            )
            inv_flat = scaler.inverse_transform(flat)
            assert isinstance(all_preds, np.ndarray)
            inv_preds = inv_flat.reshape(all_preds.shape)
        else:
            inv_preds = all_preds

        all_scores = []
        for pred_window, true_window in zip(inv_preds, targets):
            y_true = true_window.flatten()
            y_pred = pred_window.flatten()
            score = self.evaluator.evaluate(
                actual=y_true, predicted=y_pred, scaler=None
            )
            all_scores.append(score)

        mean_scores = np.mean(np.stack(all_scores), axis=0).tolist()
        avg_inf_time = total_inf_time / len(all_scores)
        actual_enc = self._encode_data(targets) if save_tp else np.nan
        pred_enc = self._encode_data(inv_preds) if save_tp else np.nan

        return mean_scores + [
            series_name,
            t1 - t0,
            avg_inf_time,
            actual_enc,
            pred_enc,
            "",
        ]

    def _extract_loader_config(
        self, loader_cfg: Dict[str, Any]
    ) -> Tuple[float, float, float, float]:
        """Extract and validate configuration from data loader.

        Args:
            loader_cfg: Data loader configuration dictionary

        Returns:
            Tuple of (train_frac, val_frac, test_frac, train_ratio_in_tv)

        Raises:
            ValueError: If split ratios are invalid
        """
        split_ratio = loader_cfg.get("split_ratio", {})
        train_frac = split_ratio.get("train")
        val_frac = split_ratio.get("val")
        test_frac = split_ratio.get("test")
        if train_frac is None or val_frac is None or test_frac is None:
            raise ValueError(
                "data_loader_config.split_ratio must define 'train', 'val' and 'test'"
            )
        tv_ratio = train_frac + val_frac
        if tv_ratio <= 0 or tv_ratio >= 1:
            raise ValueError(f"Invalid train+val ratio: {tv_ratio}")
        train_ratio_in_tv = train_frac / tv_ratio
        return train_frac, val_frac, test_frac, train_ratio_in_tv

    def _extract_covariates(
        self, covariates: Optional[Dict[str, pd.DataFrame]], train_len: int
    ) -> Dict[str, pd.DataFrame]:
        """Extract covariates for training.

        Args:
            covariates: Dictionary of covariate DataFrames
            train_len: Length of training data

        Returns:
            Dictionary of processed covariates for training
        """
        cov_train = {}
        if covariates:
            if "past_covariates" in covariates:
                cov_train["past_covariates"] = covariates["past_covariates"].iloc[
                    :train_len
                ]
            if "future_covariates" in covariates:
                cov_train["future_covariates"] = covariates["future_covariates"]
            if "static_covariates" in covariates:
                cov_train["static_covariates"] = covariates["static_covariates"]
        return cov_train

    def _extract_covariates_for_prediction(
        self, covariates: Optional[Dict[str, pd.DataFrame]], start: int
    ) -> Dict[str, pd.DataFrame]:
        """Extract covariates for prediction.

        Args:
            covariates: Dictionary of covariate DataFrames
            start: Starting index for prediction

        Returns:
            Dictionary of processed covariates for prediction
        """
        cov_pred = {}
        if covariates:
            if "past_covariates" in covariates:
                cov_pred["past_covariates"] = covariates["past_covariates"].iloc[:start]
            if "future_covariates" in covariates:
                cov_pred["future_covariates"] = covariates["future_covariates"]
            if "static_covariates" in covariates:
                cov_pred["static_covariates"] = covariates["static_covariates"]
        return cov_pred

    # pylint: disable=too-many-locals
    def _forecast_one_roll(
        self,
        approach: ForecastingApproach,
        series: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]],
        target_channel: Optional[str],
        start: int,
        horizon: int,
        scaler: Any,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any, float]:
        """Execute one rolling forecast."""
        train_i, rest = split_time(series, start)
        test_i = rest.head(horizon)
        y_true_df = test_i if target_channel is None else test_i[[target_channel]]
        lookback = train_i if target_channel is None else train_i[[target_channel]]
        cov_pred = self._extract_covariates_for_prediction(covariates, start)
        t2 = time.time()
        pred_df = approach.forecast(
            horizon=horizon, lookback_data=lookback, covariates=cov_pred
        )
        t3 = time.time()
        y_true = y_true_df.to_numpy().flatten()
        y_pred = pred_df.to_numpy().flatten()
        score = evaluate_metrics(y_true, y_pred, scaler, self.evaluator.metric_names)
        return y_true_df, pred_df, score, t3 - t2

    @staticmethod
    def accepted_metrics() -> List[str]:
        return list(get_instantiated_metric_dict().keys())

    @property
    # pylint: disable=duplicate-code
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]
