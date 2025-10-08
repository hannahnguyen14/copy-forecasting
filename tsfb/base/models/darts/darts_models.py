import contextlib
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel

from tsfb.base.models.base_model import BatchMaker, ModelBase
from tsfb.base.schema.darts_config import (
    AllWindowsForecastConfig,
    CovariateSeries,
    CovariateWindowConfig,
    DartsConfig,
    ForecastWindowConfig,
    TimeSeriesGroup,
)
from tsfb.base.utils.utils import train_val_split
from tsfb.conf.darts_conf import DARTS_STAT_MODELS_NO_SERIES_ARG

logger = logging.getLogger(__name__)


class DartsModelAdapter(ModelBase):
    """
    Adapter for Darts forecasting models.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model_class: type,
        model_args: dict,
        model_name: Optional[str] = None,
        allow_fit_on_eval: bool = False,
        supports_validation: bool = False,
        **kwargs
    ):
        """
        Initialize the DartsModelAdapter.
        :param model_class: Darts model class.
        :param model_args: Model hyperparameters.
        :param model_name: Name of the model.
        :param allow_fit_on_eval: Allow fitting during evaluation.
        :param supports_validation: Whether validation is supported.
        :param kwargs: Additional arguments.
        """
        self.model: Optional[ForecastingModel] = None
        self.model_class = model_class
        self.config = DartsConfig(**{**model_args, **kwargs})
        self._model_name = model_name
        self.allow_fit_on_eval = allow_fit_on_eval
        self.supports_validation = supports_validation
        self._full_future_cov = None
        self._cov_columns = None
        self.train_ratio_in_tv = 1.0
        self._in_columns: Optional[List[str]] = None

    @property
    def model_name(self):
        """
        Get the model name.
        :return: Model name string.
        """
        return self._model_name

    def _model_supports_past_covariates(self) -> bool:
        """
        Check if the model supports past covariates.
        :return: True if supported, else False.
        """
        return getattr(self.model, "supports_past_covariates", False)

    def _model_supports_future_covariates(self) -> bool:
        """
        Check if the model supports future covariates.
        :return: True if supported, else False.
        """
        return getattr(self.model, "supports_future_covariates", False)

    def _model_supports_static_covariates(self) -> bool:
        """
        Check if the model supports static covariates.
        :return: True if supported, else False.
        """
        return getattr(self.model, "supports_static_covariates", False)

    def _model_supports_series_input(self) -> bool:
        """
        :return: True if supported series input., else False.
        """
        model_name = type(self.model).__name__
        return model_name not in DARTS_STAT_MODELS_NO_SERIES_ARG

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        *,
        train_ratio_in_tv: float = 1.0,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs
    ) -> "ModelBase":
        """
        Fit the model to training data, optionally with validation and covariates.
        """
        train_part, valid_part = self._split_and_normalize(
            train_data, train_ratio_in_tv
        )
        self._in_columns = train_part.columns.tolist()
        covariates_split = self._prepare_covariates(train_part, valid_part, covariates)
        self.model = self.model_class(**self.config.get_darts_class_params())
        ts_data = self._convert_to_timeseries_grouped(
            train_part, valid_part, covariates_split, covariates
        )
        covs = CovariateSeries(
            past_train=ts_data["pc_ts_train"],
            past_val=ts_data["pc_ts_val"],
            future_train=ts_data["fc_ts_train"],
            future_val=ts_data["fc_ts_val"],
        )
        fit_kwargs = self._build_fit_kwargs(
            ts_data["ts_train"],
            ts_data["ts_valid"],
            covs,
        )
        with self._suppress_lightning_logs():
            self.model.fit(**fit_kwargs)

        return self

    def _split_and_normalize(
        self, train_data: pd.DataFrame, train_ratio_in_tv: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_part, valid_part = self._split_train_valid(train_data, train_ratio_in_tv)
        return train_part, valid_part

    def _convert_to_timeseries_grouped(
        self,
        train_part: pd.DataFrame,
        valid_part: pd.DataFrame,
        covariates_split: Tuple[
            Optional[pd.DataFrame],
            Optional[pd.DataFrame],
            Optional[pd.DataFrame],
            Optional[pd.DataFrame],
        ],
        covariates: Optional[Dict[str, pd.DataFrame]],
    ) -> Dict[str, Optional["TimeSeries"]]:
        covs = CovariateSeries(
            past_train=covariates_split[0],
            past_val=covariates_split[1],
            future_train=covariates_split[2],
            future_val=covariates_split[3],
        )
        ts_group = self._convert_to_timeseries(train_part, valid_part, covs, covariates)

        return {
            "ts_train": ts_group.ts_train,
            "ts_valid": ts_group.ts_valid,
            "pc_ts_train": ts_group.covariates.past_train,
            "pc_ts_val": ts_group.covariates.past_val,
            "fc_ts_train": ts_group.covariates.future_train,
            "fc_ts_val": ts_group.covariates.future_val,
        }

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Forecast future values for a given horizon.
        :param horizon: Number of steps to forecast.
        :param series: Input time series as DataFrame.
        :param covariates: Optional covariates dictionary.
        :param kwargs: Additional arguments.
        :return: Forecasted values as numpy array.
        """
        _ = kwargs
        if isinstance(series.index, pd.DatetimeIndex) and series.index.freq is None:
            inferred = pd.infer_freq(series.index)
            if inferred:
                series = series.asfreq(inferred)
        past_cov, future_cov = self._extract_forecast_covariates(
            series, horizon, covariates
        )
        sc = covariates.get("static_covariates") if covariates else None
        use_sc = sc if self._model_supports_static_covariates() else None
        series_ts = TimeSeries.from_dataframe(series, static_covariates=use_sc)
        predict_kwargs = self._build_predict_kwargs(
            horizon, series_ts, past_cov, future_cov
        )
        with self._suppress_lightning_logs():
            assert self.model is not None
            preds_ts = self.model.predict(**predict_kwargs)
        preds = preds_ts.values()

        return preds

    def _split_train_valid(self, train_data, train_ratio):
        """
        Split training data into train and validation sets.
        :param train_data: Input training data.
        :param train_ratio: Ratio for splitting.
        :return: Tuple of (train, validation) DataFrames.
        """
        if self.allow_fit_on_eval or self.model_name == "RegressionModel":
            return train_data, None
        return train_val_split(
            train_data, train_ratio, self.config.get("input_chunk_length", 0)
        )

    def _prepare_covariates(self, train_idx, valid_idx, covariates):
        """
        Prepare covariate data for training and validation.
        :param train_idx: Training data indices.
        :param valid_idx: Validation data indices.
        :param covariates: Covariates dictionary.
        :return: Tuple of (past_cov_train, past_cov_val,
            future_cov_train, future_cov_val).
        """
        pc_train = pc_val = fc_train = fc_val = None
        if covariates:
            pc = covariates.get("past_covariates")
            if pc is not None:
                pc_train = pc.loc[train_idx.index]
                pc_val = pc.loc[valid_idx.index] if valid_idx is not None else None
                self._cov_columns = pc.columns.tolist()
            fc = covariates.get("future_covariates")
            if fc is not None:
                self._full_future_cov = fc
                fc_train = fc
                fc_val = fc
        return pc_train, pc_val, fc_train, fc_val

    def _convert_to_timeseries(
        self,
        train: pd.DataFrame,
        valid: Optional[pd.DataFrame],
        covs: CovariateSeries,
        covariates: Optional[Dict[str, pd.DataFrame]],
    ) -> TimeSeriesGroup:
        """
        Convert data and covariates to Darts TimeSeries objects.
        """
        ts_train, ts_valid = self._convert_main_series(
            train, valid, self._get_static_covariates(covariates)
        )
        pt, pv = self._convert_covariates(covs.past_train, covs.past_val)
        ft, fv = self._convert_covariates(covs.future_train, covs.future_val)
        return TimeSeriesGroup(
            ts_train,
            ts_valid,
            CovariateSeries(pt, pv, ft, fv),
        )

    def _get_static_covariates(self, covariates):
        sc = covariates.get("static_covariates") if covariates else None
        return sc if self._model_supports_static_covariates() else None

    def _convert_main_series(self, train, valid, static_covariates):
        if isinstance(train.index, pd.DatetimeIndex) and train.index.freq is None:
            inferred = pd.infer_freq(train.index)
            if inferred:
                train = train.asfreq(inferred)
        if (
            valid is not None
            and isinstance(valid.index, pd.DatetimeIndex)
            and valid.index.freq is None
        ):
            inferred = pd.infer_freq(valid.index)
            if inferred:
                valid = valid.asfreq(inferred)
        ts_train = TimeSeries.from_dataframe(train, static_covariates=static_covariates)
        ts_valid = (
            TimeSeries.from_dataframe(valid, static_covariates=static_covariates)
            if valid is not None
            else None
        )
        return ts_train, ts_valid

    def _convert_covariates(self, cov_train, cov_val):
        ts_train = (
            TimeSeries.from_dataframe(cov_train) if cov_train is not None else None
        )
        ts_val = TimeSeries.from_dataframe(cov_val) if cov_val is not None else None
        return ts_train, ts_val

    def _build_fit_kwargs(
        self,
        ts_train: TimeSeries,
        ts_valid: Optional[TimeSeries],
        covs: CovariateSeries,
    ) -> Dict:
        kwargs = {"series": ts_train}
        if self.supports_validation and ts_valid:
            kwargs["val_series"] = ts_valid
        if self._model_supports_past_covariates():
            if covs.past_train is not None:
                kwargs["past_covariates"] = covs.past_train
            if self.supports_validation and covs.past_val is not None:
                kwargs["val_past_covariates"] = covs.past_val
        if self._model_supports_future_covariates():
            if covs.future_train is not None:
                kwargs["future_covariates"] = covs.future_train
            if self.supports_validation and covs.future_val is not None:
                kwargs["val_future_covariates"] = covs.future_val

        return kwargs

    def _extract_forecast_covariates(self, series, horizon, covariates):
        """
        Extract covariates for forecasting.
        :param series: Input series DataFrame.
        :param horizon: Forecast horizon.
        :param covariates: Covariates dictionary.
        :return: Tuple of (past_cov, future_cov) as TimeSeries or None.
        """
        past_cov = future_cov = None
        if covariates:
            if "past_covariates" in covariates:
                past_cov = TimeSeries.from_dataframe(covariates["past_covariates"])
            if "future_covariates" in covariates:
                future_cov = TimeSeries.from_dataframe(covariates["future_covariates"])
        elif self._full_future_cov is not None:
            delta = series.index.freq or (series.index[1] - series.index[0])
            start = series.index[-1] + delta
            future_idx = pd.date_range(start=start, periods=horizon, freq=delta)
            future_cov = TimeSeries.from_dataframe(
                self._full_future_cov.reindex(future_idx)
            )
        return past_cov, future_cov

    def _build_predict_kwargs(self, horizon, series_ts, past_cov, future_cov):
        """
        Build keyword arguments for model.predict().
        :param horizon: Forecast horizon.
        :param series_ts: Input TimeSeries.
        :param past_cov: Past covariates TimeSeries.
        :param future_cov: Future covariates TimeSeries.
        :return: Dictionary of predict arguments.
        """
        predict_kwargs = {"n": horizon}
        if self._model_supports_series_input():
            predict_kwargs["series"] = series_ts
        if self._model_supports_past_covariates() and past_cov is not None:
            predict_kwargs["past_covariates"] = past_cov
        if self._model_supports_future_covariates() and future_cov is not None:
            predict_kwargs["future_covariates"] = future_cov
        return predict_kwargs

    @contextlib.contextmanager
    def _suppress_lightning_logs(self):
        """
        Context manager to suppress PyTorch Lightning logs during model fit/predict.
        """
        pl = logging.getLogger("pytorch_lightning")
        old = pl.level
        pl.setLevel(logging.CRITICAL)
        try:
            yield
        finally:
            pl.setLevel(old)

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """
        Perform batch forecasting over multiple windows.
        """
        eval_maker = self._get_eval_maker(batch_maker)
        n_windows = len(eval_maker.index_list)
        if n_windows == 0:
            return np.empty((0, horizon, getattr(self.model, "n_series", 1)))
        lookback = self.config.get("input_chunk_length")
        output_len = self.config.get("output_chunk_length")
        win_size = lookback + output_len
        x_windows, cov_dict, full_idx = self._get_batch_data(
            batch_maker, eval_maker, n_windows, win_size
        )
        config = AllWindowsForecastConfig(
            eval_maker=eval_maker,
            x_windows=x_windows,
            cov_dict=cov_dict,
            full_idx=full_idx,
            lookback=lookback,
            output_len=output_len,
        )
        return self._forecast_all_windows(config, **kwargs)

    def _get_eval_maker(self, batch_maker: BatchMaker):
        eval_maker = getattr(batch_maker, "_batch_maker", None)
        if eval_maker is None:
            raise ValueError("batch_maker must wrap RollingForecastEvalBatchMaker")
        return eval_maker

    def _get_batch_data(
        self,
        batch_maker: BatchMaker,
        eval_maker,
        n_windows: int,
        win_size: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.Index]:
        batch = batch_maker.make_batch(batch_size=n_windows, win_size=win_size)
        x_windows = batch["input"]
        cov_dict = batch.get("covariates", {})
        full_idx = getattr(eval_maker, "index", None)
        if full_idx is None:
            full_idx = pd.RangeIndex(len(eval_maker.series))
        return x_windows, cov_dict, full_idx

    def _forecast_all_windows(
        self, config: AllWindowsForecastConfig, **kwargs
    ) -> np.ndarray:
        preds = []
        win_size = config.lookback + config.output_len
        for i, start in enumerate(config.eval_maker.index_list):
            fw_config = ForecastWindowConfig(
                i=i,
                start=start,
                full_idx=config.full_idx,
                x_window=config.x_windows[i],
                cov_dict=config.cov_dict,
                lookback=config.lookback,
                output_len=config.output_len,
                win_size=win_size,
            )
            preds.append(self._forecast_one_window(fw_config, **kwargs))

        return np.stack([np.asarray(p) for p in preds], axis=0)

    def _forecast_one_window(
        self, config: ForecastWindowConfig, **kwargs
    ) -> np.ndarray:
        window_idx = config.full_idx[config.start - config.win_size : config.start]
        df_in = self._prepare_input_window(config.x_window, window_idx, config.lookback)
        cov_cfg = CovariateWindowConfig(
            config.cov_dict, window_idx, config.i, config.lookback, config.output_len
        )
        cov_input = self._prepare_covariate_window(cov_cfg)

        return self.forecast(
            horizon=config.output_len, series=df_in, covariates=cov_input, **kwargs
        )

    def _prepare_input_window(
        self, x_window: np.ndarray, window_idx: pd.Index, lookback: int
    ) -> pd.DataFrame:
        """
        Prepare the input dataframe for forecasting.
        """
        return pd.DataFrame(
            x_window[-lookback:],
            index=window_idx[-lookback:],
            columns=self._in_columns,
        )

    def _prepare_covariate_window(
        self, config: CovariateWindowConfig
    ) -> Dict[str, pd.DataFrame]:
        cov_input: Dict[str, pd.DataFrame] = {}
        if "past_covariates" in config.cov_dict:
            arr_pc = config.cov_dict["past_covariates"]
            cov_input["past_covariates"] = pd.DataFrame(
                arr_pc[config.window_index, -config.lookback :],
                index=config.window_idx[-config.lookback :],
                columns=self._cov_columns,
            )
        if self._full_future_cov is not None:
            delta = config.window_idx.freq or (
                config.window_idx[1] - config.window_idx[0]
            )
            future_idx = pd.date_range(
                start=config.window_idx[-1] + delta,
                periods=config.output_len,
                freq=delta,
            )
            future_cov_df = self._full_future_cov
            if isinstance(future_cov_df.index, pd.DatetimeIndex):
                target_freq = future_idx.freq or delta
                if future_cov_df.index.freq != target_freq:
                    future_cov_df = future_cov_df.resample(target_freq).ffill()
            cov_input["future_covariates"] = pd.DataFrame(
                future_cov_df.reindex(future_idx),
                index=future_idx,
                columns=future_cov_df.columns,
            )

        return cov_input
