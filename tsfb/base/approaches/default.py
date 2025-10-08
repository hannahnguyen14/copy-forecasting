from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.models.base_model import BatchMaker, ModelBase


class DefaultApproach(ForecastingApproach):
    """A default forecasting approach that wraps a model
    factory for time series forecasting.

    This class implements the `ForecastingApproach` interface, providing
    methods to fit a model
    and generate forecasts or batch forecasts using a provided model factory.
    The model is initialized during fitting and used for
    subsequent forecasting tasks.

    Attributes:
        model_factory (Callable[[], ModelBase]): A callable that returns a new
            instance of a `ModelBase`-derived model.
        model (Optional[ModelBase]): The instantiated model,
            set after calling `forecast_fit`.
    """

    def __init__(self, model_factory: Callable[[], ModelBase]):
        """Initialize the DefaultApproach with a model factory.

        Args:
            model_factory (Callable[[], ModelBase]): A callable that
                creates a new instance of
                a `ModelBase`-derived model when invoked.
        """
        super().__init__()
        self.model_factory = model_factory
        self.model: Optional[ModelBase] = None

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ):
        """Fit the forecasting model using the provided training data.

        Instantiates a new model using the model factory and
        fits it with the training data and optional covariates.

        Args:
            train_data (pd.DataFrame): The training data as a pandas
                DataFrame, typically
                containing time series data with a DatetimeIndex.
            covariates (Optional[Dict[str, pd.DataFrame]]):
                Optional dictionary of covariate
                DataFrames, where keys are covariate names and values
                are DataFrames with matching indices. Defaults to None.
            **kwargs: Additional keyword arguments passed to the underlying model's
                `forecast_fit` method.
        """
        self.model = self.model_factory()
        self.model.forecast_fit(train_data, covariates=covariates, **kwargs)

    def forecast(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate forecasts for the specified horizon using lookback data.

        Produces forecasts using the fitted model. The output is a pandas DataFrame with
        a DatetimeIndex, ensuring compatibility with time series data.

        Args:
            horizon (int): The number of time steps to forecast into the future.
            lookback_data (pd.DataFrame): Historical data used as input for forecasting,
                typically containing time series data with a DatetimeIndex.
            covariates (Optional[Dict[str, pd.DataFrame]]): Optional dictionary
                of covariate DataFrames for the forecast period. Defaults to None.
            **kwargs: Additional keyword arguments passed to the underlying model's
                `forecast` method.

        Returns:
            pd.DataFrame: A DataFrame containing the forecasted values,
                with a DatetimeIndex and columns matching `lookback_data`.

        Raises:
            RuntimeError: If the model is not trained
                (i.e., `forecast_fit` has not been called).
            TypeError: If the model's forecast output is of an unsupported type.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call forecast_fit() first.")

        pred = self.model.forecast(
            horizon, lookback_data, covariates=covariates, **kwargs
        )

        if isinstance(pred, np.ndarray):
            pred = pd.DataFrame(pred, columns=lookback_data.columns)
        elif isinstance(pred, pd.Series):
            pred = pred.to_frame()
        elif not isinstance(pred, pd.DataFrame):
            raise TypeError(f"Unsupported prediction type: {type(pred)}")

        assert isinstance(pred, pd.DataFrame)

        if pred.index is None or not isinstance(pred.index, pd.DatetimeIndex):
            last_index = lookback_data.index[-1]
            freq = pd.infer_freq(lookback_data.index) or "H"
            pred.index = pd.date_range(
                start=last_index, periods=horizon + 1, freq=freq
            )[1:]

        return pred

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        covariates: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate batch forecasts for multiple time series segments.

        Uses a batch maker to process multiple forecasting
        tasks and combines the results
        into a single DataFrame with a DatetimeIndex.

        Args:
            horizon (int): The number of time steps to forecast for each batch.
            batch_maker (BatchMaker): An object that provides batched input data for
                forecasting, expected to wrap a `RollingForecastEvalBatchMaker`.
            covariates (Optional[Dict[str, Any]]): Optional dictionary of covariates for
                the forecast period. Defaults to None.
            **kwargs: Additional keyword arguments passed to the underlying model's
                `batch_forecast` method.

        Returns:
            pd.DataFrame: A concatenated DataFrame containing forecasts for all batches,
                with a DatetimeIndex and columns matching the input series.

        Raises:
            RuntimeError: If the model is not trained
                (i.e., `forecast_fit` has not been called).
            ValueError: If `batch_maker` does not wrap a
                `RollingForecastEvalBatchMaker`.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call forecast_fit() first.")

        arr = self.model.batch_forecast(horizon, batch_maker, **kwargs)

        inner = getattr(batch_maker, "_batch_maker", None)
        if inner is None:
            raise ValueError("batch_maker must wrap a RollingForecastEvalBatchMaker")

        series_idx = inner.series.index
        col_names = inner.series.columns

        dfs = []
        for i, start in enumerate(inner.index_list):
            idx = self._create_forecast_index(series_idx, start, horizon, arr.shape[1])
            df_pred = pd.DataFrame(arr[i], index=idx, columns=col_names)
            dfs.append(df_pred)

        return pd.concat(dfs, axis=0)

    def _create_forecast_index(
        self, series_idx: pd.Index, start: int, horizon: int, arr_shape: int
    ) -> pd.Index:
        """Create a forecast index for a given time series segment.

        Generates a pandas Index for forecasted values, either by slicing the existing
        series index or by creating a new date range if the slice is incompatible with
        the forecast array shape.

        Args:
            series_idx (pd.Index): The index of the original time series data.
            start (int): The starting index for the forecast slice.
            horizon (int): The number of time steps to forecast.
            arr_shape (int): The expected shape of the forecast array
                (number of time steps).

        Returns:
            pd.Index: A pandas Index (typically a DatetimeIndex) for the
                forecasted values.
        """
        idx = series_idx[start : start + horizon]
        if len(idx) != arr_shape:
            freq = pd.infer_freq(series_idx) or "H"
            last = series_idx[start - 1] if start > 0 else series_idx[0]
            idx = pd.date_range(
                last + pd.Timedelta(1, freq), periods=arr_shape, freq=freq
            )
        return idx
