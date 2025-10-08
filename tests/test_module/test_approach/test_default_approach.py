"""Tests for DefaultApproach class."""
import numpy as np
import pandas as pd
import pytest

from tsfb.base.approaches.default import DefaultApproach
from tsfb.base.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.models.base_model import ModelBase


class MockModel(ModelBase):
    """Mock model for testing DefaultApproach."""

    def __init__(self, return_type="dataframe"):
        self.return_type = return_type
        self.fitted = False
        self.forecast_params = None
        self.train_data = None

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.fitted = True
        self.train_data = train_data
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model not fitted")

        self.forecast_params = {
            "horizon": horizon,
            "lookback_data": lookback_data,
            "covariates": covariates,
            "kwargs": kwargs,
        }

        n_cols = lookback_data.shape[1]
        if self.return_type == "dataframe":
            return pd.DataFrame(
                np.ones((horizon, n_cols)),
                columns=lookback_data.columns,
                index=pd.date_range(
                    start=lookback_data.index[-1],
                    periods=horizon + 1,
                    freq=pd.infer_freq(lookback_data.index),
                )[1:],
            )
        elif self.return_type == "array":
            return np.ones((horizon, n_cols))
        elif self.return_type == "series":
            # For series return type, convert to DataFrame with all columns
            series_data = pd.Series(np.ones(horizon), name=lookback_data.columns[0])
            return pd.DataFrame({col: series_data for col in lookback_data.columns})
        else:
            return "invalid"

    def batch_forecast(self, horizon, batch_maker, **kwargs):
        if not isinstance(batch_maker, RollingForecastPredictBatchMaker):
            raise ValueError("Invalid batch maker type")
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        return np.ones(
            (
                len(batch_maker._batch_maker.index_list),
                horizon,
                self.train_data.shape[1],
            )
        )

    @property
    def model_name(self):
        return "MockModel"


@pytest.fixture
def model_factory():
    """Create model factory for testing."""
    return lambda: MockModel()


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    return pd.DataFrame(
        {"A": np.random.randn(100), "B": np.random.randn(100)}, index=dates
    )


@pytest.fixture
def sample_covariates(sample_data):
    """Create sample covariates."""
    return {
        "past_covariates": pd.DataFrame(
            np.random.randn(100, 2), index=sample_data.index, columns=["cov1", "cov2"]
        )
    }


def test_init(model_factory):
    """Test initialization."""
    approach = DefaultApproach(model_factory)
    assert approach.model is None
    assert callable(approach.model_factory)


def test_forecast_fit(model_factory, sample_data, sample_covariates):
    """Test model fitting."""
    approach = DefaultApproach(model_factory)

    # Test basic fit
    approach.forecast_fit(sample_data)
    assert approach.model is not None
    assert approach.model.fitted

    # Test fit with covariates
    approach.forecast_fit(sample_data, covariates=sample_covariates)
    assert approach.model.fitted


def test_forecast(model_factory, sample_data, sample_covariates):
    """Test forecasting."""
    approach = DefaultApproach(model_factory)
    horizon = 5

    # Test forecast without fitting
    with pytest.raises(RuntimeError):
        approach.forecast(horizon, sample_data)

    # Test basic forecast
    approach.forecast_fit(sample_data)
    result = approach.forecast(horizon, sample_data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (horizon, 2)
    assert isinstance(result.index, pd.DatetimeIndex)

    # Test forecast with covariates
    result = approach.forecast(horizon, sample_data, covariates=sample_covariates)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (horizon, 2)

    # Test different return types
    for return_type in ["array", "series", "dataframe"]:
        approach = DefaultApproach(lambda: MockModel(return_type))
        approach.forecast_fit(sample_data)
        result = approach.forecast(horizon, sample_data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (horizon, 2)

    # Test invalid return type
    approach = DefaultApproach(lambda: MockModel("invalid"))
    approach.forecast_fit(sample_data)
    with pytest.raises(TypeError):
        approach.forecast(horizon, sample_data)


def test_batch_forecast(model_factory, sample_data):
    """Test batch forecasting."""
    approach = DefaultApproach(model_factory)
    horizon = 5

    # Test without fitting
    eval_maker = RollingForecastEvalBatchMaker(
        series=sample_data, index_list=[10, 20, 30]
    )
    batch_maker = RollingForecastPredictBatchMaker(eval_maker)

    with pytest.raises(RuntimeError):
        approach.batch_forecast(horizon, batch_maker)

    # Test normal batch forecast
    approach.forecast_fit(sample_data)
    result = approach.batch_forecast(horizon, batch_maker)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 15  # 3 windows * 5 horizon steps
    assert isinstance(result.index, pd.DatetimeIndex)

    # Test with invalid batch maker
    with pytest.raises(ValueError):
        approach.batch_forecast(horizon, None)


def test_create_forecast_index(model_factory, sample_data):
    """Test index creation for forecasts."""
    approach = DefaultApproach(model_factory)

    # Test with valid slice
    idx = approach._create_forecast_index(
        sample_data.index, start=5, horizon=3, arr_shape=3
    )
    assert len(idx) == 3
    assert isinstance(idx, pd.DatetimeIndex)

    # Test with incompatible array shape
    idx = approach._create_forecast_index(
        sample_data.index, start=5, horizon=3, arr_shape=4
    )
    assert len(idx) == 4
    assert isinstance(idx, pd.DatetimeIndex)
