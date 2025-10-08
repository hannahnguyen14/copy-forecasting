"""Tests for HybridApproach class."""
import numpy as np
import pandas as pd
import pytest

from tsfb.base.approaches.hybrid import HybridApproach
from tsfb.base.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.models.base_model import ModelBase


class MockForecastModel(ModelBase):
    """Mock model for testing."""

    def __init__(self):
        self.fitted = False
        self.train_data = None

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.fitted = True
        self.train_data = train_data
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        return pd.DataFrame(
            np.ones((horizon, lookback_data.shape[1])),
            columns=lookback_data.columns,
            index=pd.date_range(
                start=lookback_data.index[-1],
                periods=horizon + 1,
                freq=pd.infer_freq(lookback_data.index),
            )[1:],
        )

    def batch_forecast(self, horizon, batch_maker, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        if not isinstance(batch_maker, RollingForecastPredictBatchMaker):
            raise ValueError("Invalid batch maker type")
        n_windows = len(batch_maker._batch_maker.index_list)
        return np.ones((n_windows, horizon, self.train_data.shape[1]))

    @property
    def model_name(self):
        return "MockForecastModel"


@pytest.fixture
def sample_data():
    """Create sample multivariate time series data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    return pd.DataFrame(
        {
            "A": np.random.randn(100),
            "B": np.random.randn(100),
            "C": np.random.randn(100),
            "D": np.random.randn(100),
        },
        index=dates,
    )


@pytest.fixture
def sample_config():
    """Create sample hybrid approach configuration."""
    return {
        "group": [
            {
                "series": ["A", "B"],
                "sub_approach": "univariate_per_series",
                "model_name": "mock_model",
                "model_hyper_params": {},
            },
            {
                "series": ["C", "D"],
                "sub_approach": "multivariate",
                "model_name": "mock_model",
                "model_hyper_params": {},
            },
        ]
    }


@pytest.fixture
def sample_covariates(sample_data):
    """Create sample covariates."""
    return {
        "past_covariates": pd.DataFrame(
            np.random.randn(100, 2), index=sample_data.index, columns=["cov1", "cov2"]
        )
    }


def test_init(sample_config):
    """Test initialization."""
    approach = HybridApproach(sample_config)
    assert approach.config == sample_config
    assert len(approach.feature_names) == 0
    assert len(approach.models) == 0


def test_forecast_fit(monkeypatch, sample_data, sample_config, sample_covariates):
    """Test model fitting."""

    def mock_get_model(*args, **kwargs):
        return MockForecastModel()

    monkeypatch.setattr("tsfb.base.approaches.hybrid.get_single_model", mock_get_model)

    approach = HybridApproach(sample_config)

    # Test basic fit
    approach.forecast_fit(sample_data)
    assert len(approach.models) == 2  # One for each group
    assert "A,B" in approach.models
    assert "C,D" in approach.models

    for group_key, (sub_approach, cols) in approach.models.items():
        if group_key == "A,B":
            assert len(sub_approach.models) == 2  # UnivariateToMultivariate
            for model in sub_approach.models.values():
                assert model.fitted
        else:
            assert sub_approach.model is not None  # DefaultApproach
            assert sub_approach.model.fitted

    # Test fit with covariates
    approach = HybridApproach(sample_config)
    approach.forecast_fit(sample_data, covariates=sample_covariates)
    assert len(approach.models) == 2


def test_forecast(monkeypatch, sample_data, sample_config, sample_covariates):
    """Test forecasting."""

    def mock_get_model(*args, **kwargs):
        return MockForecastModel()

    monkeypatch.setattr("tsfb.base.approaches.hybrid.get_single_model", mock_get_model)

    approach = HybridApproach(sample_config)
    horizon = 5

    # Test forecast before fitting
    with pytest.raises(RuntimeError):
        approach.forecast(horizon, sample_data)

    # Test basic forecast
    approach.forecast_fit(sample_data)
    result = approach.forecast(horizon, sample_data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (horizon, 4)  # All columns from both groups
    assert all(col in result.columns for col in sample_data.columns)
    assert isinstance(result.index, pd.DatetimeIndex)

    # Test forecast with covariates
    result = approach.forecast(horizon, sample_data, covariates=sample_covariates)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (horizon, 4)


def test_batch_forecast(monkeypatch, sample_data, sample_config, sample_covariates):
    """Test batch forecasting."""

    def mock_get_model(*args, **kwargs):
        return MockForecastModel()

    monkeypatch.setattr("tsfb.base.approaches.hybrid.get_single_model", mock_get_model)

    approach = HybridApproach(sample_config)
    horizon = 5

    # Create batch maker
    eval_maker = RollingForecastEvalBatchMaker(
        series=sample_data, index_list=[10, 20, 30], covariates=sample_covariates
    )
    batch_maker = RollingForecastPredictBatchMaker(eval_maker)

    # Test without fitting
    with pytest.raises(RuntimeError):
        approach.batch_forecast(horizon, batch_maker)

    # Test normal batch forecast
    approach.forecast_fit(sample_data)
    result = approach.batch_forecast(horizon, batch_maker)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 15  # 3 windows * 5 horizon steps
    assert result.shape[1] == 4  # All columns from both groups
    assert isinstance(result.index, pd.DatetimeIndex)

    # Test with invalid batch maker
    with pytest.raises(ValueError):
        approach.batch_forecast(horizon, None)


def test_invalid_config():
    """Test configuration validation."""
    # Test empty config
    empty_config = {}
    with pytest.raises(KeyError, match="group"):
        HybridApproach(empty_config).forecast_fit(pd.DataFrame())

    # Test invalid sub_approach
    config = {
        "group": [
            {"series": ["A"], "sub_approach": "invalid", "model_name": "mock_model"}
        ]
    }
    with pytest.raises(ValueError, match="Unknown sub_approach"):
        HybridApproach(config).forecast_fit(pd.DataFrame({"A": [1, 2, 3]}))
