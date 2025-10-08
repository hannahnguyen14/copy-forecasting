from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from tsfb.base.models.base_model import BatchMaker


class ForecastingApproach(ABC):
    """
    The base class for forecasting approaches.
    """

    def __init__(self) -> None:
        self.name: Optional[str] = None
        self.config: Optional[Dict[str, Any]] = None

    @abstractmethod
    def forecast_fit(self, train_data: pd.DataFrame, **kwargs) -> "ForecastingApproach":
        """
        Fit the forecasting model on the training data.

        :param train_data: The training data as a pandas DataFrame.
        :return: An instance of the fitted forecasting approach.
        """

    @abstractmethod
    def forecast(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predict future values based on the test data.

        :return: A DataFrame containing the predicted values.
        """

    @abstractmethod
    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predict future values for *all* remaining windows provided by a BatchMaker.

        :param horizon: Number of time steps to predict for each window.
        :param batch_maker: An instance of BatchMaker
            (e.g. RollingForecastPredictBatchMaker)
            that yields batches of look-back windows.
        :param covariates: (Optional) any global covariates dict to pass down.
        :return: A concatenated DataFrame of all window-level forecasts,
                 indexed appropriately over the union of all forecasted timestamps.
        """
