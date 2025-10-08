import abc
import traceback
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.data_layer.factory import DataLoaderFactory
from tsfb.base.evaluation.strategy.constants import FieldNames
from tsfb.base.evaluation.strategy.strategy import Strategy
from tsfb.base.utils.data_processing import split_time
from tsfb.base.utils.random_utils import fix_all_random_seed, fix_random_seed


class ForecastingStrategy(Strategy, metaclass=abc.ABCMeta):
    """
    The base class for forecasting strategies
    """

    REQUIRED_CONFIGS = [
        "seed",
        "deterministic",
        "data_loader_config",
    ]

    # pylint: disable=too-many-locals
    def execute(self, data_name: str, approach: ForecastingApproach) -> Any:
        """
        Entry point to run forecasting strategy

        :param data_name: Used as identifier (e.g., file name)
        :param approach: A ForecastingApproach instance,
            already configured with model and horizon.
        """
        deterministic_mode = self._get_scalar_config_value("deterministic", data_name)
        seed = self._get_scalar_config_value("seed", data_name)
        if deterministic_mode == "full":
            fix_all_random_seed(seed)
        elif deterministic_mode == "efficient":
            fix_random_seed(seed)

        loader_cfg = self._get_scalar_config_value("data_loader_config", data_name)
        loader = DataLoaderFactory.from_config(loader_cfg)
        (
            raw_target,
            raw_past_cov,
            raw_future_cov,
            raw_static_cov,
        ) = loader.reader.load_data()

        ts_data = loader.handle_duplicate_index(raw_target)

        covariates: Dict[str, pd.DataFrame] = {}
        if raw_past_cov is not None:
            covariates["past_covariates"] = raw_past_cov
        if raw_future_cov is not None:
            covariates["future_covariates"] = raw_future_cov
        if raw_static_cov is not None:
            covariates["static_covariates"] = raw_static_cov

        meta_info = pd.Series({"length": len(ts_data)})

        try:
            single_data_results = self._execute(
                ts_data,
                meta_info,
                approach,
                data_name,
                covariates=covariates,
                loader=loader,
            )
        except Exception as e:
            log = f"{traceback.format_exc()}\n{e}"
            single_data_results = self.get_default_result(**{FieldNames.LOG_INFO: log})
        return single_data_results

    @abc.abstractmethod
    # pylint: disable=too-many-arguments
    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        loader: Optional[Any] = None,
    ) -> Any:
        """
        Subclass override for strategy logic.

        :param ts_data: The time series data (DataFrame or numpy).
        :param meta_info: Metadata like length.
        :param approach: The ForecastingApproach instance to use for fit & forecast.
        :param data_name: Identifier string.
        """

    def _get_eval_scaler(
        self,
        loader: Any,
        train_valid_data: pd.DataFrame,
        train_ratio_in_tv: float,
    ) -> Optional[StandardScaler]:
        """
        Return a fitted scaler from the loader if available.
        Fallback to StandardScaler().fit() on entire
        target dataframe if loader does not have scalers.
        """
        if hasattr(loader, "target_scalers") and loader.target_scalers:
            return loader.target_scalers

        # Fallback: fit a scaler on the training portion of the target dataframe
        train_data, _ = split_time(
            train_valid_data, int(len(train_valid_data) * train_ratio_in_tv)
        )
        scaler = StandardScaler().fit(train_data)
        return scaler
