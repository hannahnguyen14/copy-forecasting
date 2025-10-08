import logging
from typing import Any, Dict, Literal, Optional, Tuple, Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from darts import TimeSeries

    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    TimeSeries = None

from tsfb.base.data_layer.read_data import DataReader

logger = logging.getLogger(__name__)

ReturnType = Literal["pandas", "darts"]
NormType = Literal["zscore", "minmax"]

SCALER_MAP = {
    "zscore": StandardScaler,
    "minmax": MinMaxScaler,
}


class BaseTimeSeriesDataLoader:
    """
    Base class for loading and processing time series data.

    This class handles loading time series data with
    support for both pandas and darts formats.
    It provides functionality for data normalization,
    splitting, and handling different types
    of time series data including target series,
    past/future covariates, and static features.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
            for data loading and processing
        spark (Any): Optional Spark session for distributed data processing
        reader (DataReader): Data reader instance for loading raw data
        past_cols (List[str]): List of past covariate column names
        future_cols (List[str]): List of future covariate column names
        static_cols (List[str]): List of static feature column names
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.reader = DataReader(config=config)

        self.past_cols = self.reader.past_cols
        self.future_cols = self.reader.future_cols
        self.static_cols = self.reader.static_cols

        # Load raw data
        target_df, past_df, future_df, static_df = self.reader.load_data()

        freq = self.config.get("data_freq", "H")
        target_df = self._check_and_fix_duplicate_index(
            self._ensure_datetime_index(target_df, freq)
        )
        if isinstance(past_df, pd.DataFrame):
            past_df = self._check_and_fix_duplicate_index(
                self._ensure_datetime_index(past_df, freq)
            )
        if isinstance(future_df, pd.DataFrame):
            future_df = self._check_and_fix_duplicate_index(
                self._ensure_datetime_index(future_df, freq)
            )

        ratios = self.config.get("split_ratio", {"train": 0.7, "val": 0.1, "test": 0.2})
        self.train_data = {
            "target": self._split(target_df, ratios)["train"],
            "past": self._split(past_df, ratios)["train"]
            if isinstance(past_df, pd.DataFrame)
            else None,
            "future": self._split(future_df, ratios)["train"]
            if isinstance(future_df, pd.DataFrame)
            else None,
            "static": static_df if isinstance(static_df, pd.DataFrame) else None,
        }

        # Fit scaler nếu normalize được bật
        if self.config.get("normalize"):
            self._fit_scalers_with_train_data()
        else:
            self.target_scalers = None
            self.past_scalers = None
            self.future_scalers = None
            self.static_scalers = None

    def _fit_scalers_with_train_data(self):
        def create_and_fit(df):
            if df is not None and not df.empty:
                scaler = self._create_scaler()
                scaler.fit(df)
                return scaler
            return None

        self.target_scalers = create_and_fit(self.train_data["target"])
        self.past_scalers = create_and_fit(self.train_data["past"])
        self.future_scalers = create_and_fit(self.train_data["future"])
        self.static_scalers = create_and_fit(self.train_data["static"])

    def _check_and_fix_duplicate_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.index.duplicated().any():
            dups = df.index[df.index.duplicated()].unique()
            logger.warning("Duplicate timestamps found: %s", dups.tolist())
            return df.groupby(df.index).mean()
        return df

    def _ensure_datetime_index(
        self, df: pd.DataFrame, freq: Optional[str] = None
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if freq:
            df = df.asfreq(freq)
        elif df.index.inferred_freq is not None:
            df = df.asfreq(df.index.inferred_freq)
        else:
            inferred = pd.infer_freq(df.index)
            if inferred:
                df = df.asfreq(inferred)
        return df

    def _split(
        self, df: pd.DataFrame, ratios: Dict[str, float]
    ) -> Dict[str, pd.DataFrame]:
        n = len(df)
        tr_end = int(n * ratios.get("train", 0.7))
        val_end = tr_end + int(n * ratios.get("val", 0.1))
        return {
            "train": df.iloc[:tr_end],
            "val": df.iloc[tr_end:val_end],
            "test": df.iloc[val_end:],
        }

    def _create_scaler(self) -> Any:
        method = self.config.get("normalize", {}).get("method", "zscore").lower()
        scaler_cls = SCALER_MAP.get(method, StandardScaler)
        return scaler_cls()

    def _apply_scaler(self, df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
        if scaler is None or df.empty:
            return df
        return pd.DataFrame(
            scaler.transform(df),
            columns=df.columns,
            # index=df.index,
        )

    def _to_darts(self, df: pd.DataFrame) -> TimeSeries:
        if not DARTS_AVAILABLE:
            raise RuntimeError("Darts library is not installed.")
        return TimeSeries.from_dataframe(df, value_cols=list(df.columns))

    def _convert_all_to_darts(
        self, splits: Dict[str, pd.DataFrame]
    ) -> Dict[str, TimeSeries]:
        return {k: self._to_darts(v) for k, v in splits.items()}

    def load(
        self,
    ) -> Tuple[
        Dict[str, Union[pd.DataFrame, TimeSeries]],
        Dict[str, Dict[str, Union[pd.DataFrame, TimeSeries]]],
        Optional[Union[pd.DataFrame, TimeSeries]],
    ]:
        """
        Load and process the time series data.

        Returns:
            Tuple containing:
            - Dict mapping split names to target data (as DataFrame or TimeSeries)
            - Dict of dicts mapping split names to covariate data
            - Optional static features data (as DataFrame or TimeSeries)

        The format of returned data (pandas DataFrame or darts TimeSeries) depends
        on the configuration settings.
        """
        # Reload raw data
        target_df, past_df, future_df, static_df = self.reader.load_data()
        freq = self.config.get("data_freq", "H")

        target_df = self._check_and_fix_duplicate_index(
            self._ensure_datetime_index(target_df, freq)
        )
        if isinstance(past_df, pd.DataFrame):
            past_df = self._check_and_fix_duplicate_index(
                self._ensure_datetime_index(past_df, freq)
            )
        if isinstance(future_df, pd.DataFrame):
            future_df = self._check_and_fix_duplicate_index(
                self._ensure_datetime_index(future_df, freq)
            )

        ratios = self.config.get("split_ratio", {"train": 0.7, "val": 0.1, "test": 0.2})

        splits_tgt = self._split(target_df, ratios)
        splits_past = {k: pd.DataFrame() for k in ("train", "val", "test")}
        splits_future = {k: pd.DataFrame() for k in ("train", "val", "test")}

        if isinstance(past_df, pd.DataFrame):
            splits_past.update(self._split(past_df, ratios))
        if isinstance(future_df, pd.DataFrame):
            for part in ("train", "val", "test"):
                splits_future[part] = future_df.copy()

        if self.config.get("normalize"):
            for part in ("train", "val", "test"):
                splits_tgt[part] = self._apply_scaler(
                    splits_tgt[part], self.target_scalers
                )
                splits_past[part] = self._apply_scaler(
                    splits_past[part], self.past_scalers
                )
                splits_future[part] = self._apply_scaler(
                    splits_future[part], self.future_scalers
                )

            if isinstance(static_df, pd.DataFrame):
                static_df = self._apply_scaler(static_df, self.static_scalers)

        if self.config.get("return_type") == "darts":
            splits_tgt = self._convert_all_to_darts(splits_tgt)
            splits_past = self._convert_all_to_darts(splits_past)
            splits_future = self._convert_all_to_darts(splits_future)
            if isinstance(static_df, pd.DataFrame):
                static_df = self._to_darts(static_df)

        splits_cov = {
            "past_covariates": splits_past,
            "future_covariates": splits_future,
        }

        return splits_tgt, splits_cov, static_df

    def handle_duplicate_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate indices in DataFrame by averaging duplicate values.

        Args:
            df: Input DataFrame to process

        Returns:
            DataFrame with duplicate indices resolved by averaging
        """
        return self._check_and_fix_duplicate_index(df)
