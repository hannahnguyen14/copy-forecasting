from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import pandas as pd

from tsfb.base.data_layer.engine import DuckDBEngine, Engine, PandasEngine, SparkEngine
from tsfb.base.data_layer.source import (
    CsvSource,
    DataSource,
    ParquetSource,
    SqlAlchemySource,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
else:
    SparkDataFrame = object

DataFrameType = Union[pd.DataFrame, "SparkDataFrame"]


class DataReader:
    """
    Data loading and splitting utility for time series forecasting.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config

        # target & covariates
        self.target_columns = config["target_columns"]
        cov = config.get("covariate", {}) or {}
        self.past_cols = cov.get("past", [])
        self.future_cols = cov.get("future", [])
        self.static_cols = cov.get("static", [])

        self.source = self._init_source(config)

        self.engine = self._init_engine(config)

    def _init_source(self, cfg: Dict[str, Any]) -> DataSource:
        """Create a DataSource from config."""
        kind = cfg.get("kind", "csv")
        if kind == "csv":
            return CsvSource(
                file_path=cfg["file_path"],
                index_col=cfg.get("index_col"),
                parse_dates=cfg.get("parse_dates", True),
                read_csv_kwargs=cfg.get("read_csv_kwargs", {}),
            )
        if kind == "parquet":
            return ParquetSource(
                file_path=cfg["file_path"],
                index_col=cfg.get("index_col"),
                parse_dates=cfg.get("parse_dates", False),
                columns=cfg.get("columns"),
                read_parquet_kwargs=cfg.get("read_parquet_kwargs", {}),
            )
        if kind == "sqlalchemy":
            return SqlAlchemySource(
                sqlalchemy_url=cfg["sqlalchemy_url"],
                table=cfg.get("table"),
                query=cfg.get("query"),
                index_col=cfg.get("index_col"),
                parse_dates=cfg.get("parse_dates", True),
                read_sql_kwargs=cfg.get("read_sql_kwargs", {}),
            )
        raise ValueError(f"Unsupported source kind: {kind}")

    def _init_engine(self, cfg: Dict[str, Any]) -> Engine:
        """Create an Engine from config."""
        backend = cfg.get("backend", cfg.get("engine", "pandas"))
        if backend == "pandas":
            return PandasEngine()
        if backend == "spark":
            if SparkDataFrame is None:
                raise RuntimeError(
                    "Spark is not available but 'spark' backend requested"
                )
            return SparkEngine()
        if backend == "duckdb":
            return DuckDBEngine()
        raise ValueError(f"Unsupported engine backend: {backend}")

    def load_data(
        self, opts: Optional[Dict[str, Any]] = None
    ) -> Tuple[
        DataFrameType,
        Optional[DataFrameType],
        Optional[DataFrameType],
        Optional[DataFrameType],
    ]:
        """Load and split data into target, past, future, static."""
        opts = opts or {}
        df = self._dispatch_load(self.source, opts)

        tgt_df = self._select_or_none(df, self.target_columns)
        past_df = self._select_or_none(df, self.past_cols)
        future_df = self._select_or_none(df, self.future_cols)
        static_df = self._select_or_none(df, self.static_cols)

        return tgt_df, past_df, future_df, static_df

    def _dispatch_load(self, source: DataSource, opts: Dict[str, Any]) -> DataFrameType:
        """Load data using the correct engine method."""
        kind = source.kind()
        if kind == "csv":
            assert isinstance(
                source, CsvSource
            ), "Source must be CsvSource for csv kind"
            return self.engine.load_csv(source, opts)
        if kind == "parquet":
            assert isinstance(
                source, ParquetSource
            ), "Source must be ParquetSource for parquet kind"
            return self.engine.load_parquet(source, opts)
        if kind == "sqlalchemy":
            assert isinstance(
                source, SqlAlchemySource
            ), "Source must be SqlAlchemySource for sqlalchemy kind"
            return self.engine.load_sqlalchemy(source, opts)
        raise ValueError(f"Unsupported source kind: {kind}")

    @staticmethod
    def _select_or_none(df: DataFrameType, cols: list[str]) -> Optional[DataFrameType]:
        """Select columns or return None if empty."""
        if not cols:
            return None
        if isinstance(df, pd.DataFrame):
            return df[cols]
        return df.select(*cols)
