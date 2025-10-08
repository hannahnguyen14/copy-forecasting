from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd
from sqlalchemy import create_engine

from tsfb.base.data_layer.engine.base_engine import Engine
from tsfb.base.data_layer.source import CsvSource, ParquetSource, SqlAlchemySource

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PandasEngine(Engine):
    """Engine for loading data using pandas."""

    name = "pandas"

    def load_csv(self, src: CsvSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load CSV file to DataFrame."""
        try:
            logger.info("Loading CSV from %s", src.file_path)
            parse_dates = (
                [src.index_col]
                if src.index_col and src.parse_dates is True
                else src.parse_dates
            )
            df = pd.read_csv(
                src.file_path,
                index_col=src.index_col,
                parse_dates=parse_dates,
                **src.read_csv_kwargs,
            )
            if src.index_col and pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
            return df
        except FileNotFoundError as e:
            logger.error("CSV file not found: %s", e)
            raise
        except Exception as e:
            logger.exception("Error loading CSV from %s: %s", src.file_path, e)
            raise

    def load_parquet(self, src: ParquetSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load Parquet file to DataFrame."""
        try:
            logger.info("Loading Parquet from %s", src.file_path)
            df = pd.read_parquet(
                src.file_path, columns=src.columns, **src.read_parquet_kwargs
            )
            if src.index_col and src.index_col in df.columns:
                df = df.set_index(src.index_col)
            if src.index_col and not pd.api.types.is_datetime64_any_dtype(df.index):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    logger.warning(
                        "Could not parse index %s to datetime", src.index_col
                    )
            if src.index_col and pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
            return df
        except FileNotFoundError as e:
            logger.error("Parquet file not found: %s", e)
            raise
        except Exception as e:
            logger.exception("Error loading Parquet from %s: %s", src.file_path, e)
            raise

    def load_sqlalchemy(
        self, src: SqlAlchemySource, opts: Dict[str, Any]
    ) -> pd.DataFrame:
        """Load SQLAlchemy source to DataFrame."""
        try:
            logger.info("Loading SQLAlchemy source: %s", src.sqlalchemy_url)
            parse_dates = (
                [src.index_col]
                if src.index_col and src.parse_dates is True
                else src.parse_dates
            )
            engine = create_engine(src.sqlalchemy_url)
            if src.query:
                df = pd.read_sql(
                    src.query,
                    con=engine,
                    parse_dates=parse_dates,
                    **src.read_sql_kwargs,
                )
            else:
                df = pd.read_sql_table(  # type: ignore[arg-type]
                    src.table,
                    con=engine,
                    parse_dates=parse_dates,
                    **src.read_sql_kwargs,
                )
            if src.index_col and src.index_col in df.columns:
                df = df.set_index(src.index_col)
            if src.index_col and pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
            return df
        except Exception as e:
            logger.exception(
                "Error loading data from SQLAlchemy source (%s): %s",
                src.sqlalchemy_url,
                e,
            )
            raise
