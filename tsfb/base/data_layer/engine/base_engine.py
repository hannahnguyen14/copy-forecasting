from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from tsfb.base.data_layer.source import CsvSource, ParquetSource, SqlAlchemySource

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Engine(ABC):
    """Abstract base class for data loading engines."""

    name: str

    @abstractmethod
    def load_csv(self, src: CsvSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load data from a CSV source."""

    @abstractmethod
    def load_parquet(self, src: ParquetSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load data from a Parquet source."""

    @abstractmethod
    def load_sqlalchemy(
        self, src: SqlAlchemySource, opts: Dict[str, Any]
    ) -> pd.DataFrame:
        """Load data from a SQLAlchemy source."""
