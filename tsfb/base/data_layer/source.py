from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


class DataSource(ABC):
    """
    Abstract base class for data source metadata.
    Engine will use this metadata to read data from various sources.
    """

    @abstractmethod
    def kind(self) -> str:
        """
        Returns the kind of data source as a string.
        """
        raise NotImplementedError


@dataclass
class CsvSource(DataSource):
    """
    Data source for reading CSV files.

    Attributes:
        file_path (str): Path to the CSV file.
        index_col (Optional[str]): Column to use as the row labels of the DataFrame.
        parse_dates (Union[bool, List[str]]): Whether to parse dates,
            or list of columns to parse as dates.
        read_csv_kwargs (Dict[str, Any]): Additional
            keyword arguments for pandas.read_csv.
    """

    file_path: str
    index_col: Optional[str] = None
    parse_dates: Union[bool, List[str]] = True
    read_csv_kwargs: Dict[str, Any] = field(default_factory=dict)

    def kind(self) -> str:
        """
        Returns the kind of data source: 'csv'.
        """
        return "csv"


@dataclass
class ParquetSource(DataSource):
    """
    Data source for reading Parquet files.

    Attributes:
        file_path (str): Path to the Parquet file.
        index_col (Optional[str]): Column to use as the row labels of the DataFrame.
        parse_dates (Union[bool, List[str]]): Whether to parse dates,
            or list of columns to parse as dates.
        columns (Optional[List[str]]): List of columns to read from the Parquet file.
        read_parquet_kwargs (Dict[str, Any]): Additional keyword
            arguments for pandas.read_parquet.
    """

    file_path: str
    index_col: Optional[str] = None
    parse_dates: Union[bool, List[str]] = False
    columns: Optional[List[str]] = None
    read_parquet_kwargs: Dict[str, Any] = field(default_factory=dict)

    def kind(self) -> str:
        """
        Returns the kind of data source: 'parquet'.
        """
        return "parquet"


@dataclass
class SqlAlchemySource(DataSource):
    """
    Read from database by SQLAlchemy URL.
    Such as:
      - postgresql+psycopg2://user:pwd@host:5432/db
      - mysql+pymysql://user:pwd@host:3306/db
      - mssql+pyodbc://user:pwd@dsn
      - sqlite:///path/to.db
    """

    sqlalchemy_url: str
    table: Optional[str] = None
    query: Optional[str] = None
    index_col: Optional[str] = None
    parse_dates: Union[bool, List[str]] = True
    read_sql_kwargs: Dict[str, Any] = field(default_factory=dict)

    def kind(self) -> str:
        """
        Returns the kind of data source: 'sqlalchemy'.
        """
        return "sqlalchemy"
