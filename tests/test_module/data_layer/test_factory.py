import pandas as pd
import pytest

from tsfb.base.data_layer import dataloader
from tsfb.base.data_layer.dataloader import BaseTimeSeriesDataLoader
from tsfb.base.data_layer.factory import DataLoaderFactory


class DummyReader:
    def __init__(self, config=None, spark=None):
        self.past_cols = ["past1"]
        self.future_cols = ["future1"]
        self.static_cols = ["static1"]

    def load_data(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "target": range(10),
                "past1": range(10),
                "future1": range(10),
                "static1": range(10),
            },
            index=idx,
        )
        return df[["target"]], df[["past1"]], df[["future1"]], df[["static1"]]


def test_factory_from_config_base(monkeypatch):
    monkeypatch.setattr(dataloader, "DataReader", DummyReader)

    config = {"loader": "base", "file_path": "dummy.csv", "target_columns": ["target"]}
    loader = DataLoaderFactory.from_config(config)

    assert isinstance(loader, BaseTimeSeriesDataLoader)


def test_factory_missing_loader():
    config = {"file_path": "dummy.csv", "target_columns": ["target"]}
    with pytest.raises(KeyError):
        DataLoaderFactory.from_config(config)


def test_factory_unknown_loader():
    config = {
        "loader": "unknown",
        "file_path": "dummy.csv",
        "target_columns": ["target"],
    }
    with pytest.raises(ValueError):
        DataLoaderFactory.from_config(config)


def test_register_loader():
    class DummyLoader(BaseTimeSeriesDataLoader):
        pass

    DataLoaderFactory.LOADERS.pop("dummy", None)
    DataLoaderFactory.register_loader("dummy", DummyLoader)
    assert DataLoaderFactory.LOADERS["dummy"] is DummyLoader

    with pytest.raises(KeyError):
        DataLoaderFactory.register_loader("dummy", DummyLoader)
