import pandas as pd
import pytest

from tsfb.base.data_layer import dataloader
from tsfb.base.data_layer.dataloader import BaseTimeSeriesDataLoader


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


@pytest.fixture
def patched_loader(monkeypatch):
    # Patch DataReader trong module dataloader trước khi tạo loader
    monkeypatch.setattr(dataloader, "DataReader", DummyReader)

    config = {
        "file_path": "dummy.csv",
        "target_columns": ["target"],
        "normalize": {"method": "zscore"},
    }
    loader = BaseTimeSeriesDataLoader(config)
    return loader


def test_base_loader_load(monkeypatch):
    monkeypatch.setattr(dataloader, "DataReader", DummyReader)

    config = {"file_path": "dummy.csv", "target_columns": ["target"]}
    loader = BaseTimeSeriesDataLoader(config)
    tgt, cov, stat = loader.load()
    assert "train" in tgt and "val" in tgt and "test" in tgt
    assert "past_covariates" in cov and "future_covariates" in cov


def test_normalize_and_apply(patched_loader):
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    scaler = patched_loader._create_scaler()
    scaler.fit(df)

    normed = patched_loader._apply_scaler(df, scaler)
    applied = patched_loader._apply_scaler(df, scaler)

    assert normed.shape == df.shape
    assert applied.shape == df.shape


def test_check_and_fix_duplicate_index(monkeypatch):
    monkeypatch.setattr(dataloader, "DataReader", DummyReader)

    config = {"file_path": "dummy.csv", "target_columns": ["target"]}
    loader = BaseTimeSeriesDataLoader(config)

    df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=[0, 1, 1, 2])
    fixed = loader._check_and_fix_duplicate_index(df)

    assert not fixed.index.duplicated().any()
