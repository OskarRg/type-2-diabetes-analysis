import pandas as pd
import pytest

from source.data_loader import DataLoader
from source.utils import RENAME_MAPPING, MissingHandlingStrategy


def test_read_csv(tmp_path, sample_df):
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)

    loader = DataLoader()
    df = loader.load_data(str(path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == sample_df.shape


def test_read_csv_missing_file():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_data("does_not_exist.csv")


def test_handle_missing_mean():
    df = pd.DataFrame({"A": [1, None, 3]})
    loader = DataLoader()
    loader.set_df(df)

    df2 = loader.handle_missing(MissingHandlingStrategy.MEAN)
    assert df2["A"].isna().sum() == 0
    assert df2["A"].iloc[1] == pytest.approx((1 + 3) / 2)


def test_handle_missing_drop():
    df = pd.DataFrame({"A": [1, None, 3], "B": [2, 2, None]})
    loader = DataLoader()
    loader.set_df(df)

    df2 = loader.handle_missing(MissingHandlingStrategy.DROP)
    assert df2.shape[0] == 1
    assert df2.iloc[0]["A"] == 1


def test_read_csv(tmp_path, sample_df):
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)

    loader = DataLoader()
    df = loader.load_data(str(path))

    assert "family_history" in df.columns
    assert "family_histroy" not in df.columns


def test_load_data_renames_columns(tmp_path):
    sample_df = pd.DataFrame(
        {
            "family_histroy": [1, 0, 1],
            "Age": [50, 60, 40],
        }
    )

    path = tmp_path / "test.csv"
    sample_df.to_csv(path, index=False)
    loader = DataLoader()
    df = loader.load_data(str(path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == sample_df.shape

    for old, new in RENAME_MAPPING.items():
        assert new in df.columns
        assert old not in df.columns
