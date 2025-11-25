import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Simple clean dataset (no missing values, valid categorical)."""
    return pd.DataFrame(
        {
            "Age": [30, 50, 40],
            "Gender": [1, 2, 1],
            "BMI": [22.5, 30.1, 27.3],
            "SBP": [120, 140, 130],
            "DBP": [80, 90, 85],
            "FPG": [95, 110, 100],
            "smoking": [1, 2, 3],
            "drinking": [1, 3, 2],
            "family_histroy": [0, 1, 1],  # intentionally misspelled as in the dataset
            "Diabetes": [0, 1, 0],
        }
    )


@pytest.fixture
def df_with_invalid() -> pd.DataFrame:
    """DataFrame where smoking and drinking contain invalid values."""
    return pd.DataFrame(
        {
            "Age": [30, 50, 40],
            "Gender": [1, 2, 1],
            "BMI": [22.5, 30.1, 27.3],
            "SBP": [120, 140, 130],
            "DBP": [80, 90, 85],
            "FPG": [95, 110, 100],
            "smoking": [1, 4.20, -2],  # invalid value not in {1, 2, 3}
            "drinking": [1, 10, 0],  # invalid value not in {1, 2, 3}
            "family_histroy": [0, 1, 1],
            "Diabetes": [0, 1, 0],
        }
    )


@pytest.fixture
def df_for_split() -> pd.DataFrame:
    """Example DataFrame to test splitting"""
    return pd.DataFrame(
        {
            "Age": [25, 30, 45, 50],
            "Gender": [1, 2, 1, 2],
            "BMI": [23.5, 28.1, 30.2, 26.4],
            "SBP": [120, 130, 140, 135],
            "DBP": [80, 85, 90, 88],
            "FPG": [95, 110, 99, 105],
            "family_history": [0, 1, 0, 1],
            "smoking_1": [1, 0, 1, 0],
            "smoking_2": [0, 1, 0, 1],
            "drinking_1": [1, 0, 1, 0],
            "drinking_2": [0, 1, 0, 1],
            "Diabetes": [0, 1, 0, 1],
        }
    )
