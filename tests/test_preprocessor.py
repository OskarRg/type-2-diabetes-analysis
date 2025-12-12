from source.preprocessor import Preprocessor
from source.utils import (
    CATEGORICAL_MULTI,
    FixCategoricalStrategy,
)
import pandas as pd


def test_fix_categorical_clip(df_with_invalid):
    pp = Preprocessor()
    df = pp.fix_categorical_values(df_with_invalid.copy(), strategy=FixCategoricalStrategy.CLIP)

    # smoking+drinking valid now
    assert df["smoking"].between(1, 3).all()
    assert df["drinking"].between(1, 3).all()


def test_fix_categorical_drop(df_with_invalid):
    pp = Preprocessor()
    df = pp.fix_categorical_values(df_with_invalid.copy(), strategy=FixCategoricalStrategy.DROP)

    # only row 0 is valid
    assert df.shape[0] == 1
    assert df["smoking"].iloc[0] == 1
    assert df["drinking"].iloc[0] == 1


def test_one_hot_encoding(sample_df):
    pp = Preprocessor()
    df = pp.encode_categorical(sample_df.copy())

    for col in CATEGORICAL_MULTI:
        for i in [1, 2, 3]:
            assert f"{col}_{i}" in df.columns


def test_binary_not_one_hot(sample_df):
    pp = Preprocessor()
    df = pp.encode_categorical(sample_df.copy())

    for col in ["family_histroy", "Gender"]:
        assert col in df.columns


def test_normalization_excludes_one_hot(sample_df):
    pp = Preprocessor()
    df = pp.encode_categorical(sample_df.copy())
    df = pp.normalize_features(df)

    assert df["Age"].mean() < 1
    assert df["BMI"].mean() < 1

    oh = [col for col in df.columns if "smoking_" in col or "drinking_" in col]
    assert df[oh].isin([0, 1]).all().all()


def test_split_data(df_for_split):
    pp = Preprocessor()
    df = pp.encode_categorical(df_for_split.copy())

    X_train, X_test, y_train, y_test = pp.split_data(df, target_col="Diabetes", test_size=0.5)

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_remove_outliers_skips_categorical():
    """
    Test if the method skips columns with <= 3 unique values,
    even if they mathematically contain 'outliers'.
    """
    # Column has only 3 unique values: 1, 2, and 100.
    # Mathematically 100 is an outlier compared to 1 and 2.
    # But logic dictates: nunique() <= 3 -> skip.
    data = {"cat_val": [1, 1, 2, 2, 100]}
    df = pd.DataFrame(data)

    pp = Preprocessor()
    df_clean = pp.remove_outliers(df)

    # Expectation: No rows removed because the column is treated as categorical
    assert len(df_clean) == 5
    assert 100 in df_clean["cat_val"].values


def test_remove_outliers_specific_columns():
    """
    Test if the method only processes columns specified in the argument.
    """
    df = pd.DataFrame(
        {
            "col_A": [10, 10, 10, 10, 1000],  # Outlier: 1000
            "col_B": [5, 5, 5, 5, 500],  # Outlier: 500
        }
    )

    pp = Preprocessor()

    # We only ask to clean col_A.
    # The last row has outliers in BOTH, but we want to ensure
    # logic works based on selection.
    # To test this better, let's make the outliers occur in different rows.

    df_mixed = pd.DataFrame(
        {
            "col_A": [10, 11, 12, 13, 1000],  # Outlier at index 3
            "col_B": [5, 5, 500, 5, 5],  # Outlier at index 2
        }
    )

    # Clean only col_A
    df_clean_A = pp.remove_outliers(df_mixed, columns=["col_A"])

    # Index 3 (outlier in A) should be removed.
    # Index 2 (outlier in B) should remain because we ignored B.
    assert 1000 not in df_clean_A["col_A"].values
    assert 500 in df_clean_A["col_B"].values
    assert len(df_clean_A) == 4


def test_remove_outliers_no_outliers():
    """Test that data remains unchanged if there are no outliers."""
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})

    pp = Preprocessor()
    df_clean = pp.remove_outliers(df)

    assert len(df_clean) == 5
    pd.testing.assert_frame_equal(df, df_clean)
