from source.preprocessor import Preprocessor
from source.utils import (
    CATEGORICAL_MULTI,
    FixCategoricalStrategy,
)


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
