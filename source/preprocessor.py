import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from source.utils import (
    CATEGORICAL_BINARY,
    CATEGORICAL_DATA_VALID_RANGES,
    CATEGORICAL_MULTI,
    TARGET,
    FixCategoricalStrategy,
)

logger: logging.Logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Handles preprocessing tasks such as cleaning, encoding, normalization,
    and splitting the dataset.

    The class provides reusable transformation utilities that prepare
    the data for machine learning models.
    """

    def __init__(self) -> None:
        """Initialize the preprocessor with optional scaler."""
        self.scaler = StandardScaler()  # Scaler class could be an argument passed from config
        self.numeric_columns: list[str] = []
        self.encoded_columns: list[str] = []

    def fix_categorical_values(
        self, df: pd.DataFrame, strategy: str = FixCategoricalStrategy.CLIP
    ) -> pd.DataFrame:
        """
        Fix incorrect categorical entries in columns "smoking" and "drinking" with a given strategy.
        Columns "smoking" and "drinking" were headlined because of the previous EDA results.

        :param df: Input dataframe.
        :param strategy: Strategy for handling invalid values. Options:
                     - "clip": Clip values to valid min/max.
                     - "drop": Remove rows containing invalid values.
        :returns: Corrected dataframe.
        """
        logger.debug("Fixing incorrect categorical values...")

        df: pd.DataFrame = df.copy()

        for col, (min_val, max_val) in CATEGORICAL_DATA_VALID_RANGES.items():
            if col not in df.columns:
                continue

            invalid_mask: pd.DataFrame = (df[col] < min_val) | (df[col] > max_val)
            invalid_rows: pd.DataFrame = df[invalid_mask]

            if invalid_rows.empty:
                continue

            logger.debug(
                f"Column '{col}' has {invalid_rows.shape[0]} invalid values "
                f"(valid range: {min_val}-{max_val})."
            )
            match strategy:
                case FixCategoricalStrategy.CLIP:
                    df[col] = df[col].clip(lower=min_val, upper=max_val).round().astype(int)
                case FixCategoricalStrategy.DROP:
                    df = df[~invalid_mask]
                    logger.debug(
                        f"Dropped {invalid_rows.shape[0]} rows due to invalid '{col}' values."
                    )
                case _:
                    raise ValueError("Invalid strategy. Use FixCategoricalStrategy.CLIP or DROP.")

        logger.debug("Categorical value correction complete.")
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns using One-Hot Encoding (for multi-class).

        :param df: Input dataframe.
        :returns: DataFrame with encoded categorical features.
        """
        logger.debug("Encoding categorical variables...")

        df: pd.DataFrame = df.copy()

        for col in CATEGORICAL_MULTI:
            if col in df.columns:
                dummies: pd.DataFrame = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self.encoded_columns.extend(dummies.columns.tolist())
                logger.debug(f"One-Hot encoded column: {col}")

        logger.debug("Categorical encoding complete.")
        return df

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize numerical features using StandardScaler.
        Does not scale binary or encoded categorical columns.

        :param df: Input dataframe after encoding.
        :returns: DataFrame with scaled numerical features.
        """
        logger.debug("Normalizing numerical features...")

        df: pd.DataFrame = df.copy()
        exclude_cols: list[str] = CATEGORICAL_BINARY + self.encoded_columns + [TARGET]

        self.numeric_columns = [
            col for col in df.columns if df[col].dtype != "object" and col not in exclude_cols
        ]

        logger.debug(f"Numeric columns to scale: {self.numeric_columns}")

        df[self.numeric_columns] = self.scaler.fit_transform(df[self.numeric_columns])

        logger.debug("Normalization complete.")
        return df

    def split_data(
        self, df: pd.DataFrame, target_col: str, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into train and test sets.

        :param df: Full dataset after preprocessing.
        :param target_col: Name of the target column.
        :param test_size: Fraction of data used for testing.
        :returns: X_train, X_test, y_train, y_test
        """
        logger.debug(f"Splitting dataset with target column '{target_col}'...")

        X: pd.DataFrame = df.drop(columns=[target_col])
        y: pd.DataFrame = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.debug(f"Split complete: train size = {X_train.shape}, test size = {X_test.shape}")

        return X_train, X_test, y_train, y_test
