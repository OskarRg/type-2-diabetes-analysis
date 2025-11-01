import pandas as pd


class Preprocessor:
    """Prepares and transforms data for modeling."""

    def __init__(self):
        pass

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features into numerical format."""
        raise NotImplementedError

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize or standardize numerical features."""
        raise NotImplementedError

    def split_data(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2):
        """Split the dataset into training and testing sets."""
        raise NotImplementedError
