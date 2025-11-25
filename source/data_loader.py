import logging

import pandas as pd

from source.utils import RENAME_MAPPING, MissingHandlingStrategy

logger: logging.Logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading, inspecting, and preprocessing of the dataset.
    """

    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None

    def set_df(self, df: pd.DataFrame) -> None:
        self.df = df

    def load_data(self, path: str) -> pd.DataFrame:
        """
        Load a dataset from a CSV file.

        :param path: Path to the CSV file.
        :returns: Loaded DataFrame.
        :raises FileNotFoundError: If the file does not exist.
        :raises Exception: For unexpected errors during loading.
        """
        try:
            self.df: pd.DataFrame = pd.read_csv(path)
            logger.debug(f"Data loaded successfully. Shape: {self.df.shape}")
            self.df = self.df.rename(columns=RENAME_MAPPING)
            logger.debug(f"Renamed columns: {RENAME_MAPPING}")
            return self.df

        except FileNotFoundError:
            logger.error(f"File not found at path: {path}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error while loading data: {e}")
            raise

    def show_info(self) -> None:
        """
        Display general information about the loaded dataset.

        Includes:
        - shape
        - data types
        - missing values summary
        - first rows preview

        :raises ValueError: If no data has been loaded yet.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info(f"Dataset shape: {self.df.shape}")
        logger.info(f"Dataset dtypes:\n{self.df.dtypes}")
        logger.info(f"Missing values:\n{self.df.isnull().sum()}")
        logger.info(f"Head:\n{self.df.head()}")

    def handle_missing(self, strategy: str = MissingHandlingStrategy.DROP) -> pd.DataFrame:
        """
        Handle missing values using a selected strategy.

        Supported strategies:
        - "mean": fill numeric NaNs with mean
        - "median": fill numeric NaNs with median
        - "drop": drop rows containing missing values

        :param strategy: Missing value handling strategy.
        :returns: DataFrame after missing-value processing.
        :raises ValueError: If the dataset is not loaded or strategy is invalid.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call `load_data()` first.")

        if strategy == MissingHandlingStrategy.MEAN:
            self.df = self.df.fillna(self.df.mean(numeric_only=True))
        elif strategy == MissingHandlingStrategy.MEDIAN:
            self.df = self.df.fillna(self.df.median(numeric_only=True))
        elif strategy == MissingHandlingStrategy.DROP:
            self.df = self.df.dropna()
        else:
            raise ValueError("Invalid strategy. Choose: 'mean', 'median', or 'drop'.")

        logger.debug(f"Missing values handled with strategy: {strategy}")
        return self.df
