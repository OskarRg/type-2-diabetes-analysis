import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

from source.utils import CATEGORICAL_BINARY, CATEGORICAL_MULTI, TARGET, AnomalyDetectionStrategy

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Class responsible for detecting and removing anomalies (outliers) from the dataset.
    Supports IQR (Interquartile Range) and Isolation Forest methods.
    """

    def __init__(
        self, strategy: str = AnomalyDetectionStrategy.IQR, contamination: float = 0.05
    ) -> None:
        """
        Initialize the detector.

        :param strategy: Strategy to use: 'iqr' for statistical thresholding,
                         'isolation_forest' for ML-based anomaly detection.
        :param contamination: Used only for 'isolation_forest'. The proportion of outliers
                              in the data set (e.g., 0.05 = 5%).
        """
        self.strategy: str = strategy
        self.contamination: float = contamination

    # TODO Currently it does not work well because columns that were
    # One Hot Encoded are still taken into consideration -- fix someday
    def get_viable_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Get columns that might have anomalies.

        :params df: DataFrame containing anomalies.
        :return: Cleaned DataFrame.
        """
        all_columns: set[str] = set(df.columns)
        excluded_columns: set[str] = set(CATEGORICAL_BINARY + CATEGORICAL_MULTI + [TARGET])
        numerical_cols: list[str] = list(all_columns - excluded_columns)

        return numerical_cols

    # TODO The plot is ugly atm.
    def visualize_boxplots(self, df: pd.DataFrame) -> None:
        """
        Visualizes distribution and outliers using boxplots before removal.

        :param df: Input DataFrame.
        :param columns: list of numerical columns to visualize.
        """
        numerical_cols: list[str] = self.get_viable_columns(df=df)
        n_cols: int = 3
        n_rows: int = (len(numerical_cols) - 1) // n_cols + 1

        plt.figure(figsize=(15, 4 * n_rows))
        for i, col in enumerate(numerical_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.boxplot(x=df[col], color="skyblue")
            plt.title(f"Distribution of {col}")

        plt.tight_layout()
        plt.show()

    def _detect_iqr(self, df: pd.DataFrame, columns: list[str]) -> pd.Index:
        """
        Detects anomalies using the Interquartile Range (IQR) method.
        Returns indices of rows to drop.
        """
        outlier_indices: list[pd.Index] = []

        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound: float = Q1 - 1.5 * IQR
            upper_bound: float = Q3 + 1.5 * IQR

            col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outlier_indices.extend(col_outliers)

        return pd.Index(list(set(outlier_indices)))

    def _detect_isolation_forest(self, df: pd.DataFrame, columns: list[str]) -> pd.Index:
        """
        Detects anomalies using Isolation Forest algorithm.
        Returns indices of rows to drop.
        """
        iso: IsolationForest = IsolationForest(contamination=self.contamination, random_state=42)
        # Isolation Forest returns -1 for outliers and 1 for inliers
        preds: np.ndarray = iso.fit_predict(df[columns])

        outlier_indices = df.index[preds == -1]
        return outlier_indices

    def remove_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes anomalies based on the selected strategy.

        :param df: Input DataFrame.
        :param columns_to_check: list of numerical columns to check for outliers.
        :return: Cleaned DataFrame with anomalies removed.
        """
        numerical_cols: list[str] = self.get_viable_columns(df=df)

        original_size: int = len(df)
        logger.debug(f"Original dataset size: {original_size}")

        drop_indices: pd.DataFrame
        match self.strategy:
            case AnomalyDetectionStrategy.IQR:
                drop_indices = self._detect_iqr(df, numerical_cols)
            case AnomalyDetectionStrategy.ISOLATION_FOREST:
                drop_indices = self._detect_isolation_forest(df, numerical_cols)
            case _:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        df_clean: pd.DataFrame = df.drop(index=drop_indices)

        removed_count: int = len(drop_indices)
        logger.debug(f"Strategy: {self.strategy.upper()}")
        logger.debug(f"Removed {removed_count} anomalies ({removed_count / original_size:.2%}).")
        logger.debug(f"New dataset size: {len(df_clean)}")

        return df_clean
