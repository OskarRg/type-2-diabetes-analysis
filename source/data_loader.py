import pandas as pd


class DataLoader:
    """Handles loading and basic inspection of the dataset."""

    def __init__(self):
        pass

    def load_data(self, path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        raise NotImplementedError

    def show_info(self):
        """Display basic dataset information such as shape, types, and missing values."""
        raise NotImplementedError

    def handle_missing(self, strategy: str = "mean"):
        """Handle missing values using the specified strategy (mean, median, drop, etc.)."""
        raise NotImplementedError
