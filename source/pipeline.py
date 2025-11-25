from typing import TYPE_CHECKING

from source.data_loader import DataLoader
from source.feature_selector import FeatureSelector
from source.model_evaluator import ModelEvaluator
from source.model_trainer import ModelTrainer
from source.preprocessor import Preprocessor
from source.visualizer import Visualizer

if TYPE_CHECKING:
    import pandas as pd


class Pipeline:
    """Main class orchestrating the entire machine learning workflow."""

    def __init__(self) -> None:
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.visualizer = Visualizer()

    def run(self, data_path: str) -> None:
        """Execute the complete data preparation and modeling pipeline."""
        self.data_loader.load_data(data_path)
        self.data_loader.show_info()
        df: pd.DataFrame = self.data_loader.handle_missing()
        df = self.preprocessor.fix_categorical_values(df)
        df = self.preprocessor.encode_categorical(df)  # OHE should be used for SVM and LR
        df = self.preprocessor.normalize_features(df)
