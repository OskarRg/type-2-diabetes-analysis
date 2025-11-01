from source.data_loader import DataLoader
from source.preprocessor import Preprocessor
from source.feature_selector import FeatureSelector
from source.model_trainer import ModelTrainer
from source.model_evaluator import ModelEvaluator
from source.visualizer import Visualizer


class Pipeline:
    """Main class orchestrating the entire machine learning workflow."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.visualizer = Visualizer()

    def run(self):
        """Execute the complete data analysis and modeling pipeline."""
        raise NotImplementedError
