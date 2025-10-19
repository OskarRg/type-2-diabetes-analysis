from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.feature_selector import FeatureSelector
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.visualizer import Visualizer

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
