class ModelEvaluator:
    """Evaluates and compares model performance."""

    def __init__(self):
        pass

    def evaluate(self, model, X_test, y_test):
        """Evaluate a model and return common performance metrics."""
        raise NotImplementedError

    def plot_confusion_matrix(self):
        """Plot the confusion matrix for a given model."""
        raise NotImplementedError

    def compare_models(self, metrics_dict):
        """Compare multiple models based on performance metrics."""
        raise NotImplementedError
