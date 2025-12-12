import pandas as pd
from sklearn.base import ClassifierMixin


class FeatureSelector:
    """
    Selects the most relevant features based on statistical correlation or model-based importance.
    """

    def correlation_analysis(
        self,
        df: pd.DataFrame,
        target: str,
        method: str = "pearson",
    ) -> pd.Series:
        """
        Perform correlation analysis between each feature and the target.

        :param df: Input dataframe.
        :param target: Name of the target column.
        :param method: Correlation method: 'pearson', 'spearman', 'kendall'.
        :return: Sorted Series of correlations (absolute value).
        """
        # TODO Consider deleting features that have a strong corelation between themselves - config.
        corr: pd.Series = df.corr(method=method)[target].drop(target)
        corr_sorted: pd.Series = corr.abs().sort_values(ascending=False)

        return corr_sorted

    def select_features_by_importance(
        self,
        model: ClassifierMixin,
        feature_names: list[str],
        top_n: int = 10,
    ) -> list[str]:
        """
        Select most important features based on model.feature_importances_.

        :param model: Trained model with `feature_importances_` attribute.
        :param feature_names: List of available feature names.
        :param top_n: Number of features to select.
        :return: List of selected feature names.
        """
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not provide `feature_importances_`.")

        importances: pd.Series = pd.Series(model.feature_importances_, index=feature_names)
        sorted_importances: pd.Series = importances.sort_values(ascending=False)
        return sorted_importances.head(top_n).index.tolist()
