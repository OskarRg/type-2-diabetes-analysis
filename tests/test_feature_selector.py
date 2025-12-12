import pandas as pd
import pytest
from source.feature_selector import FeatureSelector


class DummyModel:
    def __init__(self, importances):
        self.feature_importances_ = importances


def test_select_features_by_importance():
    model = DummyModel([0.1, 0.8, 0.3])
    feature_names = ["a", "b", "c"]

    fs = FeatureSelector()
    result = fs.select_features_by_importance(model, feature_names, top_n=2)

    assert result == ["b", "c"]  # 0.8, 0.3


def test_select_features_by_importance_raises():
    class NoImportance:
        pass

    fs = FeatureSelector()

    with pytest.raises(ValueError):
        fs.select_features_by_importance(NoImportance(), ["x", "y"])


def test_correlation_analysis_basic():
    df = pd.DataFrame(
        {
            "target": [1, 2, 3, 4, 5],
            "feat_1": [1, 2, 3, 4, 5],  # perfect correlation
            "feat_2": [5, 4, 3, 2, 1],  # perfect negative correlation
            "feat_3": [1, 1, 1, 1, 1],  # zero correlation
        }
    )

    fs = FeatureSelector()
    corr_sorted = fs.correlation_analysis(df, target="target")

    # corr_sorted return the whole list
    assert corr_sorted.index.tolist() == ["feat_1", "feat_2", "feat_3"]
