import time
from enum import Enum
from pathlib import Path
from typing import Literal, NotRequired, TypeAlias, TypedDict

import pandas as pd
from sklearn.model_selection import train_test_split


class MissingHandlingStrategy(str, Enum):
    DROP: str = "drop"
    MEAN: str = "mean"
    MEDIAN: str = "median"


class FixCategoricalStrategy(str, Enum):
    CLIP: str = "clip"
    DROP: str = "drop"


class CorelationMethod(str, Enum):
    KENDALL: str = "kendall"
    PEARSON: str = "pearson"
    SPEARMAN: str = "spearman"


class AnomalyDetectionStrategy(str, Enum):
    IQR: str = "iqr"
    ISOLATION_FOREST: str = "isolation_forest"


class RandomForestParams(TypedDict, total=False):
    """
    Parameters for RandomForestClassifier.
    The dict might still include some errors.
    """

    n_estimators: int
    criterion: Literal["gini", "entropy", "log_loss"]
    max_depth: int | None
    min_samples_split: int | float
    min_samples_leaf: int | float
    min_weight_fraction_leaf: float
    max_features: int | float | Literal["sqrt", "log2", None]
    max_leaf_nodes: int | None
    bootstrap: bool
    oob_score: bool
    n_jobs: int | None
    random_state: int | None
    class_weight: dict[str, float] | Literal["balanced", "balanced_subsample", None]


class LogisticRegressionParams(TypedDict, total=False):
    """
    Parameters for LogisticRegression.
    The dict might still include some errors.
    """

    penalty: Literal["l1", "l2", "elasticnet", "none"]
    dual: bool
    tol: float
    C: float
    fit_intercept: bool
    intercept_scaling: float
    class_weight: dict[str, float] | Literal["balanced", None]
    random_state: int | None
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    max_iter: int
    multi_class: Literal["auto", "ovr", "multinomial"]
    l1_ratio: float | None


class MetricsDict(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


class ExperimentResult(TypedDict):
    experiment_name: str
    model_type: str
    params: LogisticRegressionParams | RandomForestParams
    threshold: float
    metrics: MetricsDict
    cm_plot_path: str
    timestamp: str
    roc_plot_path: NotRequired[str]


SplittedData: TypeAlias = list[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def prepare_data_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> SplittedData:
    """
    Splits the dataframe ensuring consistent random_state and stratification.

    :param df: Dataframe with the data to be splitted.
    :param target_col: Target column.
    :param test_size: Test group size.
    """
    X: pd.DataFrame = df.drop(columns=[target_col])
    y: pd.Series = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


#  Things below could be moved to a config file.
RENAME_MAPPING: dict[str, str] = {"family_histroy": "family_history"}

CATEGORICAL_MULTI: tuple[str] = ["smoking", "drinking"]
CATEGORICAL_BINARY: tuple[str] = ["Gender", "family_history"]
TARGET: str = "Diabetes"

CATEGORICAL_DATA_VALID_RANGES: dict[str, tuple[int, int]] = {
    "smoking": (1, 3),
    "drinking": (1, 3),
}

LR_RESULTS_PATH: Path = Path("results/models/LR/")
RF_RESULTS_PATH: Path = Path("results/models/RF/")
JSON_REPORT_NAME: str = f"{int(time.time())}.json"
