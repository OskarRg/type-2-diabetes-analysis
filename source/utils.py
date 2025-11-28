from enum import Enum
from typing import Literal, TypedDict


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

#  Things below could be moved to a config file.
RENAME_MAPPING: dict[str, str] = {"family_histroy": "family_history"}

CATEGORICAL_MULTI: tuple[str] = ["smoking", "drinking"]
CATEGORICAL_BINARY: tuple[str] = ["Gender", "family_history"]
TARGET: str = "Diabetes"

CATEGORICAL_DATA_VALID_RANGES: dict[str, tuple[int, int]] = {
    "smoking": (1, 3),
    "drinking": (1, 3),
}
