from enum import Enum


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


#  Things below could be moved to a config file.
RENAME_MAPPING: dict[str, str] = {"family_histroy": "family_history"}

CATEGORICAL_MULTI: tuple[str] = ["smoking", "drinking"]
CATEGORICAL_BINARY: tuple[str] = ["Gender", "family_history"]
TARGET: str = "Diabetes"

CATEGORICAL_DATA_VALID_RANGES: dict[str, tuple[int, int]] = {
    "smoking": (1, 3),
    "drinking": (1, 3),
}
