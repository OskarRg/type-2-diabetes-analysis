from pydantic import BaseModel, Field


class LogisticRegressionParams(BaseModel):
    C: float = Field(default=1.0, gt=0)
    max_iter: int = Field(default=1000, gt=0)
    solver: str = "lbfgs"


class RandomForestParams(BaseModel):
    n_estimators: int = Field(default=100, gt=0)
    max_depth: int | None = None
    random_state: int = 42
    n_jobs: int = -1


class LRExperiment(BaseModel):
    name: str
    params: LogisticRegressionParams
    threshold: float = 0.3


class RFExperiment(BaseModel):
    name: str
    params: RandomForestParams
    threshold: float = 0.5


class Experiments(BaseModel):
    logistic_regression: list[LRExperiment] = []
    random_forest: list[RFExperiment] = []


class FeatureSelectionParams(BaseModel):
    enabled: bool = False
    method: str = "correlation"
    top_k: int | None = Field(
        default=None,
        gt=0,
        description="A number of the features with the highest corelation to be used.",
    )
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimal corelation threshold"
    )


class PreprocessingParams(BaseModel):
    drop_in_fix_categorical: bool
    remove_outliers: bool


class DataParams(BaseModel):
    input_path: str
    target_column: str
    feature_selection: FeatureSelectionParams = FeatureSelectionParams()
    preprocessing: PreprocessingParams


class PipelineParams(BaseModel):
    data: DataParams
    experiments: Experiments
