import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from models.config_base import ModelConfig


def _define_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )

def train_model(X: pd.DataFrame, y: pd.Series, model_config: ModelConfig) -> Pipeline:
    hyper_parameters = model_config.hyper_parameters
    model = _define_model()
    if hyper_parameters is None:
        hyper_parameters = {}
    model.fit(X, y, **hyper_parameters)
    return model
