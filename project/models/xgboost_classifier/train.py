import pandas as pd
from xgboost import XGBClassifier

from models.config_base import ModelConfig


def train_model(X: pd.DataFrame, y: pd.Series, model_config: ModelConfig) -> XGBClassifier:
    hyper_parameters = model_config.hyper_parameters
    features_categorical = model_config.features_categorical

    if hyper_parameters is None:
        hyper_parameters = {}

    if features_categorical is not None and len(features_categorical):
        hyper_parameters["enable_categorical"] = True
        hyper_parameters["tree_method"] = "hist"
        X[features_categorical] = X[features_categorical].astype("category")

    model = XGBClassifier(**hyper_parameters)
    model.fit(X, y)
    return model
