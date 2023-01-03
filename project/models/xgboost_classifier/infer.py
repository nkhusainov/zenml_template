import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from models.config_base import ModelConfig


def infer_model(model: XGBClassifier, X: pd.DataFrame, is_probability: True, model_config: ModelConfig) -> np.ndarray:
    features_categorical = model_config.features_categorical
    if features_categorical is not None and len(features_categorical):
        X[features_categorical] = X[features_categorical].astype("category")

    if is_probability:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)
