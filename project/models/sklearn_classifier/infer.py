import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


def infer_model(model: Pipeline, X: pd.DataFrame, is_probability: True, *args, **kwargs) -> np.ndarray:
    if is_probability:
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict(X)
