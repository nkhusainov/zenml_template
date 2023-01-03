import pandas as pd
import numpy as np
from zenml.steps import Output, step
from sklearn.pipeline import Pipeline

from models.model_selector import Models
from models.config_base import ModelConfig

# to do pass framework name to this step
@step
def predict(
        model: Pipeline,
        X: pd.DataFrame,
        model_config: ModelConfig
) -> Output(y_pred=np.ndarray, y_pred_proba=np.ndarray):
    """
    Computation of the different classification metrics for our old_models
    """
    model_handler = Models[model_config.framework].value

    y_pred = model_handler.func_infer(model, X, is_probability=False, model_config=model_config)
    y_pred_proba = model_handler.func_infer(model, X, is_probability=True, model_config=model_config)

    return y_pred, y_pred_proba
