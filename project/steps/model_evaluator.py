import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from zenml.client import Client
from zenml.steps import Output, step
from typing import Dict

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        y_test: pd.Series
) -> Output(metrics=Dict):
    """
    Computation of the different classification metrics for our old_models
    """
    metrics = {
        "f1_score": f1_score(y_true=y_test, y_pred=y_pred),
        "accuracy": accuracy_score(y_true=y_test, y_pred=y_pred),
        "recall": recall_score(y_true=y_test, y_pred=y_pred),
        "precision": precision_score(y_true=y_test, y_pred=y_pred),
    }
    if y_pred_proba is not None:
        metrics.update({
            "roc_auc": roc_auc_score(y_true=y_test, y_score=y_pred_proba)
        })
    return metrics
