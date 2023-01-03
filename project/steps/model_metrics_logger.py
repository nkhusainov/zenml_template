import mlflow
from zenml.client import Client
from zenml.steps import Output, step
from typing import Dict

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def log_metrics(metrics: Dict) -> Output(metrics=Dict):
    """
    Computation of the different classification metrics for our old_models
    """
    mlflow.log_metrics(metrics=metrics)

    return metrics
