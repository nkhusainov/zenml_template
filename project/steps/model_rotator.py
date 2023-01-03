import mlflow
from mlflow.entities.model_registry import ModelVersion
from zenml.steps import step
from zenml.client import Client
from typing import Dict, Any

from utils.config_handler import get_main_config
from models.config_base import ModelConfig

core_config = get_main_config()
TARGET_METRIC = core_config.target_metric
METRIC_LOWER_BETTER = core_config.metric_lower_better

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def rotate_models(
        metrics_new: Dict[str, Any],
        metrics_current: Dict[str, Any],
        model_config: ModelConfig
) -> None:
    model_name = model_config.registry_name

    mlflow_client = mlflow.MlflowClient()
    versions = mlflow_client.get_latest_versions(name=model_name, stages=["Staging"])
    model_version: ModelVersion = versions[0]
    if (
            METRIC_LOWER_BETTER and metrics_new[TARGET_METRIC] <= metrics_current[TARGET_METRIC]
            or
            not METRIC_LOWER_BETTER and metrics_new[TARGET_METRIC] >= metrics_current[TARGET_METRIC]
    ):
        mlflow_client.transition_model_version_stage(
            name=model_name,
            stage="Production",
            version=model_version.version,
            archive_existing_versions=True
        )
    else:
        mlflow_client.transition_model_version_stage(
            name=model_name,
            stage="Archived",
            version=model_version.version,
        )
