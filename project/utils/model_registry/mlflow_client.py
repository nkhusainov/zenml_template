import os
import json
import importlib
import logging
import mlflow

from mlflow.entities.model_registry import ModelVersion

from typing import Any, Dict

logger = logging.getLogger(__name__)


def check_model_exists(model_name: str) -> bool:
    client = mlflow.MlflowClient()
    if len(client.search_registered_models(f"name='{model_name}'")):
        return True
    else:
        logger.warning(f"There is no registered model with name: {model_name}!")
        models = client.search_registered_models()
        logger.debug(f"Existing models: {models}!")
        return False


def check_model_stage_exist(model_name: str, stage: str) -> bool:
    client = mlflow.MlflowClient()
    if check_model_exists(model_name=model_name):
        if len(client.get_latest_versions(name=model_name, stages=[stage])):
            return True
    else:
        logger.warning(f"There is no model on stage: {stage}!")
        return False


def get_meta(model_name: str, stage: str) -> ModelVersion:
    client = mlflow.MlflowClient()
    if check_model_stage_exist(model_name=model_name, stage=stage):
        return client.get_latest_versions(name=model_name, stages=[stage])[0]
    else:
        err_msg = f"There is no model on stage: {stage}!"
        raise mlflow.MlflowException(message=err_msg)


def _get_model_loader(run_id: str) -> str:
    run = mlflow.get_run(run_id=run_id)
    run_data = json.loads(run.data.tags["mlflow.log-model.history"])[0]
    loader_module = run_data["flavors"]["python_function"]["loader_module"]
    return loader_module


def get_model(model_name: str, stage: str) -> Any:
    version_meta = get_meta(model_name=model_name, stage=stage)
    run_id = version_meta.run_id
    model_uri = version_meta.source

    loader_module = _get_model_loader(run_id=run_id)
    mlflow_loader = importlib.import_module(loader_module)
    model = mlflow_loader.load_model(model_uri)

    return model


def get_input_example(model_name: str, stage: str) -> Dict[str, Any]:
    version_meta = get_meta(model_name=model_name, stage=stage)
    run_id = version_meta.run_id
    path_artifacts = version_meta.source

    run = mlflow.get_run(run_id=run_id)
    run_data = json.loads(run.data.tags["mlflow.log-model.history"])[0]
    example_info = run_data.get("saved_input_example_info", None)
    if example_info is not None:
        file_name = example_info['artifact_path']
        example = mlflow.artifacts.load_dict(os.path.join(path_artifacts, file_name))
        example_type = example_info["type"]
        if example_type == "dataframe":
            columns = example["columns"]
            data = example["data"][0]
            example_json = {}
            for i, col in enumerate(columns):
                # workaround for pydantic model bug
                val = data[i] + 0.1 if isinstance(data[i], float) else data[i]
                example_json[col] = val
            return example_json
        else:
            err_msg = "Example type: {example_type},  handling is not implemented"
            raise ValueError(err_msg)
    else:
        raise mlflow.MlflowException("Data example is not provided")


def get_requirements_path(model_uri: str) -> str:
    return mlflow.pyfunc.get_model_dependencies(model_uri)
