import os
import yaml
from typing import NoReturn
import bentoml


def import_mlflow_model(model_name, model_uri, **kwargs) -> NoReturn:
    bentoml.mlflow.import_model(
        name=model_name,
        model_uri=model_uri,
        metadata={
            "metrics": kwargs.get("metrics", {}),
            "params": kwargs.get("params", {}),
        }
    )


def create_bentofile(dest_path: str, **kwargs):
    bentofile = {
        "service": "service.py:svc",
        "description": kwargs.get("description", ""),
        "include": ["service.py", "*.yaml", "*.txt", "*.json"],
        "python": {"requirements_txt": "requirements.txt"}
    }
    with open(dest_path, 'w') as f:
        yaml.dump(bentofile, f, sort_keys=False, default_flow_style=False)


# from zenml.integrations.model_serving.services