import importlib
from zenml.steps import Output, step
from zenml.client import Client
from models.config_base import ModelConfig

from utils.zenml_meterializer.model_materializer import ModelMaterializer, ModelClass
from utils.model_registry.mlflow_client import get_model

experiment_tracker = Client().active_stack.experiment_tracker


def download_model(loader_module: str, model_uri: str):
    mlflow_loader = importlib.import_module(loader_module)
    return mlflow_loader.load_model(model_uri)


@step(
    experiment_tracker=experiment_tracker.name,
    output_materializers=ModelMaterializer,
)
def get_registry_model(
        model_config: ModelConfig
) -> Output(model=ModelClass):
    model_name = model_config.registry_name
    stage = "Production"
    model = get_model(model_name=model_name, stage=stage)
    return ModelClass(model)
