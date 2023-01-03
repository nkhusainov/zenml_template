import pandas as pd
import mlflow
from zenml.steps import Output, step
from zenml.client import Client

from models.model_selector import Models
from models.config_base import ModelConfig
from utils.zenml_meterializer.model_materializer import ModelMaterializer, ModelClass

experiment_tracker = Client().active_stack.experiment_tracker


def register_model(model_name: str, descriptions: str = None):
    mlflow_client = mlflow.MlflowClient()
    if not len(mlflow_client.search_registered_models(filter_string=f'name="{model_name}"')):
        mlflow_client.create_registered_model(name=model_name, description=descriptions)

    run = mlflow.active_run()
    model_uri = "runs:/{}/model".format(run.info.run_id)
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    mlflow_client.transition_model_version_stage(
        name=model_name,
        stage="Staging",
        version=registered_model.version,
        archive_existing_versions=True
    )
    return registered_model.run_id


@step(
    experiment_tracker=experiment_tracker.name,
    output_materializers=ModelMaterializer,
)
def train_model(
        X_train_res: pd.DataFrame,
        y_train_res: pd.Series,
        model_config: ModelConfig
) -> Output(model=ModelClass):
    model_handler = Models[model_config.framework].value

    mlflow.autolog(log_input_examples=True, log_models=True)
    model = model_handler.func_train(X=X_train_res, y=y_train_res, model_config=model_config)
    _ = register_model(model_name=model_config.registry_name, descriptions=model_config.description)

    return ModelClass(model)
