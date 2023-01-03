import pandas as pd
from zenml.steps import Output, step

from models.model_selector import Models
from models.config_base import ModelConfig


@step
def preprocess_data(data: pd.DataFrame, model_config: ModelConfig) -> Output(data_processed=pd.DataFrame):
    features_numeric = model_config.features_numeric
    features_categorical = [] if model_config.features_categorical is None else model_config.features_categorical
    target = model_config.target
    data = data[features_numeric + features_categorical + [target]]

    model_handler = Models[model_config.framework].value

    if model_handler.func_preprocess is not None:
        data = model_handler.func_preprocess(data=data, model_config=model_config)
        return data

    return data
