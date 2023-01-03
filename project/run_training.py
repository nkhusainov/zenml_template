from pipelines.train_pipeline import training_model
from steps.data_loader import download_data_from_bq
from steps.data_preprocessor import preprocess_data
from steps.data_oversampler import oversampling
from steps.data_splitter import split_data
from steps.model_trainer import train_model
from steps.predictor import predict
from steps.model_evaluator import evaluate_model
from steps.model_metrics_logger import log_metrics

from utils.config_handler import load_yaml_config
from models.config_base import ModelConfig

CONFIG_NAME = "model_config.yaml"
config = load_yaml_config(CONFIG_NAME)
model_config = ModelConfig(**config)

train_pipeline = training_model(
    download_data=download_data_from_bq(model_config),
    process_data=preprocess_data(model_config),
    oversampling=oversampling(model_config),
    split_data=split_data(model_config),
    train_model=train_model(model_config),
    predict_new_model=predict(model_config),
    evaluate_new_model=evaluate_model(),
    log_metrics=log_metrics(),
)



if __name__ == "__main__":
    # Initialize a pipeline run
    train_pipeline.run()
