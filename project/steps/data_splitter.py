import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step

from models.config_base import ModelConfig
from utils.config_handler import get_main_config

core_config = get_main_config()
OVERSAMPLER = core_config.oversampling_method
RANDOM_STATE = core_config.random_state
TEST_SIZE = core_config.test_size

@step
def split_data(
        data: pd.DataFrame, model_config: ModelConfig) -> Output(
    X_train=pd.DataFrame, X_test=pd.DataFrame, y_train=pd.Series, y_test=pd.Series
):
    features_numeric = model_config.features_numeric
    features_categorical = [] if model_config.features_categorical is None else model_config.features_categorical
    target = model_config.target
    X = data[features_numeric + features_categorical]
    y = data[target].squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"X_train_shape: {X_train.shape}")
    logger.info(f"X_test_shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test
