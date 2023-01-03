import logging
import pandas as pd
from zenml.steps import Output, step

from utils.config_handler import get_main_config
from utils.oversampling import OverSamplingMethods

core_config = get_main_config()
OVERSAMPLER = core_config.oversampling_method
RANDOM_STATE = core_config.random_state


@step
def oversampling(
        X_train: pd.DataFrame,
        y_train: pd.Series,

) -> Output(X_train_res=pd.DataFrame, y_train_res=pd.Series):
    logger = logging.getLogger(__name__)
    logger.info(f"Original dataset shape {y_train.shape}")

    oversampler = OverSamplingMethods[OVERSAMPLER].value(random_state=RANDOM_STATE)
    X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
    logger.info(f"Resampled dataset shape {y_train_res.shape}")

    return X_train_res, y_train_res
