from enum import Enum
from types import DynamicClassAttribute

from models import sklearn_classifier
from models import xgboost_classifier
from models.handler_base import ModelHandler


class Models(Enum):
    sklearn: ModelHandler = ModelHandler(
        func_preprocess=sklearn_classifier.data_preprocessing,
        func_train=sklearn_classifier.train_model,
        func_infer=sklearn_classifier.infer_model
    )
    xgboost: ModelHandler = ModelHandler(
        func_preprocess=xgboost_classifier.data_preprocessing,
        func_train=xgboost_classifier.train_model,
        func_infer=xgboost_classifier.infer_model
    )
    lightgbm: ModelHandler = None

    @DynamicClassAttribute
    def value(self) -> ModelHandler:
        return self._value_
