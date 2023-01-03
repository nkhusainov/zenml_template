from typing import List, Dict, Any
from zenml.steps import BaseParameters


class ModelConfig(BaseParameters):
    version: str
    name: str
    description: str
    framework: str
    model_type: str
    features_numeric: List[str]
    features_categorical: List[str] = None
    target: str
    test_size: float = 0.3
    hyper_parameters: Dict[str, Any] = None
    extra_values: Dict[str, Any] = None

    @property
    def registry_name(self):
        return f"model:{self.name}--version:{self.version}"
