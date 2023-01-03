import yaml
from pathlib import Path
from typing import Dict
from pydantic import BaseModel


class CoreConfig(BaseModel):
    pipeline_name: str
    data_loader_cache: bool = False
    query_file_download_data: str
    random_state: int
    oversampling_method: str
    test_size: float
    target_metric: str
    metric_lower_better: bool


PATH_CONFIGS = Path(str(Path(__file__).resolve().parents[1]))
CORE_CONFIG_NAME = "project_config.yaml"


def load_yaml_config(file_name: str) -> Dict:
    file_path = PATH_CONFIGS / file_name
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_main_config() -> CoreConfig:
    config = load_yaml_config(file_name=CORE_CONFIG_NAME)
    return CoreConfig(**config)