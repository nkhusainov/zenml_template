import pandas as pd
from zenml.steps import Output, step

from models.config_base import ModelConfig
from utils.config_handler import get_main_config
from utils.db.query_loader import read_query
from utils.db.bq_client import run_data_query_to_dataframe

config = get_main_config()
DATA_LOADER_CACHE = config.data_loader_cache
QUERY_FILE_NAME = config.query_file_download_data


@step(enable_cache=DATA_LOADER_CACHE)
def download_data_from_bq(model_config: ModelConfig) -> Output(raw_data=pd.DataFrame):
    query = read_query(file_name=QUERY_FILE_NAME)
    dataframe = run_data_query_to_dataframe(query=query)

    dtypes = {model_config.target: int}
    return dataframe.astype(dtypes)
