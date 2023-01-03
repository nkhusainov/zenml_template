import pandas as pd

from google.auth import default
from google.cloud import bigquery


def auth_project_id() -> str:
    """
    Return authenticated GCP project
    :return: GCP project
    """
    _, project_id = default()
    return project_id


# source https://github.com/adeo/opus--ranking-ml/blob/dev/ml-pipeline/utils/bigquery.py
# changes:
#   auth_project_id func
#   _run_query expects query not a path
#   deleted unused functions

def _run_query(query: str, job_config: bigquery.QueryJobConfig = None) -> bigquery.table.RowIterator:
    """
    Instantiate a BigQuery client,
    read a query in the /queries subfolder and run it on BigQuery
    """
    project_id = auth_project_id()
    client = bigquery.Client(project=project_id)
    return client.query(query=query, job_config=job_config).result()


def run_data_query(output_table_ref=None, **kwargs) -> bigquery.table.RowIterator:
    """
    Run query to create table with job configs
    """
    job_config = bigquery.QueryJobConfig(
        destination=output_table_ref,
    )
    return _run_query(job_config=job_config, **kwargs)


def run_ml_query(**kwargs) -> bigquery.table.RowIterator:
    """
    Run query to create model without job configs
    """
    return _run_query(**kwargs)


def run_data_query_to_dataframe(**kwargs) -> pd.DataFrame:
    """
    Instantiate a BigQuery client,
    read a query in the /queries subfolder and run it on BigQuery and
    returns it as a pandas DataFrame
    """
    return run_data_query(**kwargs).to_dataframe()
