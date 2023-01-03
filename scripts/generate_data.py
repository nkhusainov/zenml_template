import os
import pandas as pd
import numpy as np
import random
import string
from typing import Tuple


def generate_target(samples_qty: int = 100) -> Tuple:
    values_float = np.random.uniform(low=0, high=1, size=samples_qty)
    values_binary = np.where(values_float < 0.5, 0, 1)
    return values_float, values_binary


def generate_features(
        target_values: np.array,
        columns_qty: int = 10,
        base_name: str = "col",
        max_noize: float = 0.9,
        data_type=float,
        create_negative=True
) -> pd.DataFrame:
    column_names = [f"{base_name}_{i}" for i in range(columns_qty)]
    columns_int_values = []
    for i in range(columns_qty):
        high = np.random.randint(100) / 100 * max_noize
        if create_negative:
            high *= np.random.choice([-1, 1])
        values = np.random.uniform(low=0, high=high, size=len(target_values))
        columns_int_values.append(target_values + values)
    return pd.DataFrame(dict(zip(column_names, columns_int_values)), dtype=data_type)


def generate_ids(samples_qty: int, id_lengt: int = 50):
    return [''.join(random.choices(string.ascii_lowercase, k=id_lengt)) for i in range(samples_qty)]


if __name__ == "__main__":
    GCP_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
    BQ_DATASET = "zenml_demo"
    BQ_TABLE = "dataset"
    SAMPLES_QTY = 1000
    NUMERIC_COLUMNS_QTY = 20
    NUMERIC_MAX_NOIZE = 0.8
    CAT_COLUMNS_QTY = 20
    CAT_MAX_NOIZE = 0.8
    SEED = 42
    np.random.seed(SEED)

    # TARGETS
    values_float, values_binary = generate_target(samples_qty=SAMPLES_QTY)

    # NUMERIC COLUMNS
    base_name = "col_num"
    data_type = float
    create_negative = True

    data_numeric = generate_features(
        target_values=values_float,
        columns_qty=NUMERIC_COLUMNS_QTY,
        base_name=base_name,
        max_noize=NUMERIC_MAX_NOIZE,
        data_type=data_type,
        create_negative=create_negative
    )

    # CATEGORICAL COLUMNS
    base_name = "col_cat"
    data_type = int

    data_categorical = generate_features(
        target_values=values_float,
        columns_qty=CAT_COLUMNS_QTY,
        base_name=base_name,
        max_noize=5,
        data_type=data_type,
        create_negative=False
    )

    dataset = pd.concat([data_numeric, data_categorical], axis=1)
    dataset["target_numeric"] = values_float
    dataset["target_binary"] = values_binary
    dataset["id"] = generate_ids(SAMPLES_QTY)

    # SAVE TO BQ
    dataset.to_gbq(
        f'{BQ_DATASET}.{BQ_TABLE}',
        GCP_PROJECT,
        chunksize=None,
        if_exists='replace',
    )
