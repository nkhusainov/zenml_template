import json
import pandas
import bentoml
from bentoml.io import JSON
from fastapi import FastAPI
from typing import Dict, Any


def get_meta():
    with open("model_meta.json", "r") as file:
        return json.load(file)


fastapi_app = FastAPI()
model_meta = get_meta()
model_name = model_meta["model_name"]
version = "latest"
description = model_meta.get("description", "")
model_version = model_meta.get("version", "")
model_metrics = model_meta.get("metrics", {})
model_params = model_meta.get("params", {})
input_example, io_type = model_meta["input_example"]

DOC = f"""
    Model name: {model_name}.
    Model version mlflow: {model_version}.
    Description: {description}.
"""

runner = bentoml.mlflow.get(f"{model_name}:{version}").to_runner()
svc = bentoml.Service(model_name, runners=[runner])
svc.mount_asgi_app(fastapi_app)

Input = JSON.from_sample(input_example)

@svc.api(input=Input, output=JSON(), doc=DOC, name=model_name)
def service(input_data: Dict) -> Dict[str, Any]:
    if isinstance(input_data, Dict):
        input_data = [input_data]
    input_df = pandas.DataFrame(input_data)
    results = runner.predict.run(input_df)
    return {"predictions": results}


@fastapi_app.get("/metadata")
def metadata():
    return {
        "name": model_name,
        "version": model_version,
        "metrics": model_metrics,
        "parameters": model_params,
    }
