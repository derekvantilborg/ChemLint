from pathlib import Path
from typing import Any


TYPE_REGISTRY: dict[str, dict[str, Any]] = {}
# Supported resource types and their handlers:
# csv
# model
# json



# csv
def _save_csv(obj, path: Path):
    # assume obj is a pandas DataFrame
    import pandas as pd
    assert hasattr(obj, "to_csv"), "csv type expects a DataFrame-like object"
    obj.to_csv(path, index=False)

def _load_csv(path: Path):
    import pandas as pd
    return pd.read_csv(path)

TYPE_REGISTRY["csv"] = {
    "ext": ".csv",
    "save": _save_csv,
    "load": _load_csv,
}


# model
def _save_model(model, path: Path):
    import joblib
    joblib.dump(model, path)

def _load_model(path: Path):
    import joblib
    return joblib.load(path)

TYPE_REGISTRY["model"] = {
    "ext": ".pkl",
    "save": _save_model,
    "load": _load_model,
}


# json
def _save_json(obj, path: Path):
    import json
    with open(path, "w") as f:
        json.dump(obj, f)

def _load_json(path: Path):
    import json
    with open(path, "r") as f:
        return json.load(f) 

TYPE_REGISTRY["json"] = {
    "ext": ".json",
    "save": _save_json,
    "load": _load_json,
}