""" Logistics for storing and loading resources."""

import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from molml_mcp.resources.supported_resource_types import TYPE_REGISTRY
from molml_mcp.resources import DATA_ROOT

# import pandas as pd
# df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
# resource_id = store_resource(df, "csv")
# load_resource(resource_id)


def _generate_id(type_tag: str) -> str:
    """Generate a unique resource ID with timestamp, type tag, and random ID component."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    rand = secrets.token_hex(4).upper()  # 8 hex chars

    ext = TYPE_REGISTRY[type_tag]["ext"]
    
    return f"{ts}_{type_tag}_{rand}{ext}"


def _store_resource(obj: Any, type_tag: str) -> str:
    if type_tag not in TYPE_REGISTRY:
        raise ValueError(f"Unsupported resource type: {type_tag}")
    
    rid = _generate_id(type_tag)

    path = DATA_ROOT / f"{rid}"
    save_fn: Callable[[Any, Path], None] = TYPE_REGISTRY[type_tag]['save']
    save_fn(obj, path)

    return rid

def _load_resource(resource_id: str) -> Any:
    # infer type from the ID. They are always in the format: TIMESTAMP_TYPE_RANDOM.ext
    try:
        type_tag = resource_id.split("_", 2)[1]
    except ValueError:
        raise ValueError(f"Invalid resource_id format: {resource_id}")

    if type_tag not in TYPE_REGISTRY:
        raise ValueError(f"Unknown resource type in id: {resource_id}")

    path = DATA_ROOT / f"{resource_id}"

    if not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")

    load_fn: Callable[[Path], Any] = TYPE_REGISTRY[type_tag]["load"]

    return load_fn(path)

def load_csv_dataset(file_path: str) -> dict:
    """ Load a CSV dataset from the given file path and store it. Return the corresponding resource_id + data preview."""
    import pandas as pd

    df = pd.read_csv(file_path)

    rid = _store_resource(df, "dataset")

    return {
        "resource_id": rid,
        "n_rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


