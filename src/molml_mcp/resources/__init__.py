

# You can set your custom data directory by running (change path to desired location): export MOLML_MCP_DATA_DIR="~/user/molml_mcp_data"
# If not set, defaults to ~/.molml_mcp/

import os
from pathlib import Path

def get_data_root() -> Path:
    # Allow user to override via environment variable
    custom = os.getenv("MOLML_MCP_DATA_DIR")
    if custom:
        root = Path(custom).expanduser()
    else:
        # Default: ~/.molml_mcp/
        root = Path.home() / ".molml_mcp"

    root.mkdir(parents=True, exist_ok=True)
    return root

DATA_ROOT = get_data_root()