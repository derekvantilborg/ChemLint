"""Machine learning tools for model training and evaluation."""

from molml_mcp.tools.ml.metrics import calculate_metrics


def get_all_ml_tools():
    """Get all ML tools for registration."""
    return [
        calculate_metrics,
    ]
