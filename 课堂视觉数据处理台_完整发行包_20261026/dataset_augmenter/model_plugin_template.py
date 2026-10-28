"""Copy this file and replace predict_batch with a real task model.

The desktop app passes a pandas DataFrame containing ``input_path``,
``sample_id`` and ``condition``.  Return a DataFrame or a list of prediction
values.  Keep sample_id/condition unchanged so the evaluation page can pair
original and degraded conditions.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def predict_batch(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Replace this baseline with your classifier/regressor/model inference."""
    output = frame[["sample_id", "condition", "task"]].copy()
    # Example only: a real plugin must read input_path and run its model.
    output["y_pred"] = "待替换"
    if "y_true" in frame.columns:
        output["y_true"] = frame["y_true"]
    output["model_name"] = config.get("model_name", "example_plugin")
    output["model_version"] = config.get("model_version", "0.0")
    return output
