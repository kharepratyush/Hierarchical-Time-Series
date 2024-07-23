"""
Author: Pratyush Khare
"""

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from typing import Dict


def evaluate_model(test_df: pd.DataFrame, forecast_df: pd.DataFrame) -> float:
    """
    Evaluate a single model's forecast using RMSE.

    Args:
        test_df (pd.DataFrame): Test data.
        forecast_df (pd.DataFrame): Forecast data.

    Returns:
        float: RMSE of the forecast.
    """

    test_df = test_df.reset_index()[["unique_id", "ds", "y"]]
    forecast_df = forecast_df.reset_index().melt(
        id_vars=["ds"], var_name="unique_id", value_name="forecast"
    )

    forecast_df = (
        forecast_df.set_index(["unique_id", "ds"])
        .join(test_df.set_index(["unique_id", "ds"]))
        .reset_index()
    )

    return root_mean_squared_error(forecast_df["y"], forecast_df["forecast"])
