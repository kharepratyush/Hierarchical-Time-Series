import pickle
from typing import List

import pandas as pd
from hierarchicalforecast.utils import aggregate


def add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"y_lag_{lag}"] = df.groupby("unique_id")["y"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    for window in windows:
        df[f"y_roll_mean_{window}"] = df.groupby("unique_id")["y"].transform(
            lambda x: x.shift().rolling(window).mean()
        )
        df[f"y_roll_std_{window}"] = df.groupby("unique_id")["y"].transform(
            lambda x: x.shift().rolling(window).std()
        )
    return df


def add_ds_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ds"] = pd.to_datetime(df["ds"])
    df["quarter"] = df["ds"].dt.quarter
    df["day_of_year"] = df["ds"].dt.dayofyear
    df["week_of_year"] = df["ds"].dt.isocalendar().week
    return df


def feature_engineering(df: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    spec = [
        ["total"],
        ["total", "state"],
        ["total", "state", "zone"],
    ]
    df, S, tags = aggregate(df, spec)
    if train:
        with open("config/summing_matrix", "wb") as file:
            pickle.dump(S, file)

    df = add_lag_features(df, lags=[1, 3, 6, 12])
    df = add_rolling_features(df, windows=[3, 6, 12])
    df = add_ds_features(df)
    return df.sort_values(["unique_id", "ds"]).dropna()
