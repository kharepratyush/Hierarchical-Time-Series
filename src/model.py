"""
Author: Pratyush Khare
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Set configuration for xgboost and environment variable
xgb.set_config(verbosity=0)
os.environ["NIXTLA_ID_AS_COL"] = "1"


def train_val_split(
    df: pd.DataFrame, group_columns: str, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into training and validation sets based on the date.

    Args:
        df (pd.DataFrame): The dataframe to be split.
        group_columns (str): The column name to group by.
        test_size (float): The proportion of the dataset to include in the validation split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and validation dataframes.
    """
    df = df.drop(columns=group_columns)
    ds = sorted(df.ds.unique())
    split_index = int(len(ds) * (1 - test_size))

    df = df.sort_values(by="ds")
    train_df = df[df["ds"] <= ds[split_index - 1]]
    val_df = df[df["ds"] > ds[split_index - 1]]
    return train_df, val_df


def x_y_split(
    df: pd.DataFrame, exogenous: bool = True
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """
    Splits the dataframe into target variable y and features X.

    Args:
        df (pd.DataFrame): The dataframe to be split.
        exogenous (bool): Whether to include exogenous variables.

    Returns:
        Tuple[pd.Series, Optional[pd.DataFrame]]: The target variable y and features X.
    """
    y = df["y"]
    if exogenous:
        X = df.drop(columns=["y", "ds"])
        return y, X
    return df[["ds", "y"]], None


def fit_lightgbm(
    df: pd.DataFrame, group_columns: str, group_name: str
) -> Dict[str, Any]:
    """
    Fits a LightGBM model.

    Args:
        df (pd.DataFrame): The dataframe to fit the model on.
        group_columns (str): The column name to group by.
        group_name (str): The name of the group.

    Returns:
        Dict[str, Any]: The fitted LightGBM model and its RMSE.
    """
    train_df, val_df = train_val_split(df, group_columns)
    y_train, X_train = x_y_split(train_df)
    y_val, X_val = x_y_split(val_df)

    def lgb_evaluate(num_leaves, learning_rate, feature_fraction):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": int(num_leaves),
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "verbosity": -1,
            "verbose_eval": -1,
        }

        dtrain = lgb.Dataset(X_train, label=y_train)
        cv_results = lgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=5,
            stratified=False,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
            ],
            seed=42,
        )

        return -min(cv_results["valid rmse-mean"])

    lgb_bo = BayesianOptimization(
        lgb_evaluate,
        {
            "num_leaves": (24, 45),
            "learning_rate": (0.01, 0.2),
            "feature_fraction": (0.1, 0.9),
        },
        random_state=42,
    )

    lgb_bo.maximize(init_points=5, n_iter=100)  ## Update later

    best_params = lgb_bo.max["params"]
    best_params["num_leaves"] = int(best_params["num_leaves"])
    best_params["verbosity"] = -1

    dtrain = lgb.Dataset(X_train, label=y_train, params={"verbosity": -1})
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, params={"verbosity": -1})

    model = lgb.train(
        best_params,
        dtrain,
        num_boost_round=10000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )

    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    logging.info(f"LightGBM Group Name: {group_name}, RMSE: {rmse}")

    return {"model": model, "type": "LightGBM", "rmse": rmse}


def fit_xgboost(
    df: pd.DataFrame, group_columns: str, group_name: str
) -> Dict[str, Any]:
    """
    Fits an XGBoost model.

    Args:
        df (pd.DataFrame): The dataframe to fit the model on.
        group_columns (str): The column name to group by.
        group_name (str): The name of the group.

    Returns:
        Dict[str, Any]: The fitted XGBoost model and its RMSE.
    """
    train_df, val_df = train_val_split(df, group_columns)
    y_train, X_train = x_y_split(train_df)
    y_val, X_val = x_y_split(val_df)

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)

    def xgb_evaluate(max_depth, learning_rate, colsample_bytree):
        params = {
            "objective": "reg:squarederror",
            "max_depth": int(max_depth),
            "learning_rate": learning_rate,
            "colsample_bytree": colsample_bytree,
            "eval_metric": "rmse",
            "verbosity": 0,
            "tree": "hist",
        }

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=5,
            early_stopping_rounds=200,
            seed=42,
            verbose_eval=False,
        )
        return -min(cv_results["test-rmse-mean"])

    xgb_bo = BayesianOptimization(
        xgb_evaluate,
        {
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.2),
            "colsample_bytree": (0.3, 0.9),
        },
        random_state=42,
    )

    xgb_bo.maximize(init_points=5, n_iter=100)  ## Update later

    best_params = xgb_bo.max["params"]
    best_params["max_depth"] = int(best_params["max_depth"])

    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=10000,
        evals=[(dval, "eval")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )

    y_pred = model.predict(dval)
    rmse = root_mean_squared_error(y_val, y_pred)
    logging.info(f"XGBoost Group Name: {group_name}, RMSE: {rmse}")

    return {"model": model, "type": "XGBoost", "rmse": rmse}


def fit_models(df: pd.DataFrame, group_columns: str, group_name: str) -> Dict[str, Any]:
    """
    Fits multiple models and selects the best one based on RMSE.

    Args:
        df (pd.DataFrame): The dataframe to fit the models on.
        group_columns (str): The column name to group by.
        group_name (str): The name of the group.

    Returns:
        Dict[str, Any]: The best model and its RMSE.
    """
    models = [
        fit_lightgbm(df, group_columns, group_name),
        fit_arima(df, group_columns, group_name),
        fit_xgboost(df, group_columns, group_name),
        fit_exponential_smoothing(df, group_columns, group_name),
    ]

    best_model = min(models, key=lambda model: model["rmse"])
    return best_model


def fit_exponential_smoothing(
    df: pd.DataFrame, group_columns: str, group_name: str
) -> Dict[str, Any]:
    """
    Fits an Exponential Smoothing model.

    Args:
        df (pd.DataFrame): The dataframe to fit the model on.
        group_columns (str): The column name to group by.
        group_name (str): The name of the group.

    Returns:
        Dict[str, Any]: The fitted Exponential Smoothing model and its RMSE.
    """
    train_df, val_df = train_val_split(df, group_columns)
    y_train, _ = x_y_split(train_df, exogenous=False)
    y_val, _ = x_y_split(val_df, exogenous=False)

    y_train = pd.concat(
        [y_train.reset_index(drop=True), pd.Series([group_name] * len(y_train))],
        axis=1,
        ignore_index=True,
    )
    y_train.columns = ["ds", "y", group_columns]

    sf = StatsForecast(models=[AutoETS()], freq="D")
    sf.fit(y_train)
    y_pred = sf.predict(h=len(y_val))["AutoETS"].values

    rmse = root_mean_squared_error(y_val.y.values, y_pred)
    logging.info(f"AutoETS Group Name: {group_name}, RMSE: {rmse}")

    return {"model": sf, "type": "AutoETS", "rmse": rmse}


def fit_arima(df: pd.DataFrame, group_columns: str, group_name: str) -> Dict[str, Any]:
    """
    Fits an AutoARIMA model.

    Args:
        df (pd.DataFrame): The dataframe to fit the model on.
        group_columns (str): The column name to group by.
        group_name (str): The name of the group.

    Returns:
        Dict[str, Any]: The fitted AutoARIMA model and its RMSE.
    """
    train_df, val_df = train_val_split(df, group_columns)
    y_train, _ = x_y_split(train_df, exogenous=False)
    y_val, _ = x_y_split(val_df, exogenous=False)

    y_train = pd.concat(
        [y_train.reset_index(drop=True), pd.Series([group_name] * len(y_train))],
        axis=1,
        ignore_index=True,
    )
    y_train.columns = ["ds", "y", group_columns]

    sf = StatsForecast(models=[AutoARIMA()], freq="D", n_jobs=-1)
    sf.fit(y_train)
    y_pred = sf.predict(h=len(y_val))["AutoARIMA"].values

    rmse = root_mean_squared_error(y_val.y.values, y_pred)
    logging.info(f"AutoARIMA Group Name: {group_name}, RMSE: {rmse}")

    return {"model": sf, "type": "AutoARIMA", "rmse": rmse}


def fit_time_series_model(
    df: pd.DataFrame, group_column: str
) -> Dict[str, Dict[str, Any]]:
    """
    Fits time series models for each group in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to fit the models on.
        group_column (str): The column name to group by.

    Returns:
        Dict[str, Dict[str, Any]]: The best model for each group and their RMSE.
    """
    models = {}
    df = df.reset_index()
    grouped = df.groupby(group_column)

    for name, group in grouped:
        logging.info("-" * 50)
        best_model = fit_models(group, group_column, name)
        models[name] = {
            "best_model": best_model["model"],
            "model_type": best_model["type"],
            "best_rmse": best_model["rmse"],
        }
        logging.info(
            f"Best Model Group Name: {name}, RMSE: {best_model['rmse']}, Model Type: {best_model['type']}"
        )

    return models


def forecast(
    models: Dict[str, Dict[str, Any]],
    future_data_path: str,
    group_columns: str,
    output_prediction_path: str,
) -> pd.DataFrame:
    """
    Generates forecasts using the fitted models and future data.

    Args:
        models (Dict[str, Dict[str, Any]]): The fitted models.
        future_data_path (str): The path to the future data.
        group_columns (str): The column name to group by.
        output_prediction_path (str): The path to save the forecast output.

    Returns:
        pd.DataFrame: The forecast dataframe.
    """
    forecasts = {}
    future_df = pd.read_csv(future_data_path)
    grouped = future_df.groupby(group_columns)

    for name, group in grouped:
        try:
            model_info = models[name]
            model = model_info["best_model"]
            model_type = model_info["model_type"]

            if model_type == "XGBoost":
                test_df = group.drop(columns=group_columns)
                X_test = test_df.drop(columns=["y", "ds"])
                y_test = test_df["y"]
                dtest = xgb.DMatrix(data=X_test, label=y_test)
                group["forecast"] = model.predict(dtest)

            elif model_type in ["AutoARIMA", "AutoETS"]:
                group["forecast"] = model.predict(h=len(group))[model_type].values

            elif model_type == "LightGBM":
                test_df = group.drop(columns=group_columns)
                X_test = test_df.drop(columns=["y", "ds"])
                group["forecast"] = model.predict(X_test)

            forecasts[name] = group[[group_columns, "ds", "y", "forecast"]]

        except Exception as e:
            logging.info(f"Prediction Failed for {name}: {e}")
            raise e

    forecast_df = pd.concat(forecasts.values())
    forecast_df = forecast_df.pivot_table(
        values="forecast", columns="unique_id", index="ds"
    )

    forecast_df.to_csv(output_prediction_path)
    return forecast_df
