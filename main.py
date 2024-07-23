"""
Author: Pratyush Khare
"""

import logging

from src.config import configuration_setup
from src.custom_exception import HTSCustomException
from src.data_loader import load_data
from src.feature_engineering import feature_engineering
from src.preprocess import (
    preprocess_data,
    save_processed_data,
    train_test_split_time_series,
)
from src.model import fit_time_series_model, forecast
from src.reconciliation import perform_forecast_reconciliation
from src.utils import setup_logging, create_folders
from src.evaluation import evaluate_model


def main() -> int:
    # Initialize logging
    setup_logging()

    # Step 1: Create necessary folders
    try:
        create_folders()
    except OSError:
        logging.exception("Exception occurred during creating bootstrap folders")
        return 1

    # Step 2: Define paths
    try:
        cfg = configuration_setup()
        raw_data_path = cfg["raw_data_path"]
        holidays_path = cfg["holidays"]
        processed_data_path = cfg["processed_data_path"]
        train_path = cfg["train_path"]
        test_path = cfg["test_path"]
        feature_engineered_train_path = cfg["feature_engineered_train_path"]
        feature_engineered_test_path = cfg["feature_engineered_test_path"]
        output_prediction_path = cfg["output_prediction_path"]
        output_reconciliation_path = cfg["output_reconciliation_path"]
        logging.info("Configuration loaded successfully")
    except Exception as e:
        logging.exception("Exception occurred during defining data paths")
        return 1

    # Step 3: Load raw data
    try:
        df = load_data(raw_data_path)
        holidays = load_data(holidays_path)
        logging.info("Raw dataframe loaded successfully")
    except HTSCustomException as e:
        logging.exception("Exception occurred during loading raw data")
        logging.exception(e)
        return 1

    # Step 4: Preprocess data
    try:
        df = preprocess_data(df, holidays)
        save_processed_data(df, processed_data_path, index=True)
        logging.info("Processed dataframe saved successfully")
    except Exception as e:
        logging.exception("Exception occurred during preprocessing of raw data")
        logging.exception(e)
        return 1

    # Step 5: Split data based on time series
    try:
        train_df, test_df = train_test_split_time_series(
            processed_data_path,
            train_path,
            test_path,
            test_size=cfg["forecast"]["test_size"],
        )
        logging.info("Train & Test dataset split and saved successfully")
    except Exception as e:
        logging.exception("Exception occurred during splitting test-train datasets")
        logging.exception(e)
        return 1

    # Step 6: Feature engineering
    try:
        train_df = feature_engineering(train_df)
        save_processed_data(train_df, feature_engineered_train_path)

        test_df = feature_engineering(test_df, train=False)
        save_processed_data(test_df, feature_engineered_test_path)

        logging.info("Feature engineered datasets saved successfully")
    except Exception as e:
        logging.exception("Exception occurred during feature engineering")
        logging.exception(e)
        return 1

    # Step 7: Fit models
    try:
        models = fit_time_series_model(train_df, "unique_id")
        logging.info("Model training completed successfully!")
    except Exception as e:
        logging.exception("Exception occurred during model training")
        logging.exception(e)
        return 1

    # Step 8: Forecast
    try:
        forecast_df = forecast(
            models,
            future_data_path=feature_engineered_test_path,
            group_columns="unique_id",
            output_prediction_path=output_prediction_path,
        )
        logging.info("Model Forecasting completed successfully!")
    except Exception as e:
        logging.exception("Exception occurred during model forecasting")
        logging.exception(e)
        raise e
        return 1

    # Step 9: Reconcile Forecast
    try:
        reconciled_forecasts = perform_forecast_reconciliation(
            forecast_df, output_reconciliation_path
        )
        logging.info("Output Reconciliation completed successfully!")
    except Exception as e:
        logging.exception("Exception occurred during output reconciliation")
        logging.exception(e)
        return 1

    # Step 10: Evaluate models
    try:
        reconciled_forecasts = evaluate_model(test_df, reconciled_forecasts)
        evaluation_results = evaluate_model(test_df, forecast_df)

        logging.info("Evaluation results completed successfully!")
        logging.info(
            f"Evaluation Results: Forecast RMSE = {evaluation_results}, Reconciled RMSE = {reconciled_forecasts}"
        )
        return 0
    except Exception as e:
        logging.exception("Exception occurred during output evaluation")
        logging.exception(e)
        return 1


if __name__ == "__main__":
    main()
