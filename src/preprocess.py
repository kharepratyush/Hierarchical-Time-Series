"""
Author: Pratyush Khare
"""

from typing import Tuple
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load FMCG data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


def load_holidays(file_path: str) -> pd.DataFrame:
    """
    Load holidays data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded holidays data.
    """
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame, holidays: pd.DataFrame = None) -> pd.DataFrame:
    """
    Preprocess FMCG data.

    Args:
        df (pd.DataFrame): Raw data.
        holidays (pd.DataFrame): Holidays data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={"date": "ds", "sales": "y"}, inplace=True)

    if holidays is not None:
        holidays.rename(columns={"date": "ds"}, inplace=True)
        holidays["ds"] = pd.to_datetime(holidays["ds"])
        df = df.reset_index().merge(holidays, on="ds", how="left")
        df["holiday"] = df["holiday"].fillna(0)

    return df


def train_test_split_time_series(
    data_path: str,
    train_path: str,
    test_path: str,
    test_size: float = 0.2,
    save_dataset: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets.

    Args:
        data_path (str): Path to the CSV file containing the data.
        train_path (str): Path to save the train dataset.
        test_path (str): Path to save the test dataset.
        test_size (float, optional): Fraction of data to be used as test set. Defaults to 0.2.
        save_dataset (bool, optional): Whether to save the train and test datasets to files. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test sets.
    """
    df = pd.read_csv(data_path)
    df.sort_values(by="ds", inplace=True)

    ds = sorted(df.ds.unique())
    split_index = int(len(ds) * (1 - test_size))

    train_df = df[df.ds.isin(ds[:split_index])]
    test_df = df[df.ds.isin(ds[split_index:])]

    if save_dataset:
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

    return train_df, test_df


def save_processed_data(df: pd.DataFrame, filepath: str, index: bool = True) -> None:
    """
    Save the processed DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The processed DataFrame to be saved.
        filepath (str): The file path to save the DataFrame as a CSV file.
        index (bool, optional): Whether to include the index in the saved CSV file. Defaults to True.

    Returns:
        None
    """
    df.to_csv(filepath, index=index)
