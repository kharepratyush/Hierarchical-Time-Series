"""
Author: Pratyush Khare
"""

import pandas as pd
from pandas.errors import EmptyDataError
from .custom_exception import HTSCustomException


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a Pandas DataFrame.

    Parameters:
    filepath (str): Path of the file to be loaded as DataFrame.

    Returns:
    pd.DataFrame: Loaded DataFrame from the CSV file.

    Raises:
    LTRCustomException: If the DataFrame is empty or any other error occurs during loading.
    """
    try:
        df = pd.read_csv(filepath)

        if df.empty:
            raise HTSCustomException(f"Dataframe at {filepath} is empty.")

    except EmptyDataError:
        raise HTSCustomException(f"Dataframe at {filepath} is empty.")
    except FileNotFoundError:
        raise HTSCustomException(f"File not found: {filepath}")
    except Exception as e:
        raise HTSCustomException(f"An error occurred while loading the data: {str(e)}")

    return df
