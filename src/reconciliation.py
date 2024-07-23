"""
Author: Pratyush Khare
"""

import pickle
from typing import Dict, List, Tuple
import collections

import pandas as pd
import numpy as np
import hts


def load_hierarchy_configuration() -> Dict[str, List[str]]:
    """
    Load the hierarchy from the summing_matrix configuration file.

    Returns:
        Dict[str, List[str]]: Hierarchy mapping from parent to children.
    """
    with open("config/summing_matrix", "rb") as file:
        summing_matrix = pickle.load(file)

    hierarchy: Dict[str, List[str]] = {}
    columns = set(summing_matrix.columns).union(summing_matrix.index)

    for parent in columns:
        child_list = [
            child
            for child in columns
            if parent != child
            and child.startswith(f"{parent}/")
            and len(child.replace(f"{parent}/", "").split("/")) == 1
        ]
        if child_list:
            hierarchy[parent] = child_list

    return hierarchy


def prepare_forecast_reconciliation(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, Dict[str, List[str]], collections.OrderedDict, np.ndarray, List[str]
]:
    """
    Prepare data structures required for forecast reconciliation.

    Args:
        df (pd.DataFrame): Dataframe with forecast data.

    Returns:
        tuple: Contains original dataframe, hierarchy, prediction dictionary,
               summation matrix, and summation matrix labels.
    """
    hierarchy = load_hierarchy_configuration()

    tree = hts.hierarchy.HierarchyTree.from_nodes(nodes=hierarchy, df=df)
    sum_mat, sum_mat_labels = hts.functions.to_sum_mat(tree)

    pred_dict = collections.OrderedDict()
    for label in sum_mat_labels:
        pred_dict[label] = pd.DataFrame(data=df[label].values, columns=["yhat"])

    return df, hierarchy, pred_dict, sum_mat, sum_mat_labels


def perform_forecast_reconciliation(
    df: pd.DataFrame, output_prediction_path: str
) -> pd.DataFrame:
    """
    Perform forecast reconciliation and save the reconciled forecast to a CSV file.

    Args:
        df (pd.DataFrame): Dataframe with forecast data.
        output_prediction_path (str): Path to save the reconciled forecast CSV file.

    Returns:
        pd.DataFrame: Reconciled forecast data.
    """
    hts_df, hierarchy, pred_dict, sum_mat, sum_mat_labels = (
        prepare_forecast_reconciliation(df)
    )

    # Perform forecast reconciliation
    revised = hts.functions.optimal_combination(
        pred_dict, sum_mat, method="OLS", mse={}
    )

    revised_forecasts = pd.DataFrame(
        data=revised, index=hts_df.index, columns=sum_mat_labels
    )

    revised_forecasts = revised_forecasts[hts_df.columns]
    revised_forecasts.to_csv(output_prediction_path)

    return revised_forecasts[hts_df.columns]
