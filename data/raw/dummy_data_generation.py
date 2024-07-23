"""
Author: Pratyush Khare
"""

import random
from datetime import datetime

import numpy as np
import pandas as pd

# Define the hierarchical levels
# countries = ["Country A", "Country B", "Country C"]
countries = ["total"]
states = ["State 1", "State 2", "State 3"]
divisions = ["Division X", "Division Y", "Division Z"]
districts = ["District I", "District II", "District III"]
zones = ["Zone P", "Zone Q", "Zone R"]
routes = ["Route 1", "Route 2", "Route 3"]

# Define the date range for data generation
s_date: datetime = datetime(2020, 1, 1)
e_date: datetime = datetime(2023, 12, 31)
dt_range = pd.date_range(start=s_date, end=e_date, freq="D")


# Generate holidays data
def generate_holidays(
    start_date: datetime, end_date: datetime, frequency: str = "W-MON"
) -> pd.DataFrame:
    """
    Generate a DataFrame of holiday dates within a given range.

    Args:
        start_date (datetime): The start date of the range.
        end_date (datetime): The end date of the range.
        frequency (str): The frequency of holidays. Default is 'W-MON' (every Monday).

    Returns:
        pd.DataFrame: DataFrame containing holiday dates.
    """
    holiday_dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    holidays = pd.DataFrame({"date": holiday_dates, "holiday": 1})
    return holidays


# Function to generate base sales for a hierarchical level
def generate_base_sales(level: str) -> int:
    """
    Generate base sales for a given hierarchical level.

    Args:
        level (str): The hierarchical level.

    Returns:
        int: The base sales value.
    """
    sales_range = {
        "country": (10000, 20000),
        "state": (5000, 10000),
        "division": (2000, 5000),
        "district": (1000, 2000),
        "zone": (500, 1000),
        "route": (100, 500),
    }
    return random.randint(*sales_range[level])


# Function to apply seasonality to the sales data
def apply_seasonality(sales: float, date: datetime) -> float:
    """
    Apply seasonality effects to the sales data.

    Args:
        sales (float): The base sales value.
        date (datetime): The date.

    Returns:
        float: The sales value with seasonality applied.
    """
    month = date.month
    if month in [11, 12, 1]:  # High sales season (e.g., holidays)
        sales *= 1.3
    elif month in [6, 7, 8]:  # Low sales season (e.g., summer)
        sales *= 0.7
    return sales


# Function to apply holiday effects to the sales data
def apply_holiday_effects(
    sales: float, date: datetime, holidays: pd.DataFrame
) -> float:
    """
    Apply holiday effects to the sales data.

    Args:
        sales (float): The base sales value.
        date (datetime): The date.
        holidays (pd.DataFrame): DataFrame containing holiday dates.

    Returns:
        float: The sales value with holiday effects applied.
    """
    if date in holidays["date"].values:
        sales *= 1.5
    return sales


# Function to apply day-of-week effects to the sales data
def apply_day_of_week_effects(sales: float, date: datetime) -> float:
    """
    Apply day-of-week effects to the sales data.

    Args:
        sales (float): The base sales value.
        date (datetime): The date.

    Returns:
        float: The sales value with day-of-week effects applied.
    """
    day_of_week = date.weekday()
    if day_of_week in [5, 6]:  # Higher sales during weekends
        sales *= 1.2
    elif day_of_week in [0]:  # Lower sales on Mondays
        sales *= 0.8
    return sales


# Function to generate a trend component for the sales data
def generate_trend(
    start_date: datetime, end_date: datetime, start_value: float, end_value: float
) -> pd.Series:
    """
    Generate a trend component for the sales data.

    Args:
        start_date (datetime): The start date of the range.
        end_date (datetime): The end date of the range.
        start_value (float): The starting value of the trend.
        end_value (float): The ending value of the trend.

    Returns:
        pd.Series: The trend component as a time series.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    trend = np.linspace(start_value, end_value, len(date_range))
    return pd.Series(trend, index=date_range)


# Function to generate random sales data with hierarchical structure
def generate_sales_data(
    date_range: pd.DatetimeIndex, holidays: pd.DataFrame, trend: pd.Series
) -> pd.DataFrame:
    """
    Generate random sales data with hierarchical structure.

    Args:
        date_range (pd.DatetimeIndex): The date range for data generation.
        holidays (pd.DataFrame): DataFrame containing holiday dates.
        trend (pd.Series): The trend component of the sales data.

    Returns:
        pd.DataFrame: The generated sales data.
    """
    data = []
    for country in countries:
        country_base_sales = generate_base_sales("country")
        for state in states:
            state_base_sales = country_base_sales * (0.5 + random.random() * 0.5)
            for zone in zones:
                for date in date_range:
                    sales = state_base_sales * (0.5 + random.random() * 0.5)
                    sales = apply_seasonality(sales, date)
                    # sales = apply_holiday_effects(sales, date, holidays)
                    sales = apply_day_of_week_effects(sales, date)
                    sales += trend[date]
                    data.append(
                        [
                            country,
                            state,
                            zone,
                            date,
                            int(sales),
                        ]
                    )

    columns = [
        "total",
        "state",
        "zone",
        "date",
        "sales",
    ]
    return pd.DataFrame(data, columns=columns)


# Generate holidays
hl_days = generate_holidays(s_date, e_date)

# Generate trend
data_trend = generate_trend(s_date, e_date, start_value=100, end_value=200)

# Generate the dataset
df = generate_sales_data(dt_range, hl_days, data_trend)

# Shuffle the DataFrame to randomize data
df = df.sample(frac=1).reset_index(drop=True)

# Save the generated dataset to a CSV file
df.to_csv("fmcg_data.csv", index=False)
hl_days.to_csv("holiday.csv", index=False)
# Display the first few rows of the dataset
print(df.head())
