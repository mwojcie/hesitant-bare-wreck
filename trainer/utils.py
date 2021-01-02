import gzip
from typing import List

import pandas as pd

GROUPBY_COL = "customer_id"


def get_extracted_dataframe(path: str) -> pd.DataFrame:
    """Helper function to extract gzip archive as pandas DataFrame."""
    with gzip.open(path, "rb") as file:
        return pd.read_csv(file)


def get_clean_dataset(order_data: pd.DataFrame) -> pd.DataFrame:
    """Helper function to fill nans with `0` and remove duplicates."""
    if order_data.isna().sum().sum() > 0:
        order_data.fillna(value=0, inplace=True)
    if order_data.duplicated(keep="first").sum() > 0:
        order_data.drop_duplicates(keep="first", inplace=True)
    return order_data


def get_aggregated_order_data(order_data: pd.DataFrame,
                              cumulative_cols: List[str],
                              mode_cols: List[str],
                              max_cols: List[str]
                              ) -> pd.DataFrame:
    """This function aggregates order data.

    Take subset of order data and aggregate on `customer_id` by selecting most
    frequent value for that `customer_id`, addition or selecting max value.

    Args:
        order_data (pd.Dataframe): DataFrame with historical orders
        cumulative_cols (list): columns to aggregate by adding values
        mode_cols (list): columns to aggregate by selecting most frequent value
        max_cols (list): columns to aggregate by selecting maximum value

    Returns:
        pd.DataFrame: aggregated order dataset
    """
    mode_df = order_data[mode_cols].groupby(GROUPBY_COL).agg(
        lambda x: x.value_counts().index[0])
    cumulative_df = order_data[cumulative_cols].groupby(GROUPBY_COL).agg("sum")
    max_df = order_data[max_cols].groupby(GROUPBY_COL).agg("max")
    aggregated_df = mode_df.merge(cumulative_df, on=GROUPBY_COL).\
        merge(max_df, on=GROUPBY_COL)
    return aggregated_df
