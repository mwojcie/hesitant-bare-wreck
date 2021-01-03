import glob
import gzip
import logging
import os
from typing import List, Tuple, Any

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Set constants
DATA_DIRECTORY = "../data/"
GROUPBY_COL = "customer_id"
MODE_COLS = [GROUPBY_COL, "payment_id", "transmission_id", "platform_id",
             "order_hour"]
MAX_COLS = [GROUPBY_COL, "customer_order_rank"]
SUM_COLS = [GROUPBY_COL, "is_failed", "voucher_amount", "delivery_fee",
            "amount_paid"]
AGGREGATED_DATA_PATH = os.path.join(DATA_DIRECTORY, "aggregated_order_data.csv")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def get_extracted_dataframe(path: str) -> pd.DataFrame:
    """Helper function to extract gzip archive as pandas DataFrame."""
    with gzip.open(path, "rb") as file:
        logger.info(f"Extracting and reading gzipped .csv from path = '{path}'")
        return pd.read_csv(file)


def get_clean_dataset(order_data: pd.DataFrame) -> pd.DataFrame:
    """Helper function to fill nans with `0` and remove duplicates."""
    logger.info("Checking dataset for null values")
    if order_data.isna().sum().sum() > 0:
        logger.info("Filling null values with `0`")
        order_data.fillna(value=0, inplace=True)
    logger.info("Checking dataset for duplicated entries")
    if order_data.duplicated(keep="first").sum() > 0:
        logger.info("Removing duplicated entries")
        order_data.drop_duplicates(keep="first", inplace=True)
    return order_data


def get_aggregated_order_data(
        order_data: pd.DataFrame,
        cumulative_cols: List[str],
        mode_cols: List[str],
        max_cols: List[str]
) -> pd.DataFrame:
    """This function aggregates order data.

    Take subset of order data and aggregate on `customer_id` by selecting most
    frequent value for that `customer_id`, adding values or selecting max value.
    Merge all intermediate dataframes into one.

    Args:
        order_data (pd.Dataframe): DataFrame with historical orders
        cumulative_cols (list): columns to aggregate by adding values
        mode_cols (list): columns to aggregate by selecting most frequent value
        max_cols (list): columns to aggregate by selecting maximum value

    Returns:
        pd.DataFrame: aggregated order dataset
    """
    logger.info("Aggregating order data to customer level")
    mode_df = order_data[mode_cols].groupby(
        GROUPBY_COL, as_index=False).agg(lambda x: x.value_counts().index[0])
    cumulative_df = order_data[cumulative_cols].groupby(
        GROUPBY_COL, as_index=False).agg("sum")
    max_df = order_data[max_cols].groupby(
        GROUPBY_COL, as_index=False).agg("max")
    aggregated_df = mode_df.merge(cumulative_df, on=GROUPBY_COL).\
        merge(max_df, on=GROUPBY_COL)
    aggregated_df.to_csv(path_or_buf=AGGREGATED_DATA_PATH)
    return aggregated_df


def get_labeled_dataset(
        aggregated_df: pd.DataFrame,
        labeled_df: pd.DataFrame
) -> pd.DataFrame:
    """Helper function joining aggregated dataset with labels."""
    logger.info("Adding labels to aggregated dataset")
    final_df = aggregated_df.merge(labeled_df, on=GROUPBY_COL)
    final_df.drop(labels=GROUPBY_COL, axis=1, inplace=True)
    return final_df


def preprocess_aggregated_dataset(
        aggregated_df: pd.DataFrame
) -> Tuple[Any, Any, Any, Any]:
    """Split and preprocess training set."""
    logger.info("Starting data pre-processing")
    numeric_cols = ["voucher_amount", "delivery_fee", "amount_paid"]
    categorical_cols = ["payment_id", "transmission_id", "platform_id"]
    y = aggregated_df.pop("is_returning_customer")
    logger.info("One-Hot encoding categorical features")
    x = pd.get_dummies(aggregated_df, columns=categorical_cols, drop_first=True)
    logger.info("Splitting data into training and test set")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42, test_size=0.2)
    logger.info("Scaling numeric features")
    numeric_scaler = ColumnTransformer([
        ("numeric_scaler", StandardScaler(), numeric_cols)],
        remainder="passthrough")
    numeric_scaler.fit_transform(x_train)
    numeric_scaler.transform(x_test)
    logger.info("Oversampling training set to balance classes")
    oversampler = SMOTE()
    x_train, y_train = oversampler.fit_resample(x_train, y_train)
    return x_train, x_test, y_train, y_test


def prepare_training_data(
        data_directory_path: str
) -> Tuple[Any, Any, Any, Any]:
    """Match files in specified directory and tie together all the functions."""
    labeled_data = get_extracted_dataframe(
        next(glob.iglob(
            os.path.join(data_directory_path, "*_labeled_data.csv.gz"))
        )
    )
    if not os.path.exists(AGGREGATED_DATA_PATH):
        order_data = get_extracted_dataframe(
            next(glob.iglob(
                os.path.join(data_directory_path, "*_order_data.csv.gz"))
            )
        )
        clean_order_data = get_clean_dataset(order_data)
        aggregated_order_data = get_aggregated_order_data(
            clean_order_data, SUM_COLS, MODE_COLS, MAX_COLS
        )
    else:
        logger.info("Reading previously saved aggregated order dataset")
        aggregated_order_data = pd.read_csv(
            filepath_or_buffer=AGGREGATED_DATA_PATH)
    label_order_data = get_labeled_dataset(aggregated_order_data, labeled_data)
    x_train, x_test, y_train, y_test = \
        preprocess_aggregated_dataset(label_order_data)
    return x_train, x_test, y_train, y_test
