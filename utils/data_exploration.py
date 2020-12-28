#%%
import gzip

import pandas as pd
import seaborn as sns


def get_extracted_dataframe(path: str) -> pd.DataFrame:
    with gzip.open(path, "rb") as file:
        return pd.read_csv(file)


def get_num_nans(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()


def get_num_duplicated(df: pd.DataFrame) -> pd.Series:
    return df.duplicated().sum()


def plot_class_distribution(df: pd.DataFrame, class_name: str) -> None:
    sns.displot(df, x=class_name)


#%%
data = get_extracted_dataframe("/Users/mu378ws/PycharmProjects/hesitant-bare-wreck/data/"
                               "machine_learning_challenge_labeled_data.csv.gz")
plot_class_distribution(data, "is_returning_customer")
