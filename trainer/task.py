import logging
import os

import pandas as pd
import tensorflow as tf

from trainer.utils import DATA_DIRECTORY
from trainer.utils import prepare_training_data
from trainer.model import model

# Define column names for the data sets.
COLUMNS = ["city_id", "restaurant_id", "payment_id", "transmission_id",
           "platform_id", "order_hour", "is_failed", "voucher_amount",
           "delivery_fee", "amount_paid", "customer_order_rank",
           "is_returning_customer"]
LABEL_COLUMN = 'is_returning_customer'
CATEGORICAL_COLUMNS = ["city_id", "restaurant_id", "payment_id",
                       "transmission_id", "platform_id"]
CONTINUOUS_COLUMNS = ["order_hour", "is_failed", "voucher_amount",
                      "delivery_fee", "amount_paid", "customer_order_rank"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Prepare dataset if not already prepared
if not os.path.exists(os.path.join(DATA_DIRECTORY, "train_data.csv")):
    prepare_training_data(DATA_DIRECTORY)
logger.info("Training the model")

# Read the training and test data sets into Pandas dataframe.
train_file = os.path.join(DATA_DIRECTORY, "train_data.csv")
test_file = os.path.join(DATA_DIRECTORY, "test_data.csv")
df_train = pd.read_csv(
    train_file, names=COLUMNS, skipinitialspace=True, skiprows=1, index_col=0
)
df_test = pd.read_csv(
    test_file, names=COLUMNS, skipinitialspace=True, skiprows=1, index_col=0
)


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k)
    # to the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = {**continuous_cols, **categorical_cols}
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_input_fn():
    return input_fn(df_train)


def eval_input_fn():
    return input_fn(df_test)


model.train(input_fn=train_input_fn, steps=500)
results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
