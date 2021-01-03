import tempfile

import tensorflow as tf

from trainer.featurizer import deep_columns, wide_columns

model_dir = tempfile.mkdtemp()
model = tf.estimator.DNNLinearCombinedClassifier(
    config=tf.estimator.RunConfig(tf_random_seed=42),
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[64, 32]
)
