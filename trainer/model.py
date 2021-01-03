import tempfile
import tensorflow as tf
from trainer.featurizer import deep_columns, wide_columns

model_dir = tempfile.mkdtemp()
model = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
