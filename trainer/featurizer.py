import tensorflow as tf

# Categorical columns
city_id = tf.feature_column.categorical_column_with_hash_bucket(
    "city_id", hash_bucket_size=100, dtype=tf.int64
)
restaurant_id = tf.feature_column.categorical_column_with_hash_bucket(
    "restaurant_id", hash_bucket_size=100, dtype=tf.int64
)
transmission_id = tf.feature_column.categorical_column_with_vocabulary_list(
    "transmission_id", vocabulary_list=[212, 1988, 2020, 4196, 4228, 4260, 4324,
                                        4356, 4996, 21124]
)
payment_id = tf.feature_column.categorical_column_with_vocabulary_list(
    "payment_id", vocabulary_list=[1491, 1523, 1619, 1779, 1811]
)
platform_id = tf.feature_column.categorical_column_with_vocabulary_list(
    "platform_id", vocabulary_list=[525, 22167, 22263, 22295, 29463, 29495,
                                    29751, 29815, 30135, 30199, 30231, 30359,
                                    30391, 30423]
)

# Continuous columns
voucher_amount = tf.feature_column.numeric_column(
    "voucher_amount", dtype=tf.float64
)
delivery_fee = tf.feature_column.numeric_column(
    "delivery_fee", dtype=tf.float64
)
amount_paid = tf.feature_column.numeric_column(
    "amount_paid", dtype=tf.float64
)
customer_order_rank = tf.feature_column.numeric_column(
    "customer_order_rank", dtype=tf.int64
)
is_failed = tf.feature_column.numeric_column("is_failed", dtype=tf.int64)
order_hour = tf.feature_column.numeric_column("order_hour", dtype=tf.int64)

wide_columns = [city_id, restaurant_id, transmission_id, payment_id,
                platform_id,
                tf.feature_column.crossed_column([payment_id, platform_id],
                                                 hash_bucket_size=1000)]
deep_columns = [voucher_amount, delivery_fee, amount_paid, order_hour,
                customer_order_rank, is_failed,
                tf.feature_column.embedding_column(city_id, dimension=8),
                tf.feature_column.embedding_column(restaurant_id, dimension=8),
                tf.feature_column.embedding_column(platform_id, dimension=8),
                tf.feature_column.embedding_column(transmission_id,
                                                   dimension=8),
                tf.feature_column.embedding_column(payment_id, dimension=8)]
