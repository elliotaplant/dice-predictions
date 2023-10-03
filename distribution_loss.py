import tensorflow as tf


def distribution_loss(y_true: tf.Tensor, y_pred: tf.Tensor, bucket_width: float = 0.05) -> tf.Tensor:
    # Define bucket thresholds
    bucket_ranges = tf.linspace(0.0, 1.0 - bucket_width, int(1 / bucket_width))

    # Initialize a tensor array to store the losses
    losses = tf.TensorArray(dtype=tf.float32, size=int(1 / bucket_width))

    # Loop through each bucket threshold
    def loop_body(i, losses_array):
        bucket_start = bucket_ranges[i]

        # Create a mask of values indicating which predicted probabilities fall within this bucket
        lower_mask = y_pred >= bucket_start
        upper_mask = y_pred < (bucket_start + bucket_width)
        bucket_mask = tf.cast(lower_mask & upper_mask, tf.float32)

        # Count how many predictions in total fall into this bucket
        bucket_prediction_count = tf.reduce_sum(bucket_mask)

        # Calculate the actual occurrences within this bucket
        bucket_actual_count = tf.reduce_sum(bucket_mask * y_true)

        # Calculate the actual rate
        actual_rate = tf.math.divide_no_nan(
            bucket_actual_count, bucket_prediction_count)

        # The target rate for this bucket is the midpoint
        expected_rate = bucket_start + bucket_width / 2

        # Calculate the difference between expected and actual rate
        difference = expected_rate - actual_rate

        # We're going to use the squared difference for the loss.
        loss_value = bucket_prediction_count * tf.square(difference)
        losses_array = losses_array.write(i, loss_value)

        return i + 1, losses_array

    _, losses = tf.while_loop(
        cond=lambda i, _: i < int(1 / bucket_width),
        body=loop_body,
        loop_vars=(0, losses),
        parallel_iterations=10
    )

    # Stack the tensor array to get the tensor of losses
    losses_tensor = losses.stack()

    # Average the losses over all buckets
    return tf.reduce_mean(losses_tensor)
