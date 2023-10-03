import tensorflow as tf


def distribution_loss(y_true: tf.Tensor, y_pred: tf.Tensor, bucket_width: float = 0.05) -> tf.Tensor:
    """
    Custom loss function that checks the model's accuracy across its predicted confidence levels.

    For a given confidence level (e.g., 0.7 or 70% sure), over many samples, a specific outcome
    should ideally occur approximately 70% of the time when predicted with that confidence.

    The loss is computed by bucketing the predicted probabilities and comparing
    the actual rates of outcomes to the model's expected rates for each bucket.

    Parameters:
    - y_true: The ground truth values. Shape: [batch_size, 1]. Each value is between 1 and 10.
    - y_pred: The predicted probabilities for each outcome. Shape: [batch_size, 10].

    Returns:
    - A scalar Tensor representing the loss.
    """

    # Define bucket thresholds for the intervals [0, 0.05), [0.05, 0.1), ... , [0.95, 1]
    bucket_ranges = tf.linspace(0.0, 1.0 - bucket_width, int(1 / bucket_width))

    # Initialize loss to zero
    loss = tf.Variable(0.0)

    # Define a mask for the actual outcomes
    actual_outcomes_mask = y_true > 0

    # Loop through each bucket threshold
    for bucket_start in bucket_ranges:
        # Create a mask of True/False values indicating which predicted probabilities fall within this bucket
        result_prediction_bucket_mask = tf.logical_and(y_pred >= bucket_start,
                                                       y_pred < bucket_start + bucket_width)

        # Count how many predicted probabilities fall into this bucket for each outcome
        result_bucket_prediction_count = tf.reduce_sum(
            tf.cast(result_prediction_bucket_mask, tf.float32), axis=0)

        # Sum the number of predictions in the bucket over all dice values
        bucket_prediction_count = tf.reduce_sum(result_bucket_prediction_count)

        # Count how many of those predicted probabilities were correct for each dice value
        result_bucket_actual_count = tf.reduce_sum(tf.cast(tf.logical_and(
            result_prediction_bucket_mask, actual_outcomes_mask), tf.float32), axis=0)

        # Sum the correct predictions over all dice values
        bucket_actual_count = tf.reduce_sum(result_bucket_actual_count)

        # Calculate the actual rate at which each outcome occurred for this bucket
        actual_rate = tf.math.divide_no_nan(
            bucket_actual_count, bucket_prediction_count)

        # The target rate for this bucket is the midpoint of its range
        expected_rate = bucket_start + bucket_width / 2

        # Calculate the distance between the expected rate and the actual rate
        difference = expected_rate - actual_rate

        # Reduce loss to 0 for results within range of bucket
        outside_bucket = tf.where(tf.greater(tf.math.abs(
            difference), bucket_width / 2), difference, 0)

        # Calculate the loss for the entire batch in this bucket
        batch_bucket_loss = tf.square(outside_bucket)

        loss_increment = tf.math.multiply(
            batch_bucket_loss, bucket_prediction_count)

        # Increment the loss by the batch bucket loss times the size of the batch bucket
        loss.assign_add(loss_increment)

    # Return the final computed loss
    return loss
