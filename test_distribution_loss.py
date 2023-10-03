import tensorflow as tf
from distribution_loss import distribution_loss  # Ensure to import your function


def test_ideal_predictions():
    # Test Case: Ideal predictions for dice with 2 sides
    y_true = tf.keras.utils.to_categorical([[0], [1]], num_classes=10)
    y_pred = tf.constant([[0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]])
    loss_value = distribution_loss(y_true, y_pred)
    assert tf.math.equal(
        loss_value, 0.0), f"Expected 0.0, but got {loss_value}"


def test_worst_case_scenario():
    # Test Case: Worst case scenario for dice with 10 sides
    # All predictions are 100% confident in an incorrect number
    y_true = tf.keras.utils.to_categorical(
        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], num_classes=10)
    y_pred = tf.constant([
        [0.01, 0.99, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.01, 0, 0.99, 0, 0, 0, 0, 0, 0, 0],
        [0.01, 0, 0, 0.99, 0, 0, 0, 0, 0, 0],
        [0.01, 0, 0, 0, 0.99, 0, 0, 0, 0, 0],
        [0.01, 0, 0, 0, 0, 0.99, 0, 0, 0, 0],
        [0.01, 0, 0, 0, 0, 0, 0.99, 0, 0, 0],
        [0.01, 0, 0, 0, 0, 0, 0, 0.99, 0, 0],
        [0.01, 0, 0, 0, 0, 0, 0, 0, 0.99, 0],
        [0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0.99],
        [0.99, 0, 0, 0, 0, 0, 0, 0, 0, 0.01],
    ], dtype=tf.float32)
    loss_value = distribution_loss(y_true, y_pred)
    assert tf.math.greater(
        loss_value, 0.95), f"Expected loss greater than 0, but got {loss_value}"


def test_random_predictions():
    # Test Case: Garden variety predictions for dice with 5 sides
    y_true = tf.keras.utils.to_categorical(
        [[0], [1], [2], [3], [4]], num_classes=10)
    y_pred = tf.constant([
        [0.1, 0.2, 0.1, 0.3, 0.2, 0, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.2, 0.1, 0.3, 0.2, 0, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.2, 0.1, 0.3, 0.2, 0, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.2, 0.1, 0.3, 0.2, 0, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.2, 0.1, 0.3, 0.2, 0, 0.1, 0.1, 0.1, 0.1],
    ])  # truncated for brevity
    loss_value = distribution_loss(y_true, y_pred)
    assert tf.math.greater(
        loss_value, 0.1), f"Expected loss greater than 0, but got {loss_value}"
