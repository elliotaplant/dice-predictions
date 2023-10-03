import tensorflow as tf
import pandas as pd
import argparse


def one_hot_loss(y_true, y_pred):
    pass


def distribution_loss(y_true, y_pred):
    # Create probability buckets of width 0.05 with keys {possible, actual}
    # for index in y_pred:
    #   for value, probability in y_pred[index]:
    #     bucket[probability].possible += 1
    #     if y_true[index] === value:
    #       bucket[probability].actual += 1

    # At this point we now have a comparison of the actual and possible rates for each bucket.
    # The loss should be low if possible/actual is close to the bucket value
    pass


def build_model(loss_function):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        # 10 classes for outcomes 1 to 10
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    return model


def main(num_epochs: int, loss_function: str):
    # Load dataset
    data = pd.read_csv('dice_data.csv')

    # Normalize dice_sides feature
    data['dice_sides'] = data['dice_sides'] / 10.0

    # One-hot encode the outcome
    data['outcome'] -= 1  # Shift outcomes to 0-9
    y = tf.keras.utils.to_categorical(data['outcome'], num_classes=10)

    # Split the data into training and validation sets
    train_data = data.sample(frac=0.8, random_state=0)
    val_data = data.drop(train_data.index)

    # Build and train the model
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
    ) if loss_function == 'one_hot' else distribution_loss
    model = build_model(loss_fn)
    history = model.fit(
        train_data['dice_sides'], train_data['outcome'],
        epochs=num_epochs,
        validation_data=(val_data['dice_sides'], val_data['outcome'])
    )

    # You may want to save the model and/or the training history
    model.save(f'model_{loss_function}.h5')
    pd.DataFrame(history.history).to_csv(
        f'history_{loss_function}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network with specified loss function")
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--loss_function', type=str, choices=[
                        'one_hot', 'distribution'], default='one_hot', help='Loss function to use')
    args = parser.parse_args()

    main(args.num_epochs, args.loss_function)
