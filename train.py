import tensorflow as tf
import pandas as pd
import argparse
from distribution_loss import distribution_loss


def one_hot_loss():
    return tf.keras.losses.CategoricalCrossentropy()


def build_model(loss_function):
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(1,)),
        # 10 classes for outcomes 1 to 10
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    return model


def main(num_epochs: int, loss_function: str):
    # Load dataset
    data = pd.read_csv('data/dice_data.csv')

    # Normalize dice_sides feature
    data['dice_sides'] = data['dice_sides'] / 10.0

    # One-hot encode the outcome
    data['outcome'] -= 1  # Shift outcomes to 0-9
    y = tf.keras.utils.to_categorical(data['outcome'], num_classes=10)

    # Split the data into training and validation sets
    train_indices = data.sample(frac=0.8, random_state=0).index
    val_indices = data.drop(train_indices).index

    train_data = data.loc[train_indices]
    val_data = data.loc[val_indices]

    train_labels = y[train_indices]
    val_labels = y[val_indices]

    # Build and train the model
    loss_fn = one_hot_loss() if loss_function == 'one_hot' else distribution_loss
    model = build_model(loss_fn)
    history = model.fit(
        train_data['dice_sides'], train_labels,
        epochs=num_epochs,
        batch_size=1000,
        validation_data=(val_data['dice_sides'], val_labels)
    )

    # You may want to save the model and/or the training history
    model.save(f'model_{loss_function}.keras')
    pd.DataFrame(history.history).to_csv(
        f'history_{loss_function}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network with specified loss function")
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epochs')
    args = parser.parse_args()

    main(args.num_epochs, 'distribution')
    main(args.num_epochs, 'one_hot')
