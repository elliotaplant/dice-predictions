import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def plot_losses():
    history_one_hot = pd.read_csv('history/history_one_hot.csv')
    history_distribution = pd.read_csv('history/history_distribution.csv')

    plt.figure(figsize=(12, 6))

    plt.plot(history_one_hot['loss'], label='Training Loss (One-Hot)')
    plt.plot(history_one_hot['val_loss'],
             linestyle='--', label='Validation Loss (One-Hot)')

    plt.plot(history_distribution['loss'],
             label='Training Loss (Distribution)')
    plt.plot(history_distribution['val_loss'],
             linestyle='--', label='Validation Loss (Distribution)')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()


def plot_predictions():
    model_one_hot = tf.keras.models.load_model('models/model_one_hot.keras')
    model_distribution = tf.keras.models.load_model(
        'models/model_distribution.keras')

    dice_sizes = np.arange(1, 11) / 10.0  # Normalized
    predictions_one_hot = model_one_hot.predict(dice_sizes)
    predictions_distribution = model_distribution.predict(dice_sizes)

    for i, dice_size in enumerate(dice_sizes):
        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(1, 11),
                predictions_one_hot[i], alpha=0.6, label='One-Hot Loss')
        plt.bar(np.arange(
            1, 11), predictions_distribution[i], alpha=0.6, label='Distribution Loss', width=0.4)
        plt.xlabel('Outcome')
        plt.ylabel('Predicted Probability')
        plt.title(f'Predicted Probabilities for Dice Size {dice_size*10}')
        plt.xticks(np.arange(1, 11))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    plot_losses()
    plot_predictions()
