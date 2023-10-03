# Dice Prediction with Neural Networks

This project involves training a neural network to predict the outcome of dice rolls. Two loss functions are explored for the prediction task: one-hot categorical crossentropy and a custom distribution-based loss. The neural network is designed using TensorFlow and Keras.

## Project Structure

### Main Files:
- `train.py`: The main script to train the neural network models using two different loss functions.
- `distribution_loss.py`: Defines a custom distribution-based loss function.

## How to Run

### Requirements:
- TensorFlow
- Keras
- pandas
- argparse
and many more

You can install these using pip:

```bash
pip install -r requirements.txt
```

This project uses venv, so make sure to activate your venv before installation

### Training the Models:

To train the models with the default number of epochs (50):

```bash
python train.py
```

To train with a custom number of epochs, use the `--num_epochs` flag:

```bash
python train.py --num_epochs 100
```

This will train two models:
1. Using the one-hot categorical crossentropy loss.
2. Using the custom distribution-based loss.

The trained models will be saved in the `models/` directory and the training history, including loss and accuracy metrics, will be saved in the `history/` directory.

## Model Architecture:

The neural network used for both models consists of:
- An input layer with a single neuron for normalized dice sides.
- Two hidden layers, each with 64 neurons and ReLU activation.
- An output layer with 10 neurons (one for each possible dice outcome) and a softmax activation.

## Loss Functions:

1. **One-hot Categorical Crossentropy**:
    - Standard loss for categorical classification tasks.
    - Implemented using TensorFlow's built-in function.

2. **Distribution Loss**:
    - A custom loss designed to measure how well predicted probabilities match the actual occurrences across defined bucket ranges.
    - Refer to `distribution_loss.py` for implementation details.

## Data:

Generate data using `generate_data.py` and specify the number of rows you want

The output should be named `dice_data.csv` and be present in the `data/` directory. The dataset is expected to have two columns:
- `dice_sides`: The number of sides of the dice.
- `outcome`: The outcome of the dice roll (1-10).

## Future Work:

1. Explore other architectures and hyperparameters to improve performance.
2. Implement additional custom loss functions for comparison.
3. Visualize predictions and compare how each loss function affects the predictions.

## License

This project is open source and available under the MIT License.
