
## Fashion MNIST CNN Classification

This repository (or .ipynb file) demonstrates a 6-layer Convolutional Neural Network (CNN) built with Keras to classify items in the Fashion MNIST dataset. The code trains the model, evaluates accuracy on a test set, and makes predictions for the first two images in the test set.

## Requirements
Python 3.7+

TensorFlow 2.x (or later, which includes Keras)

NumPy

Example installation (via pip):


pip install tensorflow numpy
## Running the program
Download/clone this repository or place cnn_fashion_mnist.ipynb in your working directory.

Open cnn_fashion_mnist.ipynb in Jupyter (or another Python environment that supports notebooks).

Run all cells in order:

The dataset is automatically loaded from tensorflow.keras.datasets.fashion_mnist.

The model is trained for 5 epochs (you can adjust this).

The script evaluates on the test set and prints final accuracy.

Predictions for the first two test images are displayed.
## Command-Line Option (NBConvert)
Alternatively, you can convert the notebook to a Python script and run it:


jupyter nbconvert --to script cnn_fashion_mnist.ipynb
python cnn_fashion_mnist.py
## What the Code Does
Imports the Fashion MNIST data (x_train, y_train, x_test, y_test)

Reshapes and normalizes the images from [0, 255] to [0, 1] and shape (28, 28, 1).

Defines a 6-layer CNN with:

2 Convolution layers (Conv2D)

1 MaxPooling layer

1 Flatten layer

2 Dense (Fully Connected) layers (one is output with 10 classes)

Compiles the model using adam optimizer and sparse_categorical_crossentropy loss.

Trains the model for 5 epochs, printing training/validation metrics.

Evaluates model accuracy on the test set.

Predicts classes for the first two images in x_test.
## Sample Output
You will see logs similar to:


...
Test accuracy: 0.88
Raw prediction outputs:
 [[...]]
Predicted classes for first two images:  [9 2]
Actual classes for first two images:     [9 2]
## Notes
You can increase epochs to improve accuracy.

Adjust batch sizes, layers, or dropout for experimentation.

Ensure you have a GPU-enabled TensorFlow installation if you want faster training.