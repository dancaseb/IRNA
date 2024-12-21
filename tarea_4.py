import numpy as np
from tarea_3 import Perceptron_SIGMOID
import random
import matplotlib.pyplot as plt


class ANN:
    def __init__(self, n_iter=1000, alpha=0.01):
        self.n_iter = n_iter
        self.alpha = alpha
        self.hidden_layer = [Perceptron_SIGMOID(n_iter=self.n_iter, alpha=self.alpha) for _ in range(2)]
        self.output_layer = Perceptron_SIGMOID(n_iter=self.n_iter, alpha=self.alpha)

    def forward(self, x_i):
        # Forward pass
        hidden_layer_outputs = []
        for perceptron in self.hidden_layer:
            sum_w_x_hidden = np.dot(perceptron.weights, x_i) + perceptron.bias
            hidden_layer_output = perceptron.activation_function(sum_w_x_hidden)
            hidden_layer_outputs.append(hidden_layer_output)

        sum_w_x_output = np.dot(self.output_layer.weights, hidden_layer_outputs) + self.output_layer.bias
        y_pred = self.output_layer.activation_function(sum_w_x_output)
        return y_pred, hidden_layer_outputs

    def backward(self, x_i, y_train, y_pred, hidden_layer_outputs):
        # Backward pass
        output_error = y_train - y_pred
        output_delta = output_error * y_pred * (1 - y_pred)

        hidden_errors = output_delta * self.output_layer.weights
        hidden_deltas = hidden_errors * np.array(hidden_layer_outputs) * (1 - np.array(hidden_layer_outputs))

        # Update output layer weights and bias
        self.output_layer.weights += self.alpha * output_delta * np.array(hidden_layer_outputs)
        self.output_layer.bias += self.alpha * output_delta

        # Update hidden layer weights and biases
        for i, perceptron in enumerate(self.hidden_layer):
            perceptron.weights += self.alpha * hidden_deltas[i] * x_i
            perceptron.bias += self.alpha * hidden_deltas[i]

    def fit(self, X_train, y_train):
        np.random.seed(3)
        # Initialize weights and biases for hidden layer
        for perceptron in self.hidden_layer:
            perceptron.weights = np.random.randn(X_train.shape[1])
            perceptron.bias = np.random.randn()

        # Initialize weights and biases for output layer
        self.output_layer.weights = np.random.randn(len(self.hidden_layer))
        self.output_layer.bias = np.random.randn()

        # Learning loop
        for _ in range(self.n_iter):
            for i, x_i in enumerate(X_train):
                # Forward pass
                y_pred, hidden_layer_outputs = self.forward(x_i)
                # Backward pass, adjusting weights and biases
                self.backward(x_i, y_train[i], y_pred, hidden_layer_outputs)

    def predict(self, X):
        """
        Prediction function. Predicts the class of the input data X
        """
        predictions = []
        # to show that the model converts a non lineal problem to a linear one
        hidden_outputs = []
        for x_i in X:
            y_pred, hidden_layer_outputs = self.forward(x_i)
            predictions.append(y_pred)
            hidden_outputs.append(hidden_layer_outputs)
        return np.array(predictions), np.array(hidden_outputs)

if __name__ == '__main__':
    # TAREA a) Train an ANN with one hidden layer with 2 sigmoid perceptrons and one output layer using BP rule

    # indexes of our linear problem
    # expected output, 0 == A, 1 == B, the interval of sigmoid is [0,1], we will use this interval
    # if output is less than 0.5 class is A, if greater than 0.5 class is B
    X_train = [[1, 6], [3, 3], [2, 8], [6, 3], [5, 10], [4, 6], [7, 5], [7, 8], [9, 5], [9, 8]]
    y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    # shuffled_data = list(zip(X_train, y_train))
    # random.shuffle(shuffled_data)
    # X_train, y_train = zip(*shuffled_data)
    
    # init ANN and train with training data
    ann = ANN(n_iter=5000, alpha=0.01)
    ann.fit(np.array(X_train), np.array(y_train))

    # TAREA b) Predict the class of the following points

    X_test = np.array([[2, 6], [5, 7]])
    y_test, _ = ann.predict(X_test)

    print(f"Predictions for X_test: {X_test} are {y_test}")

    # Tarea c) Show that the hidden layer converts the from non linear to linear
    _, hidden_outputs_train = ann.predict(X_train)
    plt.scatter(hidden_outputs_train[:, 0], hidden_outputs_train[:, 1], c=y_train, cmap='bwr')
    plt.xlabel('Hidden Neuron 1 Output')
    plt.ylabel('Hidden Neuron 2 Output')
    plt.title('Hidden Layer Outputs')
    plt.show()