import numpy as np
import matplotlib.pyplot as plt
from perceptron_base import Perceptron, plot_seperation_line

class Perceptron_SIGMOID(Perceptron):
    def activation_function(self, x):
        # Linear activation function
        return 1 / (1 + np.exp(-x))

    

if __name__ == '__main__':
    # TAREA a) Train a perceptron with with sigmoid function and delta rule 

    # indexes of our linear problem
    X_train = np.array([[1, 5], [3, 3], [5, 2], [3, 9], [6, 9], [8, 7]])
    # expected output, 0 == A, 1 == B, the interval of sigmoid is [0,1], we will use this interval
    # if output is less than 0.5 class is A, if greater than 0.5 class is B
    y_train = np.array([0,0,0, 1, 1, 1])

    # init perceptron and train with training data
    perceptron = Perceptron_SIGMOID(n_iter=1000, alpha=0.01)
    perceptron.fit(X_train, y_train)

    plot_seperation_line(perceptron.initial_weights, perceptron.initial_bias, X_train, y_train)

    plot_seperation_line(perceptron.weights, perceptron.bias, X_train, y_train)


    # TAREA b) Predict the class of the following points

    X_test = np.array([[4,4], [5,7]])
    y_test = perceptron.predict(X_test)

    print(f"Predictions for X_test: {X_test} are {y_test}")
    