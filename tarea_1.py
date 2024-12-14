import numpy as np
import matplotlib.pyplot as plt
from perceptron_base import Perceptron, plot_seperation_line



class Simple_Perceptron(Perceptron):
    """
    Simple perceptron with hard limit activation function, 
    we inherit from Perceptron class the init, fit and predict function
    """
    def activation_function(self, x):
        # Hard limit activation function, if x >= 0 return 1, else 0
        return np.where(x >= 0, 1, 0)



if __name__ == '__main__':
    # TAREA a) Train a perceptron with hard limit function

    # indexes of our linear problem
    X_train = np.array([[1, 5], [3, 3], [5, 2], [3, 9], [6, 9], [8, 7]])
    # expected output, 0 == A, 1 == B
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # init perceptron and train with training data
    perceptron = Simple_Perceptron()
    perceptron.fit(X_train, y_train)


    plot_seperation_line(perceptron.weights, perceptron.bias, X_train, y_train)


    # TAREA b) Predict the class of the following points

    X_test = np.array([[4,4], [5,7]])
    y_test = perceptron.predict(X_test)

    print(f"Predictions for X_test: {X_test} are {y_test}")


