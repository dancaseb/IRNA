import numpy as np
import matplotlib.pyplot as plt

def calculate_y_point(x_point, weights, bias):
    # x*w0 + y*w1 + b = 0 is the seperation line
    return (-weights[0] * x_point - bias) / weights[1]


def plot_seperation_line(weights, bias, X_train, y_train):
    # function to plot the seperation line between two classes
    x_points = np.array([0, 10]) # two points to draw the line
    y_points = calculate_y_point(x_points, weights, bias) # calculate the y points
    plt.plot(x_points, y_points)
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train) # plot the 2 class points
    plt.show()


class Perceptron:
    def __init__(self,n_iter=100, alpha=0.01):
        # a perceptron has a learning rate and number of iterations, weights and bias set to none
        self.n_iter = n_iter
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def activation_function(self, x):
        pass

    def fit(self, X_train, y_train):
        """
        Train the perceptron with X_train data and y_train labels.
        Weights and bias are proposed small non zero values.
        """
        self.weights = np.array([0.01, 0.02])
        self.bias = 0.1
        
        # learing loop
        for _ in range(self.n_iter):
            for i, x_i in enumerate(X_train):
                # calculate the sum of weights and input + bias
                sum_w_x = np.dot(self.weights, x_i) + self.bias
                # apply the activation function and get a prediction y value
                y_pred = self.activation_function(sum_w_x)

                # update the weights and bias
                update = self.alpha * (y_train[i] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Our prediction function of our perceptron, we calculate the sum of weights and input + bias
        then use the activation function
        """
        sum_w_x = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_function(sum_w_x)
        return y_pred

