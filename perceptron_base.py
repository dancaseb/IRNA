import numpy as np
import matplotlib.pyplot as plt

def calculate_y_point(x_point, weights, bias):
    """
    Calculate the y point of a line given the x point, weights and bias
    according to this formula x*w_0 + y*w_1 + b = 0
    """
    
    return (-weights[0] * x_point - bias) / weights[1]


def plot_seperation_line(weights, bias, X_train, y_train, title="Seperation line"):
    """"
    Function to plot the seperation line between two classes in a graph
    """
    x_points = np.array([0, 10]) # two points to draw the line
    y_points = calculate_y_point(x_points, weights, bias) # calculate the y points
    plt.plot(x_points, y_points)
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train) # plot the 2 class points
    plt.title(title)
    plt.xlabel(f"weights: {weights}, bias: {bias}")
    plt.show()


class Perceptron:
    def __init__(self,n_iter=100, alpha=0.01):
        """
        A perceptron has a learning rate alpha and number of iterations. Weights and bias set to none
        """
        self.n_iter = n_iter
        self.alpha = alpha
        self.weights = None
        self.bias = None
        # Save the initial weights and bias for graph plot
        self.initial_weights = None
        self.initial_bias = None

    def activation_function(self, x):
        """
        The activation function of the perceptron, takes the summed values and returns an output value
        Define in child classes 
        """
        pass

    def fit(self, X_train, y_train):
        """
        Train the perceptron with X_train data and y_train labels.
        Weights and bias are proposed small non zero values.
        """
        np.random.seed(7)
        self.initial_weights = np.random.randn(X_train.shape[1])
        self.initial_bias = np.random.randn()
        self.weights = self.initial_weights
        self.bias = self.initial_bias
        
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

