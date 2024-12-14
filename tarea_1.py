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

    def __init__(self, n_iter=100, alpha=0.01):
        # a perceptron has a learning rate and number of iterations, weights and bias set to none
        self.n_iter = n_iter
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def hard_limit(self, x):
        # Hard limit activation function, if x >= 0 return 1, else 0
        return np.where(x >= 0, 1, 0)

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
                y_pred = self.hard_limit(sum_w_x)

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
        y_pred = self.hard_limit(sum_w_x)
        return y_pred


if __name__ == '__main__':
    # TAREA a) Train a perceptron with hard limit function

    # indexes of our linear problem
    X_train = np.array([[1, 5], [3, 3], [5, 2], [3, 9], [6, 9], [8, 7]])
    # expected output, 0 == A, 1 == B
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # init perceptron and train with training data
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)


    plot_seperation_line(perceptron.weights, perceptron.bias, X_train, y_train)


    # TAREA b) Predict the class of the following points

    X_test = np.array([[4,4], [5,7]])
    y_test = perceptron.predict(X_test)

    print(f"Predictions for X_test: {X_test} are {y_test}")


