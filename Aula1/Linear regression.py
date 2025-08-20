# We will calculate the linear regression of a dataset using the least squares method.


import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt
# import sklearn

# 1.a) Inspecione os dados fornecidos e confirme que o método da regressão linear é uma boa opção para este conjunto

np.random.seed(42)  # to make this code example reproducible
m = 300  # number of instances
x_train = 2 * np.random.rand(m, 1)  # column vector
y_train = 4 + 3 * x_train + np.random.randn(m, 1)  # column vector

# Visualize the data
plt.scatter(x_train, y_train, color="blue", label="Data points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter plot of training data")
plt.show()


# 1.b) Linear regression algorithm implementation
def linear_regression(X, w, b):
    """
    Calculate the predictions using the linear regression model.

    Parameters:
    X : np.ndarray
        Input features (m x n matrix).
    w : float
        Weights.
    b : float
        Bias term.

    Returns:
    np.array
        Predictions (m x 1 vector).
    """
    y = X.dot(w) + b
    return y


w = float(input("Enter the weight (w): "))
b = float(input("Enter the bias (b): "))

y_pred = linear_regression(x_train, w, b)

# Visualize the predictions
plt.scatter(x_train, y_train, color="blue", label="Data points")
plt.plot(x_train, y_pred, color="red", label="Linear regression line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Predictions")
plt.show()

# End of first part of the code

# 2.a) Implement the least squares method to find the optimal parameters w and b


def compute_cost(X, y, w, b):
    """
    Compute the cost function for linear regression.

    Parameters:
    X : array
        Input features (vector).
    y : array
        True values (vector).
    w : float
        Weights.
    b : float
        Bias term.

    Returns:
    float
        Cost value.
    """
    m = len(y)
    y_pred = linear_regression(X, w, b)
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost


def compute_gradient(X, Y, w, b):
    """
    Compute the gradient of the cost function.

    Parameters:
    X : array
        Input features (vector).
    y : array
        True values (vector).
    w : float
        Weights.
    b : float
        Bias term.

    Returns:
    tuple
        Gradient with respect to w and b.
    """
    m = len(Y)
    y_pred = linear_regression(X, w, b)
    error = y_pred - Y
    dw = (1 / m) * np.dot(X.T, error)
    db = (1 / m) * np.sum(error)
    return dw, db


def gradient_descent(X, Y, w_init, b_init, learning_rate, num_iterations):
    """
    Perform gradient descent to find the optimal parameters w and b.

    Parameters:
    X : array
        Input features (vector).
    Y : array
        True values (vector).
    w_init : float
        Initial weight.
    b_init : float
        Initial bias term.
    learning_rate : float
        Learning rate for gradient descent.
    num_iterations : int
        Number of iterations for gradient descent.
    Returns:
        w : float
            Optimal weight.
        b : float
            Optimal bias term.
    """
    w = w_init
    b = b_init
    for i in range(num_iterations):
        dw, db = compute_gradient(X, Y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 100 == 0:  # Print cost every 100 iterations
            cost = compute_cost(X, Y, w, b)
            print(f"Iteration {i}: Cost = {cost}, w = {w}, b = {b}")
    return w, b

# Run the analysis
w_init = 0.0
b_init = 0.0
learning_rate = 0.1
num_iterations = 1000
optimal_w, optimal_b = gradient_descent(x_train, y_train, w_init, b_init, learning_rate, num_iterations)
print(f"Optimal parameters: w = {optimal_w}, b = {optimal_b}")
# Visualize the final predictions
y_final_pred = linear_regression(x_train, optimal_w, optimal_b)
plt.scatter(x_train, y_train, color="blue", label="Data points")
plt.plot(x_train, y_final_pred, color="red", label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit after Gradient Descent")
plt.show()

