# We will calculate the linear regression of a dataset using the least squares method.


import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt
# import sklearn

# 1.a) Inspecione os dados fornecidos e confirme que o método da regressão linear é uma boa opção para este conjunto

np.random.seed(42)
m = 300
x_train = 2 * np.random.rand(m, 1)
y_train = 4 + 3 * x_train + np.random.randn(m, 1)

# Visualize the data
plt.scatter(x_train, y_train, color="blue", label="Data points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter plot of training data")
plt.show()


# 1.b) Linear regression algorithm implementation
def linear_regression(X, w, b):
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


# Implement the least squares method to find the optimal parameters w and b


def compute_cost(X, y, w, b):
    m = len(y)
    y_pred = linear_regression(X, w, b)
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost


def compute_gradient(X, Y, w, b):
    m = len(Y)
    y_pred = linear_regression(X, w, b)
    error = y_pred - Y
    dw = (1 / m) * np.dot(X.T, error)
    db = (1 / m) * np.sum(error)
    return dw, db


def gradient_descent(X, Y, w_init, b_init, learning_rate, num_iterations):
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
optimal_w, optimal_b = gradient_descent(
    x_train, y_train, w_init, b_init, learning_rate, num_iterations
)
print(f"Optimal parameters: w = {optimal_w}, b = {optimal_b}")

# Visualize the final predictions
y_final_pred = linear_regression(x_train, optimal_w, optimal_b)
plt.scatter(x_train, y_train, color="blue", label="Data points")
plt.plot(x_train, y_final_pred, color="red", label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit after Gradient Descent")
plt.show()


#  Define a const function
def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """

    return cost


# Define a function to compute the gradient
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    # INSIRA SEU CÓDIGO AQUI
    # ...
    return dj_dw, dj_db


# Define a function of gradient descent
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
    """
    # INSIRA SEU CÓDIGO AQUI
    # ...

    return w, b, J_history
