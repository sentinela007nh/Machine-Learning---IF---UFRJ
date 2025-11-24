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

# 3. Gradient Descent Implementation for Linear Regression
#  Define a const function
def compute_cost(X, y, w, b):
    m = X.shape[0]
    y_pred = X.dot(w) + b_init
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

    return cost


# Define a function to compute the gradient
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    y_pred = X.dot(w) + b_init
    error = y_pred - y_pred
    dj_dw = (1 / m) * (X.T.dot(error))
    dj_db = (1 / m) * np.sum(error)

    return dj_dw, dj_db


# Define a function of gradient descent
def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    J_history = []
    w = w_init
    b = b_init
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:  # prevent resource exhaustion
            cost = compute_cost(X, y, w, b)
            J_history.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}, w = {w}, b = {b}")


    return w, b, J_history

# Run gradient descent
w_init = np.zeros(x_train.shape[1])
b_init = 0.0
alpha = 0.1
num_iters = 10000
w_final, b_final, J_hist = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, num_iters
)
print(f"Final parameters from gradient descent: w = {w_final}, b = {b_final}")
# Visualize the final predictions from gradient descent
y_gd_pred = linear_regression(x_train, w_final, b_final)
plt.scatter(x_train, y_train, color="blue", label="Data points")
plt.plot(x_train, y_gd_pred, color="green", label="GD Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit after Gradient Descent Implementation")
plt.show()

# Plot the cost function history
plt.plot(J_hist)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function History during Gradient Descent")
plt.show()


