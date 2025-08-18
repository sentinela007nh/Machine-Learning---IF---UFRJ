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

