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

# 1.b) Linear regression using the least squares method
def fitLR(x, w, b):
    
    Fit a linear regression model using the least squares method.
    
    Parameters:
    x (numpy.ndarray): Input features (shape: [m, n])
    w (numpy.ndarray): Weights (shape: [n, 1])
    b (float): Bias term
    
    Returns:
    numpy.ndarray: Predicted values (shape: [m, 1])
    
    return np.dot(x, w) + b
