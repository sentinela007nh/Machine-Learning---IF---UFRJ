# We will calculate the linear regression of a dataset using the least squares method. #


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
# plt.legend()

# 1.b) Construa uma função que recebe um vetor numpy de preditores ("features"), os parâmetros "w" e "b" e produz um vetor de previsões "y".

# def f_wb(x, w, b):
#    """
#    Computes the prediction of a linear model
#    Args:
#      x (ndarray (m,)): input values, m examples
#      w,b (scalar)    : model parameters
#    Returns
#      y (ndarray (m,)): target values
#    """
#    # INSIRA SEU CÓDIGO AQUI
#    # ...
#    return y
