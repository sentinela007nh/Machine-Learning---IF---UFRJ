# Photo Analizer of Redshift #


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# Load the first CSV file (assumes it contains the test data)
data = pd.read_csv("./des_match_vvds_clean.csv")

# The 'z' columns is the target and the 'mag_*' and 'mag_err_' columns are the predictor
y = data["z"]
X = data[
    [
        "mag_auto_g_dered",
        "mag_auto_r_dered",
        "mag_auto_i_dered",
        "mag_auto_z_dered",
        "mag_auto_y_dered",
        "magerr_auto_g",
        "magerr_auto_r",
        "magerr_auto_i",
        "magerr_auto_z",
        "magerr_auto_y",
    ]
]

# Split the data into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 1. Multi-Layer Perceptron

mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
# Evaluate MLP performance
mlp_mae = mean_absolute_error(y_test, y_pred_mlp)
mlp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
print(f"MLP MAE: {mlp_mae}, MLP RMSE: {mlp_rmse}")


# 2. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
# Evaluate Random Forest performance
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest MAE: {rf_mae}, Random Forest RMSE: {rf_rmse}")

# 3. Decision tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
# Evaluate Decision Tree performance
dt_mae = mean_absolute_error(y_test, y_pred_dt)
dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
print(f"Decision Tree MAE: {dt_mae}, Decision Tree RMSE: {dt_rmse}")

# Compare the models
results = pd.DataFrame({
    "Model": ["MLP", "Random Forest", "Decision Tree"],
    "MAE": [mlp_mae, rf_mae, dt_mae],
    "RMSE": [mlp_rmse, rf_rmse, dt_rmse]
})
print(results)
