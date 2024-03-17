import math

import numpy as np
import pandas as pd

np.random.seed(123)

N = 100
F = 5

data = pd.DataFrame(
    np.random.uniform(-1, 2, size=(N, F)).round(4),
    columns=[f"Feature_{i}" for i in range(1, F + 1)],
)


# Create a "Demand" column that generates demands (with expected value of 100) based on the features given above
# The features are independent variables and the demands are dependent variables
# this data will be used for ML training
# Use a nonlinear relationship between features and demand
# Define the polynomial function
def polynomial_function(row):
    fs = np.array([row[f"Feature_{i+1}"] for i in range(F)])

    val = (
        2
        + 0.3 * fs[0]
        + 0.5 * fs[1] ** 3
        + 0.7 * fs[2] * fs[3]
        + 0.9 * (fs[1] + fs[3])
        + math.sin(fs[4])
    )

    err = np.random.normal(4, 2)

    return round(12 * val + err)


# Generate demands based on the features using the polynomial function
data["Demand"] = data.apply(polynomial_function, axis=1)


data.index = pd.RangeIndex(1, len(data) + 1)
data.index.name = "Observation"

data.to_csv("./modelling/part-4/nv_hist_data_100.csv", index=True)
