import pandas as pd
from sklearn.linear_model import LinearRegression

from features import pack


def fit_cost_model():
    history = pd.DataFrame({
        "labor_hours": [1, 4, 1, 3],
        "machine_hours": [1, 1, 4, 4],
    })
    training_matrix = pack(history, ["labor_hours", "machine_hours"])
    model = LinearRegression()
    model.fit(training_matrix, [100, 340, 160, 320])
    return model
