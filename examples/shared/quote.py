"""Either schema is valid here: the same choice reaches fit and predict."""

import sys

import pandas as pd
from sklearn.linear_model import LinearRegression


if len(sys.argv) > 1:
    columns = ["machine_hours", "labor_hours"]
else:
    columns = ["labor_hours", "machine_hours"]

data = pd.DataFrame({"labor_hours": [1, 4, 1, 3], "machine_hours": [1, 1, 4, 4]})
request = pd.DataFrame({"labor_hours": [2], "machine_hours": [8]})
model = LinearRegression()
model.fit(data[columns].to_numpy(), [100, 340, 160, 320])
quote = model.predict(request[columns].to_numpy())
print(quote)
