"""Manufacturing quote: labor costs 80/hour; machine time costs 20/hour."""

import pandas as pd

from features import pack
from training import fit_cost_model


model = fit_cost_model()
request = pd.DataFrame({"labor_hours": [2], "machine_hours": [8]})
features = pack(request, ["machine_hours", "labor_hours"])
quote = model.predict(features)
print(quote)
