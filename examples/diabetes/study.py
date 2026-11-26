"""Offline hold-out experiment on the diabetes dataset bundled with sklearn."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def main():
    # Keep raw inputs: ordinary least squares does not require a fitted scaler.
    data = load_diabetes(return_X_y=False, as_frame=True, scaled=False)
    train, holdout, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42
    )
    train_frame = pd.DataFrame(train)
    test_frame = pd.DataFrame(holdout)
    columns = ["bmi", "bp", "s5"]

    model = LinearRegression()
    model.fit(train_frame[columns].to_numpy(), y_train)
    correct = test_frame[columns].to_numpy()
    swapped = test_frame[["bp", "bmi", "s5"]].to_numpy()
    reference = model.predict(correct)
    drifted = model.predict(swapped)

    print(f"Train/test rows: {len(train)}/{len(holdout)}")
    print(f"Matrix shapes: {correct.shape} and {swapped.shape}")
    print(f"MAE, correct order: {mean_absolute_error(y_test, reference):.3f}")
    print(f"MAE, swapped order: {mean_absolute_error(y_test, drifted):.3f}")
    changed = np.count_nonzero(np.abs(reference - drifted) > 1e-8)
    print(f"Predictions changed (absolute difference > 1e-8): {changed}/{len(y_test)}")


if __name__ == "__main__":
    main()
