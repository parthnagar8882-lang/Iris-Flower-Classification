"""Flask web interface for the saved Iris KNN classifier."""

import json
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "knn_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_FIELDS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

app = Flask(__name__)


def load_artifacts():
    """Load saved artifacts without ever retraining the model."""
    missing_files = [
        str(path.name) for path in (MODEL_PATH, SCALER_PATH, METRICS_PATH) if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing_files) + ". "
            "Run 'python train_model.py' first."
        )

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), metrics


def get_form_values():
    return {field: request.form.get(field, "").strip() for field in FEATURE_FIELDS}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = {field: "" for field in FEATURE_FIELDS}

    try:
        model, scaler, metrics = load_artifacts()
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exception:
        return render_template(
            "index.html", metrics=None, result=None, error=str(exception), form_values=form_values
        )

    if request.method == "POST":
        form_values = get_form_values()
        try:
            if any(value == "" for value in form_values.values()):
                raise ValueError("Please enter a value for all four measurements.")

            values = [float(form_values[field]) for field in FEATURE_FIELDS]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Measurements must be valid finite numbers.")
            if any(value < 0 for value in values):
                raise ValueError("Measurements cannot be negative.")

            # A one-row DataFrame preserves the exact training feature order.
            input_data = pd.DataFrame([values], columns=FEATURE_FIELDS)
            scaled_input = scaler.transform(input_data)
            result = model.predict(scaled_input)[0]
        except ValueError as exception:
            error = str(exception) or "Please enter valid numeric measurements."
        except Exception:
            error = "Unable to make a prediction. Please check the input values and try again."

    return render_template(
        "index.html", metrics=metrics, result=result, error=error, form_values=form_values
    )


if __name__ == "__main__":
    app.run(debug=True)
