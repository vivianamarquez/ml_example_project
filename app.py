from functools import lru_cache
import os
from pathlib import Path
from uuid import uuid4

import joblib
import matplotlib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, url_for

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cdc_diabetes_top7_best_model.joblib"
BACKGROUND_PATH = BASE_DIR / "models" / "shap_background_top7.csv"
SHAP_OUTPUT_DIR = BASE_DIR / "static" / "shap_outputs"

FEATURES = ["GenHlth", "Age", "BMI", "HighBP", "HighChol", "Sex", "Income"]

CHOICES = {
    "GenHlth": [
        (1, "Excellent"),
        (2, "Very good"),
        (3, "Good"),
        (4, "Fair"),
        (5, "Poor"),
    ],
    "Age": [
        (1, "18-24"),
        (2, "25-29"),
        (3, "30-34"),
        (4, "35-39"),
        (5, "40-44"),
        (6, "45-49"),
        (7, "50-54"),
        (8, "55-59"),
        (9, "60-64"),
        (10, "65-69"),
        (11, "70-74"),
        (12, "75-79"),
        (13, "80+"),
    ],
    "HighBP": [(0, "No"), (1, "Yes")],
    "HighChol": [(0, "No"), (1, "Yes")],
    "Sex": [(0, "Female"), (1, "Male")],
    "Income": [
        (1, "Less than $10,000"),
        (2, "$10,000-$14,999"),
        (3, "$15,000-$19,999"),
        (4, "$20,000-$24,999"),
        (5, "$25,000-$34,999"),
        (6, "$35,000-$49,999"),
        (7, "$50,000-$74,999"),
        (8, "$75,000+"),
    ],
}

FEATURE_LABELS = {
    "GenHlth": "General health",
    "Age": "Age group",
    "BMI": "BMI",
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "Sex": "Sex",
    "Income": "Income group",
}

DEFAULT_VALUES = {
    "GenHlth": 3,
    "Age": 9,
    "BMI": 28,
    "HighBP": 0,
    "HighChol": 0,
    "Sex": 0,
    "Income": 5,
}

app = Flask(__name__)


@lru_cache(maxsize=1)
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError(f"SHAP background file not found: {BACKGROUND_PATH}")

    pipeline = joblib.load(MODEL_PATH)
    background_raw = pd.read_csv(BACKGROUND_PATH)[FEATURES]

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    background_processed = preprocessor.transform(background_raw)

    if not isinstance(background_processed, pd.DataFrame):
        background_processed = pd.DataFrame(
            background_processed,
            columns=preprocessor.get_feature_names_out(),
        )

    explainer = shap.TreeExplainer(
        model,
        data=background_processed,
        model_output="probability",
    )
    return pipeline, preprocessor, explainer


def parse_payload(payload):
    values = {}

    def get_float(name, minimum=None, maximum=None):
        raw_value = payload.get(name, DEFAULT_VALUES[name])
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{FEATURE_LABELS[name]} must be numeric.") from exc

        if minimum is not None and value < minimum:
            raise ValueError(f"{FEATURE_LABELS[name]} must be at least {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"{FEATURE_LABELS[name]} must be no more than {maximum}.")
        return value

    def get_choice(name):
        allowed = {choice_value for choice_value, _ in CHOICES[name]}
        value = int(get_float(name))
        if value not in allowed:
            raise ValueError(f"{FEATURE_LABELS[name]} has an invalid value.")
        return value

    values["GenHlth"] = get_choice("GenHlth")
    values["Age"] = get_choice("Age")
    values["BMI"] = get_float("BMI", minimum=10, maximum=80)
    values["HighBP"] = get_choice("HighBP")
    values["HighChol"] = get_choice("HighChol")
    values["Sex"] = get_choice("Sex")
    values["Income"] = get_choice("Income")

    return values, pd.DataFrame([values], columns=FEATURES)


def format_feature_value(feature, value):
    if feature == "BMI":
        return f"{value:.1f}"

    labels = dict(CHOICES.get(feature, []))
    return labels.get(int(value), str(value))


def clean_old_shap_plots(max_files=40):
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_files = sorted(
        SHAP_OUTPUT_DIR.glob("*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_file in png_files[max_files:]:
        old_file.unlink(missing_ok=True)


def build_shap_explanation(input_df):
    _, preprocessor, explainer = load_resources()
    input_processed = preprocessor.transform(input_df)

    if not isinstance(input_processed, pd.DataFrame):
        input_processed = pd.DataFrame(
            input_processed,
            columns=preprocessor.get_feature_names_out(),
        )

    shap_values = explainer(input_processed)
    shap_values_array = shap_values.values
    shap_base_values = shap_values.base_values

    if shap_values_array.ndim == 3:
        shap_values_array = shap_values_array[:, :, 1]
        if np.ndim(shap_base_values) == 2:
            shap_base_values = shap_base_values[:, 1]

    raw_values_for_plot = input_df[list(input_processed.columns)].to_numpy()
    explanation = shap.Explanation(
        values=shap_values_array,
        base_values=shap_base_values,
        data=raw_values_for_plot,
        feature_names=list(input_processed.columns),
    )

    clean_old_shap_plots()
    filename = f"shap_{uuid4().hex}.png"
    output_path = SHAP_OUTPUT_DIR / filename

    shap.plots.waterfall(explanation[0], max_display=len(FEATURES), show=False)
    plt.title("SHAP explanation for this prediction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()

    contributions = []
    raw_values = input_df.iloc[0].to_dict()
    for feature, shap_value in zip(explanation.feature_names, explanation.values[0]):
        contributions.append(
            {
                "feature": FEATURE_LABELS[feature],
                "value": format_feature_value(feature, raw_values[feature]),
                "impact": float(shap_value),
                "direction": "Increases risk" if shap_value >= 0 else "Lowers risk",
            }
        )

    contributions.sort(key=lambda row: abs(row["impact"]), reverse=True)
    shap_url = url_for("static", filename=f"shap_outputs/{filename}")
    return shap_url, contributions


def predict(input_df):
    pipeline, _, _ = load_resources()
    probability = float(pipeline.predict_proba(input_df)[0, 1])
    prediction = int(pipeline.predict(input_df)[0])
    shap_url, contributions = build_shap_explanation(input_df)

    return {
        "risk_label": "Higher risk" if prediction == 1 else "Lower risk",
        "risk_class": "high" if prediction == 1 else "low",
        "probability": probability,
        "probability_percent": probability * 100,
        "shap_url": shap_url,
        "contributions": contributions,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    form_values = dict(DEFAULT_VALUES)
    result = None
    error = None

    if request.method == "POST":
        try:
            form_values, input_df = parse_payload(request.form)
            result = predict(input_df)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        choices=CHOICES,
        defaults=DEFAULT_VALUES,
        error=error,
        feature_labels=FEATURE_LABELS,
        form_values=form_values,
        result=result,
    )


@app.post("/api/predict")
def api_predict():
    try:
        payload = request.get_json(silent=True) or {}
        values, input_df = parse_payload(payload)
        result = predict(input_df)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "inputs": values,
            "risk_label": result["risk_label"],
            "probability": result["probability"],
            "shap_image_url": url_for(
                "static",
                filename=result["shap_url"].replace("/static/", ""),
                _external=True,
            ),
            "contributions": result["contributions"],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port)
