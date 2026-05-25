from functools import lru_cache
import base64
from html import escape
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

MODEL_PATH = BASE_DIR / "models" / "cdc_diabetes_top7_best_model.joblib"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

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

AI_EXPLANATION_INSTRUCTIONS = """
You explain an educational diabetes risk prediction to the person who just used
the web app.

Write like a friendly person explaining the result, not like a report.
Keep it short, direct, and easy to skim.

Output exactly 3 short paragraphs:
1. What the result means in one or two plain sentences.
2. The main reasons for the result, mentioning only the top 2 factors from the
   feature contributions.
3. A gentle next step the person can take.

Style:
- Use "you."
- Avoid technical words such as SHAP, class, model output, contribution, false
  positive, or feature importance.
- Avoid long lists.
- Avoid saying "CDC survey dataset" more than once.
- Keep the full response under 95 words.

Safety rules:
- Do not diagnose diabetes or prediabetes.
- Do not claim the model is clinically validated.
- Do not say the person definitely has or does not have diabetes.
- Do not recommend medication, supplements, or treatment changes.
- Appropriate next steps include asking about A1C or fasting glucose screening,
  checking blood pressure and cholesterol, and reviewing activity, nutrition,
  and weight-related goals with a professional.
- Keep the response under 180 words.
"""

app = Flask(__name__, static_folder="public", static_url_path="")


@lru_cache(maxsize=1)
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    return pipeline, preprocessor, model


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


def readable_inputs(values):
    return {
        FEATURE_LABELS[feature]: format_feature_value(feature, values[feature])
        for feature in FEATURES
    }


def generate_ai_explanation(input_values, result):
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "OpenAI explanation unavailable: no API key was found in `.env`.", False

    prompt_payload = {
        "inputs": readable_inputs(input_values),
        "model_result": {
            "risk_label": result["risk_label"],
            "probability_percent": round(result["probability_percent"], 1),
        },
        "top_feature_contributions": result["contributions"],
        "educational_disclaimer": (
            "This is a machine learning demo and should not be used for actual "
            "diagnosis or treatment decisions."
        ),
    }

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=AI_EXPLANATION_INSTRUCTIONS,
            input=json.dumps(prompt_payload, indent=2),
            max_output_tokens=1200,
            reasoning={"effort": "minimal"},
        )
        explanation = response.output_text.strip()
        if not explanation:
            incomplete_details = getattr(response, "incomplete_details", None)
            if incomplete_details:
                raise RuntimeError(f"OpenAI response was incomplete: {incomplete_details}")
            raise RuntimeError("OpenAI returned an empty explanation.")
        return explanation, None, True
    except Exception:
        return (
            None,
            "OpenAI explanation failed. Check the API key, quota, billing, model name, or network connection.",
            False,
        )


def build_shap_svg(contributions):
    width = 860
    row_height = 54
    margin_left = 230
    margin_right = 90
    top = 58
    height = top + (len(contributions) * row_height) + 28
    max_abs = max(abs(row["impact"]) for row in contributions) or 1
    axis_x = margin_left + ((width - margin_left - margin_right) / 2)
    max_bar = (width - margin_left - margin_right) / 2

    rows = []
    for index, row in enumerate(contributions):
        y = top + index * row_height
        impact = row["impact"]
        bar_width = max(4, abs(impact) / max_abs * max_bar)
        if impact >= 0:
            x = axis_x
            color = "#ea7a1f"
            direction_label = "Increases risk"
        else:
            x = axis_x - bar_width
            color = "#2563eb"
            direction_label = "Lowers risk"

        rows.append(
            f"""
            <text x="24" y="{y + 18}" class="feature">{escape(row["feature"])}</text>
            <text x="24" y="{y + 38}" class="value">{escape(row["value"])}</text>
            <rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="24" rx="5" fill="{color}" />
            <text x="{axis_x + max_bar + 18}" y="{y + 18}" class="impact">{direction_label}</text>
            <text x="{axis_x + max_bar + 18}" y="{y + 38}" class="value">{impact:+.4f}</text>
            """
        )

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <style>
        .title {{ font: 700 24px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #182230; }}
        .subtitle {{ font: 500 13px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #667085; }}
        .feature {{ font: 700 15px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #182230; }}
        .value {{ font: 500 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #667085; }}
        .impact {{ font: 700 13px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #344054; }}
        .axis {{ stroke: #98a2b3; stroke-width: 2; }}
        .grid {{ stroke: #e4e7ec; stroke-width: 1; }}
      </style>
      <rect width="100%" height="100%" fill="#ffffff" />
      <text x="24" y="30" class="title">Feature impact for this prediction</text>
      <text x="24" y="50" class="subtitle">Orange pushes the score higher. Blue pushes it lower.</text>
      <line x1="{axis_x:.2f}" y1="{top - 8}" x2="{axis_x:.2f}" y2="{height - 20}" class="axis" />
      <line x1="{margin_left}" y1="{top - 8}" x2="{width - margin_right}" y2="{top - 8}" class="grid" />
      {''.join(rows)}
    </svg>
    """
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def build_shap_explanation(input_df):
    _, preprocessor, model = load_resources()
    input_processed = preprocessor.transform(input_df)

    if not isinstance(input_processed, pd.DataFrame):
        input_processed = pd.DataFrame(
            input_processed,
            columns=preprocessor.get_feature_names_out(),
        )

    shap_contributions = model.booster_.predict(input_processed, pred_contrib=True)
    shap_values = np.asarray(shap_contributions)[0][:-1]

    contributions = []
    raw_values = input_df.iloc[0].to_dict()
    for feature, shap_value in zip(input_processed.columns, shap_values):
        contributions.append(
            {
                "feature": FEATURE_LABELS[feature],
                "value": format_feature_value(feature, raw_values[feature]),
                "impact": float(shap_value),
                "direction": "Increases risk" if shap_value >= 0 else "Lowers risk",
            }
        )

    contributions.sort(key=lambda row: abs(row["impact"]), reverse=True)
    shap_image = build_shap_svg(contributions)
    return shap_image, contributions


def predict(input_df):
    pipeline, _, _ = load_resources()
    probability = float(pipeline.predict_proba(input_df)[0, 1])
    prediction = int(pipeline.predict(input_df)[0])
    shap_image, contributions = build_shap_explanation(input_df)

    result = {
        "risk_label": "Higher risk" if prediction == 1 else "Lower risk",
        "risk_class": "high" if prediction == 1 else "low",
        "probability": probability,
        "probability_percent": probability * 100,
        "shap_image": shap_image,
        "contributions": contributions,
    }
    ai_explanation, ai_explanation_error, ai_explanation_from_openai = generate_ai_explanation(
        input_df.iloc[0].to_dict(),
        result,
    )
    result["ai_explanation"] = ai_explanation
    result["ai_explanation_error"] = ai_explanation_error
    result["ai_explanation_enabled"] = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    result["ai_explanation_from_openai"] = ai_explanation_from_openai
    return result


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
            "ai_explanation": result["ai_explanation"],
            "ai_explanation_error": result["ai_explanation_error"],
            "ai_explanation_from_openai": result["ai_explanation_from_openai"],
            "shap_image": result["shap_image"],
            "contributions": result["contributions"],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port)
