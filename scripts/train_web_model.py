from pathlib import Path
import json

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TOP_FEATURES = ["GenHlth", "Age", "BMI", "HighBP", "HighChol", "Sex", "Income"]
SCALED_FEATURES = ["BMI", "GenHlth", "Age", "Income"]
BINARY_FEATURES = ["HighBP", "HighChol", "Sex"]


def evaluate_predictions(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_score),
        "Average Precision": average_precision_score(y_true, y_score),
    }


def main():
    cdc_diabetes = fetch_ucirepo(id=891)
    x_raw = cdc_diabetes.data.features.copy()
    y_raw = cdc_diabetes.data.targets.copy()
    y = y_raw.squeeze()

    x = x_raw[TOP_FEATURES].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaled", StandardScaler(), SCALED_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_score = pipeline.predict_proba(x_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_pred, y_score)

    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_model = pipeline.named_steps["model"]
    scaler = fitted_preprocessor.named_transformers_["scaled"]
    model_features = list(fitted_preprocessor.get_feature_names_out())

    coefficients = {
        feature: float(coef)
        for feature, coef in zip(model_features, fitted_model.coef_[0])
    }
    scaler_stats = {
        feature: {
            "mean": float(mean),
            "scale": float(scale),
        }
        for feature, mean, scale in zip(
            SCALED_FEATURES,
            scaler.mean_,
            scaler.scale_,
        )
    }

    export = {
        "selected_model": "Logistic Regression",
        "model_purpose": "Lightweight Vercel web-app model",
        "feature_set": TOP_FEATURES,
        "scaled_features": SCALED_FEATURES,
        "binary_features": BINARY_FEATURES,
        "model_features": model_features,
        "coefficients": coefficients,
        "intercept": float(fitted_model.intercept_[0]),
        "threshold": 0.5,
        "scaler": scaler_stats,
        "metrics": metrics,
        "positive_class_rate_train": float(y_train.mean()),
        "positive_class_rate_test": float(y_test.mean()),
        "random_state": RANDOM_STATE,
    }

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "cdc_diabetes_top7_web_model.json"
    metrics_path = models_dir / "cdc_diabetes_top7_web_model_metrics.json"
    model_path.write_text(json.dumps(export, indent=2))
    metrics_path.write_text(json.dumps(export["metrics"], indent=2))

    print(f"Saved web model to: {model_path}")
    print(f"Saved web model metrics to: {metrics_path}")
    print(json.dumps(export["metrics"], indent=2))


if __name__ == "__main__":
    main()
