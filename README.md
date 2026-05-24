# CDC Diabetes Risk Prediction

This project builds a machine learning model to predict whether a survey
respondent is in the positive `Diabetes_binary` class, meaning they report
prediabetes or diabetes. The dataset is the CDC Diabetes Health Indicators
dataset from the UCI Machine Learning Repository.

The current notebook focuses on a simplified 7-feature model designed to be
easier to explain and easier to discuss with non-technical stakeholders.

## Project Goal

The goal is not to diagnose diabetes. Instead, the model explores whether a
small group of survey-based health indicators can help identify respondents who
may be at higher risk and therefore worth prioritizing for outreach,
screening, or follow-up.

The model uses these 7 features:

- `GenHlth`: self-reported general health
- `Age`: age category
- `BMI`: body mass index
- `HighBP`: high blood pressure indicator
- `HighChol`: high cholesterol indicator
- `Sex`: binary sex indicator in the dataset
- `Income`: income category

## Main Notebook

- [Predicting_CDC_Diabetes_7_Features.ipynb](Predicting_CDC_Diabetes_7_Features.ipynb)

The notebook walks through:

- Data loading from UCI with `ucimlrepo`
- Exploratory data analysis
- Train/test split
- Preprocessing
- Model comparison
- Evaluation with accuracy, precision, recall, F1, ROC AUC, and average precision
- Model interpretation with permutation importance and SHAP
- Saving the trained model and metrics

## Current Results

The selected model is `LightGBM`.

| Metric | Value |
|:--|--:|
| Accuracy | 0.712 |
| Precision | 0.299 |
| Recall | 0.795 |
| F1 | 0.435 |
| ROC AUC | 0.822 |
| Average Precision | 0.414 |

The positive class is imbalanced, so accuracy is not the most important metric.
The model has high recall, meaning it catches many respondents in the positive
class, but precision is around 30%, meaning many flagged respondents are false
positives. That tradeoff may be acceptable for low-cost outreach or screening,
but it would not be appropriate for diagnosis or high-stakes decisions by
itself.

## Stakeholder Insights

- The strongest signals are general health, BMI, age, high blood pressure, and
  high cholesterol.
- The model is better suited for risk ranking than yes/no diagnosis.
- A high-recall model can be useful when the cost of missing a likely diabetes
  case is higher than the cost of following up with someone who is not positive.
- Precision around 30% means the model should be paired with human review,
  additional screening, or clinical context.
- `Sex` and `Income` should be interpreted carefully because they may reflect
  broader social, access-to-care, and diagnosis patterns rather than direct
  biological effects.

## Technical Insights

- The target is imbalanced, with the majority class being no diabetes.
- Average precision is emphasized because it is more informative than accuracy
  for imbalanced classification.
- Class weighting is used to help models pay more attention to the minority
  class.
- The 7-feature model is intentionally simpler than the full-feature workflow.
  It compares Logistic Regression, Random Forest, and LightGBM.
- SHAP and permutation importance agree that the model relies most heavily on
  health-status and risk-factor variables.

Current saved outputs:

- `models/cdc_diabetes_top7_best_model.joblib`
- `models/cdc_diabetes_top7_model_metrics.json`

## How To Run

This project uses a conda environment defined in `environment.yml`.

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate ml_example_project
```

If the environment already exists, update it instead:

```bash
conda env update -f environment.yml --prune
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Predicting_CDC_Diabetes_7_Features.ipynb
```

Run the notebook from top to bottom. The notebook fetches the dataset directly
from UCI, so no local CSV is required.

Optional: register the conda environment as a named Jupyter kernel:

```bash
python -m ipykernel install --user --name ml_example_project --display-name "Python (ml_example_project)"
```

## Environment

Key packages include:

- Jupyter Notebook and IPython kernel support
- NumPy, pandas, and SciPy
- scikit-learn
- LightGBM, XGBoost, and CatBoost
- imbalanced-learn
- SHAP
- Matplotlib, seaborn, and Plotly
- ucimlrepo

## Important Caveat

This is an educational machine learning project using survey data. The model
should not be treated as a medical device, diagnostic tool, or replacement for
clinical judgment.
