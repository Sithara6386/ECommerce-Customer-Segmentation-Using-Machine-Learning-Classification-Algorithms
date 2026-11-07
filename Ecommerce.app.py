import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

st.title("E-Commerce Customer Segmentation")
st.subheader("Machine Learning Model Evaluation Dashboard")

st.markdown("""
Upload your **test dataset** containing the actual customer segment
and the required model input features.

The application evaluates previously trained models without
retraining them.
""")


# ============================================================
# 2. LOCATE MODEL FILES
# ============================================================

APP_DIR = Path(__file__).resolve().parent

FEATURES_PATH = APP_DIR / "feature_columns.joblib"

# Find saved model files in the repository root
MODEL_FILES = {}

# Include best_model.joblib
best_model_path = APP_DIR / "best_model.joblib"

if best_model_path.is_file():
    MODEL_FILES["Best Model"] = best_model_path

# Include any additional saved models
for path in APP_DIR.glob("*model*.joblib"):

    if path.name == "best_model.joblib":
        continue

    model_name = path.stem.replace("_model", "").replace("_", " ").title()

    MODEL_FILES[model_name] = path


if not FEATURES_PATH.is_file():

    st.error(
        f"Feature columns file not found: {FEATURES_PATH}"
    )

    st.stop()


if not MODEL_FILES:

    st.error("No trained model files were found in the repository root.")

    st.stop()


# ============================================================
# 3. LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_artifacts(model_files, features_path):

    models = {}

    for model_name, model_path in model_files:

        models[model_name] = joblib.load(model_path)

    feature_columns = joblib.load(features_path)

    return models, list(feature_columns)


try:

    model_files_tuple = tuple(
        (name, str(path))
        for name, path in MODEL_FILES.items()
    )

    models, feature_columns = load_artifacts(
        model_files_tuple,
        str(FEATURES_PATH)
    )

    st.success(
        f"{len(models)} trained model(s) loaded successfully."
    )

except Exception as e:

    st.error(f"Error loading model: {e}")

    st.stop()


# ============================================================
# 4. TEST DATASET UPLOAD
# ============================================================

st.divider()

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Please upload your test CSV file to begin evaluation.")

    st.stop()


try:

    test_data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read uploaded CSV: {e}")

    st.stop()


st.success("Test dataset uploaded successfully!")

st.write("Dataset dimensions:", test_data.shape)

with st.expander("Preview Test Dataset"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )



# ============================================================
# 5. VALIDATE TEST DATA
# ============================================================

st.divider()

st.header("2. Validate Test Dataset")

TARGET = "customer_segment"

# Check target column
if TARGET not in test_data.columns:
    st.error(
        f"Target column '{TARGET}' is missing from the uploaded CSV."
    )
    st.stop()

# Check required input features
missing_features = [
    col for col in feature_columns
    if col not in test_data.columns
]

if missing_features:
    st.error("Required model features are missing.")

    st.write(missing_features)
    st.stop()

# Extract actual target values
y_test = test_data[TARGET].copy()

# Extract features in the exact training order
# IMPORTANT: Preserve categorical and numerical data types.
X_test = test_data[feature_columns].copy()

# Check missing values without converting categorical data
missing_counts = X_test.isnull().sum()
total_missing = int(missing_counts.sum())

if total_missing > 0:

    st.warning(
        f"The test dataset contains {total_missing} missing feature values. "
        "The saved model's preprocessing pipeline will attempt to handle them."
    )

    st.write(
        missing_counts[missing_counts > 0]
    )

# Check infinite values in numerical columns only
numeric_data = X_test.select_dtypes(include=np.number)

if not np.isfinite(numeric_data.to_numpy(dtype=float)).all():

    st.error("Infinite values found in numerical input features.")
    st.stop()

# Check target values
if y_test.isnull().any():

    st.error("Missing values found in customer_segment.")
    st.stop()

if len(test_data) == 0:

    st.error("The uploaded test dataset is empty.")
    st.stop()

st.success("Test dataset validation completed successfully!")

col1, col2, col3 = st.columns(3)

col1.metric("Test Records", len(test_data))
col2.metric("Input Features", len(feature_columns))
col3.metric("Available Models", len(models))

st.write("Input feature data types:")

st.dataframe(
    X_test.dtypes.astype(str).reset_index().rename(
        columns={"index": "Feature", 0: "Data Type"}
    ),
    use_container_width=True
)


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    results = {

        "Accuracy": accuracy_score(y, predictions),

        "Precision": precision_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "F1 Weighted": f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "F1 Macro": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y,
            predictions
        ),

        "MCC": matthews_corrcoef(
            y,
            predictions
        )
    }


    # Multiclass ROC-AUC
    try:

        if hasattr(model, "predict_proba"):

            y_score = model.predict_proba(X)

            results["AUC"] = roc_auc_score(
                y,
                y_score,
                labels=model.classes_,
                multi_class="ovr",
                average="weighted"
            )

        else:

            results["AUC"] = np.nan

    except Exception:

        results["AUC"] = np.nan


    return predictions, results


# ============================================================
# 7. EVALUATE ALL SAVED MODELS
# ============================================================

st.divider()

st.header("3. Model Performance Comparison")

if st.button(
    "Evaluate All Models",
    type="primary",
    use_container_width=True
):

    comparison_results = []
    predictions_by_model = {}
    errors = {}

    for model_name, model in models.items():

        try:

            predictions, metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            predictions_by_model[model_name] = predictions

            comparison_results.append({
                "Model": model_name,
                **metrics
            })

        except Exception as e:

            errors[model_name] = str(e)


    if comparison_results:

        comparison_df = pd.DataFrame(comparison_results)

        st.session_state["comparison_df"] = comparison_df

        st.session_state["predictions_by_model"] = predictions_by_model

        st.session_state["evaluation_errors"] = errors

        st.success("Model evaluation completed!")

    else:

        st.error("None of the saved models could be evaluated.")

        for name, error in errors.items():

            st.write(name, error)


# ============================================================
# 8. DISPLAY MODEL COMPARISON TABLE
# ============================================================

if "comparison_df" in st.session_state:

    comparison_df = st.session_state["comparison_df"]

    st.subheader("Performance of Different Models")

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )


    # Best model based on weighted F1
    best_index = comparison_df["F1 Weighted"].idxmax()

    best_name = comparison_df.loc[
        best_index, "Model"
    ]

    st.success(
        f"Best model on uploaded test data based on Weighted F1: {best_name}"
    )


    if st.session_state["evaluation_errors"]:

        st.warning("Some models could not be evaluated.")

        for name, error in st.session_state["evaluation_errors"].items():

            st.write(f"{name}: {error}")


    # ========================================================
    # 9. MODEL SELECTION DROPDOWN
    # ========================================================

    st.divider()

    st.header("4. Detailed Model Evaluation")

    available_models = list(
        st.session_state["predictions_by_model"].keys()
    )

    selected_model_name = st.selectbox(
        "Select a model to view detailed results:",
        available_models
    )

    selected_model = models[selected_model_name]

    selected_predictions = (
        st.session_state["predictions_by_model"][selected_model_name]
    )


    # ========================================================
    # 10. DISPLAY SELECTED MODEL METRICS
    # ========================================================

    selected_metrics = comparison_df[
        comparison_df["Model"] == selected_model_name
    ].iloc[0]


    st.subheader(f"{selected_model_name} - Evaluation Metrics")

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Accuracy",
        f"{selected_metrics['Accuracy']:.4f}"
    )

    metric_cols[1].metric(
        "Precision (Weighted)",
        f"{selected_metrics['Precision']:.4f}"
    )

    metric_cols[2].metric(
        "Recall (Weighted)",
        f"{selected_metrics['Recall']:.4f}"
    )

    metric_cols[3].metric(
        "F1 Weighted",
        f"{selected_metrics['F1 Weighted']:.4f}"
    )


    metric_cols2 = st.columns(4)

    metric_cols2[0].metric(
        "F1 Macro",
        f"{selected_metrics['F1 Macro']:.4f}"
    )

    metric_cols2[1].metric(
        "Balanced Accuracy",
        f"{selected_metrics['Balanced Accuracy']:.4f}"
    )

    metric_cols2[2].metric(
        "MCC",
        f"{selected_metrics['MCC']:.4f}"
    )

    auc_value = selected_metrics["AUC"]

    metric_cols2[3].metric(
        "ROC-AUC (Weighted OvR)",
        f"{auc_value:.4f}" if pd.notna(auc_value) else "N/A"
    )


    # ========================================================
    # 11. CONFUSION MATRIX
    # ========================================================

    st.divider()

    st.subheader("Confusion Matrix")

    labels = sorted(
        set(y_test.unique()).union(
            set(selected_predictions)
        )
    )

    cm = confusion_matrix(
        y_test,
        selected_predictions,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )


    # ========================================================
    # 12. CLASSIFICATION REPORT
    # ========================================================

    st.divider()

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        selected_predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


    # ========================================================
    # 13. PREDICTION RESULTS
    # ========================================================

    st.divider()

    st.subheader("Test Data Predictions")

    prediction_results = test_data.copy()

    prediction_results["Predicted_Segment"] = selected_predictions

    prediction_results["Prediction_Correct"] = (
        prediction_results[TARGET]
        == prediction_results["Predicted_Segment"]
    )

    st.dataframe(
        prediction_results.head(100),
        use_container_width=True
    )


    # ========================================================
    # 14. DOWNLOAD RESULTS
    # ========================================================

    csv_output = prediction_results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Test Predictions CSV",
        data=csv_output,
        file_name="customer_segment_test_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "E-Commerce Customer Segmentation | "
    "Machine Learning Classification Project"
)
