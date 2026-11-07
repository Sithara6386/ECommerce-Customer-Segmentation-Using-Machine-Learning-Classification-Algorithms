import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Customer Segmentation")
st.markdown(
    "### Predict customer segments using Machine Learning"
)

st.write(
    "Upload a customer-level CSV file to predict whether "
    "customers belong to the Consumer, Premium, VIP, or Business segment."
)


# ============================================================
# 2. FIND MODEL FILES AUTOMATICALLY
# ============================================================

APP_DIR = Path(__file__).resolve().parent

# Possible model folder locations
possible_model_dirs = [
    APP_DIR / "models",
    APP_DIR.parent / "models",
    Path.cwd() / "models",
    Path.cwd() / "Ecommerce" / "models"
]

MODEL_PATH = None
FEATURES_PATH = None

for folder in possible_model_dirs:

    model_file = folder / "best_model.joblib"
    feature_file = folder / "feature_columns.joblib"

    if model_file.is_file() and feature_file.is_file():

        MODEL_PATH = model_file
        FEATURES_PATH = feature_file

        break


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model(model_path, features_path):

    model = joblib.load(model_path)

    feature_columns = joblib.load(features_path)

    return model, list(feature_columns)


if MODEL_PATH is None or FEATURES_PATH is None:

    st.error("Trained model files could not be located.")

    st.write("Application directory:", str(APP_DIR))

    st.write("Folders searched:")

    for folder in possible_model_dirs:
        st.write(str(folder))

    st.info(
        "Make sure best_model.joblib and "
        "feature_columns.joblib are committed to GitHub "
        "inside the same models folder."
    )

    st.stop()


try:

    model, feature_columns = load_model(
        str(MODEL_PATH),
        str(FEATURES_PATH)
    )

    st.success("Trained model loaded successfully!")

except Exception as e:

    st.error("Unable to load the trained model.")

    st.exception(e)

    st.stop()


# ============================================================
# 4. DISPLAY MODEL INFORMATION
# ============================================================

with st.expander("View Model Information"):

    st.write("Model:", type(model).__name__)

    st.write("Number of required features:", len(feature_columns))

    st.write("Model file:", str(MODEL_PATH))

    st.write("Feature file:", str(FEATURES_PATH))


# ============================================================
# 5. UPLOAD CUSTOMER DATA
# ============================================================

st.divider()

st.subheader("Upload Customer Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully!")

        st.subheader("Uploaded Data Preview")

        st.dataframe(data.head(10), use_container_width=True)

        st.write("Number of records:", len(data))

        st.write("Number of columns:", len(data.columns))


        # ====================================================
        # 6. VALIDATE INPUT FEATURES
        # ====================================================

        missing_features = [
            col for col in feature_columns
            if col not in data.columns
        ]

        if missing_features:

            st.error(
                "The uploaded dataset is missing required model features."
            )

            st.write("Missing features:")

            st.write(missing_features)

            st.warning(
                "Upload a customer-level dataset containing the "
                "same engineered and encoded features used during "
                "model training."
            )

            st.stop()


        # Select features in exactly the training order
        X = data[feature_columns].copy()


        # ====================================================
        # 7. DATA TYPE AND MISSING VALUE VALIDATION
        # ====================================================

        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isnull().any().any():

            st.error(
                "Missing or non-numeric values were found "
                "in the required model features."
            )

            missing_counts = X.isnull().sum()

            st.write(
                missing_counts[missing_counts > 0]
            )

            st.stop()


        if not np.isfinite(X.to_numpy(dtype=float)).all():

            st.error(
                "Infinite values were found in the input data."
            )

            st.stop()


        # ====================================================
        # 8. PREDICTION
        # ====================================================

        st.divider()

        st.subheader("Customer Segment Prediction")

        if st.button(
            "Predict Customer Segments",
            type="primary",
            use_container_width=True
        ):

            predictions = model.predict(X)

            # Append predictions to original dataset
            results = data.copy()

            results["Predicted_Segment"] = predictions

            st.success("Customer segmentation completed!")

            # ================================================
            # 9. DISPLAY PREDICTION RESULTS
            # ================================================

            st.subheader("Prediction Results")

            st.dataframe(
                results,
                use_container_width=True
            )


            # ================================================
            # 10. SEGMENT DISTRIBUTION
            # ================================================

            st.subheader("Customer Segment Distribution")

            segment_counts = (
                pd.Series(predictions)
                .value_counts()
                .sort_values(ascending=False)
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Total Customers",
                    len(predictions)
                )

            with col2:

                st.metric(
                    "Number of Segments Predicted",
                    len(segment_counts)
                )

            st.bar_chart(segment_counts)


            # ================================================
            # 11. DOWNLOAD RESULTS
            # ================================================

            st.subheader("Download Predictions")

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Customer Predictions CSV",
                data=csv_data,
                file_name="customer_segmentation_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:

        st.error("An error occurred while processing the uploaded file.")

        st.exception(e)


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.caption(
    "E-Commerce Customer Segmentation | "
    "Machine Learning Classification Project"
)
