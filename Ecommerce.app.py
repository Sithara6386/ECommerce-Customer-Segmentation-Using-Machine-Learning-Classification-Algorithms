import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# --------------------------------------------------
# APPLICATION TITLE
# --------------------------------------------------

st.title("🛍️ E-Commerce Customer Segmentation")
st.subheader("Customer Segment Prediction Using Machine Learning")

st.write(
    """
    This application uses a trained Linear Support Vector Machine
    (Linear SVM) model to predict customer segments based on
    purchasing behaviour and customer analytics.
    """
)

st.divider()

# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = "models/best_model.joblib"
    columns_path = "models/feature_columns.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found.")

    if not os.path.exists(columns_path):
        raise FileNotFoundError("Feature columns file not found.")

    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)

    return model, feature_columns


try:
    model, feature_columns = load_model()
    st.success("Trained Linear SVM model loaded successfully!")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --------------------------------------------------
# CSV FILE UPLOAD
# --------------------------------------------------

st.header("Upload Customer Data")

uploaded_file = st.file_uploader(
    "Upload a customer-level CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data")
    st.dataframe(data.head())

    # --------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------

    missing_features = [
        col for col in feature_columns
        if col not in data.columns
    ]

    if missing_features:
        st.error(
            "The uploaded file is missing required model features. "
            "Please upload data with the same feature engineering "
            "and preprocessing used during model training."
        )

        st.write("Missing features:", missing_features)

    else:

        X = data[feature_columns].copy()

        # Check numeric values
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isnull().any().any():
            st.error(
                "The uploaded data contains missing or non-numeric "
                "values in the model features. Please clean the data."
            )

        elif not np.isfinite(X.to_numpy()).all():
            st.error("The uploaded data contains infinite values.")

        else:

            # --------------------------------------------------
            # PREDICTION
            # --------------------------------------------------

            if st.button("Predict Customer Segments"):

                predictions = model.predict(X)

                results = data.copy()
                results["Predicted_Segment"] = predictions

                st.success("Customer segment predictions completed!")

                # --------------------------------------------------
                # DISPLAY RESULTS
                # --------------------------------------------------

                st.header("Prediction Results")

                st.dataframe(results)

                # --------------------------------------------------
                # SEGMENT DISTRIBUTION
                # --------------------------------------------------

                st.header("Customer Segment Distribution")

                segment_counts = (
                    pd.Series(predictions)
                    .value_counts()
                    .rename_axis("Customer Segment")
                    .reset_index(name="Number of Customers")
                )

                st.bar_chart(
                    segment_counts.set_index("Customer Segment")
                )

                st.write("### Segment Summary")
                st.dataframe(segment_counts)

                # --------------------------------------------------
                # DOWNLOAD RESULTS
                # --------------------------------------------------

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="customer_segment_predictions.csv",
                    mime="text/csv"
                )

else:

    st.info("Please upload a customer-level CSV file to begin.")