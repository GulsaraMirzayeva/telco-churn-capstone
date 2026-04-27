import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "conservative_logistic_regression_pipeline.joblib"


# -----------------------------
# Expected model features
# -----------------------------
NUMERIC_FEATURES = [
    "Age",
    "Number of Dependents",
    "Population",
    "Number of Referrals",
    "Tenure in Months",
    "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Offer",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Internet Type",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

EXPECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    object_cols = df.select_dtypes(include="object").columns.tolist()
    df[object_cols] = df[object_cols].replace("None", np.nan)
    df[object_cols] = df[object_cols].replace(r"^\s*$", np.nan, regex=True)

    if {"Internet Service", "Internet Type"}.issubset(df.columns):
        no_internet_mask = df["Internet Service"].eq("No") & df["Internet Type"].isna()
        df.loc[no_internet_mask, "Internet Type"] = "No Internet"

    if "Internet Type" in df.columns:
        df["Internet Type"] = df["Internet Type"].fillna("Missing")

    if "Offer" in df.columns:
        df["Offer"] = df["Offer"].fillna("No Offer")

    return df


def classify_risk(probability: float) -> str:
    if probability >= 0.65:
        return "High Risk"
    elif probability >= 0.35:
        return "Medium Risk"
    else:
        return "Low Risk"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [col for col in EXPECTED_FEATURES if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df[EXPECTED_FEATURES].copy()


# -----------------------------
# Load model
# -----------------------------
model = load_model()


# -----------------------------
# App title
# -----------------------------
st.title("📊 Telco Customer Churn Prediction")

st.write(
    """
    This app predicts the probability that a telecom customer may churn.
    The prediction is based on the conservative Logistic Regression model
    trained without `Satisfaction Score`.
    """
)

st.info(
    "This version is designed for cautious business-facing use. "
    "It avoids `Satisfaction Score` because that feature may be too close to the churn decision."
)


# -----------------------------
# Tabs
# -----------------------------
tab_single, tab_bulk, tab_about = st.tabs(
    ["Single Prediction", "Bulk CSV Prediction", "About Model"]
)


# -----------------------------
# Single prediction tab
# -----------------------------
with tab_single:
    st.subheader("Single Customer Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
        population = st.number_input("Population", min_value=0, value=10000)
        number_of_referrals = st.number_input("Number of Referrals", min_value=0, max_value=20, value=0)

    with col2:
        tenure = st.number_input("Tenure in Months", min_value=0, max_value=100, value=12)
        avg_long_distance = st.number_input("Avg Monthly Long Distance Charges", min_value=0.0, value=10.0)
        avg_gb_download = st.number_input("Avg Monthly GB Download", min_value=0.0, value=20.0)
        monthly_charge = st.number_input("Monthly Charge", min_value=0.0, value=70.0)

    with col3:
        total_charges = st.number_input("Total Charges", min_value=0.0, value=800.0)
        total_refunds = st.number_input("Total Refunds", min_value=0.0, value=0.0)
        total_extra_data = st.number_input("Total Extra Data Charges", min_value=0.0, value=0.0)
        total_long_distance = st.number_input("Total Long Distance Charges", min_value=0.0, value=100.0)

    st.markdown("---")

    col4, col5, col6 = st.columns(3)

    with col4:
        gender = st.selectbox("Gender", ["Female", "Male"])
        married = st.selectbox("Married", ["No", "Yes"])
        offer = st.selectbox("Offer", ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

    with col5:
        internet_service = st.selectbox("Internet Service", ["Yes", "No"])
        internet_type = st.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "No Internet", "Missing"])
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection Plan", ["No", "Yes"])
        tech_support = st.selectbox("Premium Tech Support", ["No", "Yes"])

    with col6:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        streaming_music = st.selectbox("Streaming Music", ["No", "Yes"])
        unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
        contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])

    input_data = pd.DataFrame([{
        "Age": age,
        "Number of Dependents": number_of_dependents,
        "Population": population,
        "Number of Referrals": number_of_referrals,
        "Tenure in Months": tenure,
        "Avg Monthly Long Distance Charges": avg_long_distance,
        "Avg Monthly GB Download": avg_gb_download,
        "Monthly Charge": monthly_charge,
        "Total Charges": total_charges,
        "Total Refunds": total_refunds,
        "Total Extra Data Charges": total_extra_data,
        "Total Long Distance Charges": total_long_distance,
        "Gender": gender,
        "Married": married,
        "Offer": offer,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Internet Type": internet_type,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection Plan": device_protection,
        "Premium Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Streaming Music": streaming_music,
        "Unlimited Data": unlimited_data,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
    }])

    if st.button("Predict Churn Probability"):
        prediction_probability = model.predict_proba(input_data)[0][1]
        prediction_class = model.predict(input_data)[0]
        risk_label = classify_risk(prediction_probability)

        st.markdown("### Prediction Result")

        col_result1, col_result2, col_result3 = st.columns(3)

        col_result1.metric("Churn Probability", f"{prediction_probability:.1%}")
        col_result2.metric("Predicted Churn", "Yes" if prediction_class == 1 else "No")
        col_result3.metric("Risk Level", risk_label)

        if risk_label == "High Risk":
            st.error("This customer is in a high-risk churn segment.")
        elif risk_label == "Medium Risk":
            st.warning("This customer has a moderate churn risk.")
        else:
            st.success("This customer has a relatively low churn risk.")


# -----------------------------
# Bulk prediction tab
# -----------------------------
with tab_bulk:
    st.subheader("Bulk CSV Prediction")

    st.write(
        """
        Upload a CSV file with the required customer feature columns.
        The app will return churn probability and risk level for each customer.
        """
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(raw_df.head())

        try:
            cleaned_df = normalize_missing_values(raw_df)
            feature_df = prepare_features(cleaned_df)

            probabilities = model.predict_proba(feature_df)[:, 1]
            predictions = model.predict(feature_df)

            result_df = raw_df.copy()
            result_df["churn_probability"] = probabilities
            result_df["predicted_churn"] = np.where(predictions == 1, "Yes", "No")
            result_df["risk_level"] = [classify_risk(prob) for prob in probabilities]

            st.success("Predictions generated successfully.")
            st.dataframe(result_df.head(20))

            csv_output = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="telco_churn_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.write(e)


# -----------------------------
# About model tab
# -----------------------------
with tab_about:
    st.subheader("About the Final Model")

    st.write(
        """
        The main model used in this app is the conservative Logistic Regression model
        trained without `Satisfaction Score`.

        This model was selected because it gives a better balance between:

        - predictive performance
        - interpretability
        - business explainability
        - leakage sensitivity
        - deployment defensibility
        """
    )

    st.write(
        """
        The expanded XGBoost model achieved stronger raw performance, but it depended heavily
        on `Satisfaction Score`. Because that feature may reflect customer experience very close
        to churn behavior, the conservative model is used as the primary app model.
        """
    )