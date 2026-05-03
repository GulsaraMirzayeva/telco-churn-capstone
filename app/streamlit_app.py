import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from sqlalchemy import create_engine, text


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Paths and model settings
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "conservative_logistic_regression_v2_pipeline.joblib"

FINAL_THRESHOLD = 0.30
LINKEDIN_URL = "https://www.linkedin.com/in/gulsara-mirzayeva/"
DB_TABLE_NAME = "telco_customer_lookup"

# -----------------------------
# Expected model features
# Total Charges is intentionally removed from v2 model
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
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 850;
        margin-bottom: 4px;
        line-height: 1.15;
    }

    .subtitle {
        font-size: 18px;
        color: #A3A3A3;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    .hero-card {
        padding: 28px 30px;
        border-radius: 22px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.92));
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.20);
        margin-bottom: 24px;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.14);
        color: #93C5FD;
        font-size: 13px;
        font-weight: 650;
        margin-bottom: 14px;
        border: 1px solid rgba(147, 197, 253, 0.22);
    }

    .small-note {
        font-size: 13px;
        color: #CBD5E1;
    }

    .soft-card {
        padding: 18px 20px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.20);
        background: rgba(15, 23, 42, 0.45);
        margin-bottom: 16px;
    }

    div.stButton > button:first-child {
        border-radius: 12px;
        padding: 0.65rem 1.15rem;
        font-weight: 750;
        border: 1px solid rgba(239, 68, 68, 0.45);
        background: rgba(239, 68, 68, 0.12);
    }

    div.stButton > button:first-child:hover {
        border: 1px solid rgba(239, 68, 68, 0.85);
        background: rgba(239, 68, 68, 0.20);
    }

    [data-testid="stMetricValue"] {
        font-size: 30px;
        font-weight: 800;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error("Model file was not found. Please check the models folder.")
        st.stop()
    except Exception as error:
        st.error("Model could not be loaded.")
        st.write(error)
        st.stop()


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    object_cols = df.select_dtypes(include="object").columns.tolist()

    if object_cols:
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


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [col for col in EXPECTED_FEATURES if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df[EXPECTED_FEATURES].copy()

DB_TO_MODEL_COLUMN_MAP = {
    "age": "Age",
    "number_of_dependents": "Number of Dependents",
    "population": "Population",
    "number_of_referrals": "Number of Referrals",
    "tenure_in_months": "Tenure in Months",
    "avg_monthly_long_distance_charges": "Avg Monthly Long Distance Charges",
    "avg_monthly_gb_download": "Avg Monthly GB Download",
    "monthly_charge": "Monthly Charge",
    "total_refunds": "Total Refunds",
    "total_extra_data_charges": "Total Extra Data Charges",
    "total_long_distance_charges": "Total Long Distance Charges",
    "gender": "Gender",
    "married": "Married",
    "offer": "Offer",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "internet_type": "Internet Type",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection_plan": "Device Protection Plan",
    "premium_tech_support": "Premium Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "streaming_music": "Streaming Music",
    "unlimited_data": "Unlimited Data",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method",
}


@st.cache_resource
def get_db_engine():
    try:
        return create_engine(st.secrets["DATABASE_URL"], pool_pre_ping=True)
    except KeyError:
        st.error("DATABASE_URL is missing from Streamlit secrets.")
        st.stop()
    except Exception as error:
        st.error("Database connection could not be created.")
        st.write(error)
        st.stop()


@st.cache_data(ttl=600)
def load_customer_options() -> pd.DataFrame:
    engine = get_db_engine()

    query = text(
        f"""
        SELECT customer_id, contract, internet_type, monthly_charge, churn_label
        FROM "{DB_TABLE_NAME}"
        ORDER BY customer_id
        """
    )

    customers = pd.read_sql_query(query, engine)

    customers["customer_label"] = customers.apply(
        lambda row: (
            f"{row['customer_id']} | "
            f"{row['contract']} | "
            f"{row['internet_type']} | "
            f"{row['monthly_charge']} | "
            f"Actual Churn: {row['churn_label']}"
        ),
        axis=1
    )

    return customers


def get_customer_by_id(customer_id: str) -> pd.DataFrame:
    engine = get_db_engine()

    query = text(
        f"""
        SELECT *
        FROM "{DB_TABLE_NAME}"
        WHERE customer_id = :customer_id
        LIMIT 1
        """
    )

    return pd.read_sql_query(
        query,
        engine,
        params={"customer_id": customer_id}
    )


def convert_db_customer_to_model_features(customer_df: pd.DataFrame) -> pd.DataFrame:
    renamed_df = customer_df.rename(columns=DB_TO_MODEL_COLUMN_MAP)
    cleaned_df = normalize_missing_values(renamed_df)
    feature_df = prepare_features(cleaned_df)

    return feature_df


def classify_risk(probability: float) -> str:
    if probability >= 0.65:
        return "High Risk"
    elif probability >= FINAL_THRESHOLD:
        return "Medium Risk"
    else:
        return "Low Risk"


def show_risk_message(risk_label: str):
    if risk_label == "High Risk":
        st.error("This customer is in a high-risk churn segment.")
    elif risk_label == "Medium Risk":
        st.warning("This customer has a moderate churn risk.")
    else:
        st.success("This customer has a relatively low churn risk.")


def validate_customer_input(row: dict) -> list:
    warnings = []

    if row["Internet Service"] == "No" and row["Internet Type"] != "No Internet":
        warnings.append(
            "Internet Service is 'No', but Internet Type is not 'No Internet'. Please review the service information."
        )

    if row["Tenure in Months"] == 0 and row["Monthly Charge"] > 0:
        warnings.append(
            "Tenure is 0 while Monthly Charge is greater than 0. Please check whether this is a new customer."
        )

    return warnings


def clean_feature_name(feature_name: str) -> str:
    cleaned = feature_name

    prefixes = ["numeric__", "categorical__", "num__", "cat__"]

    for prefix in prefixes:
        cleaned = cleaned.replace(prefix, "")

    cleaned = cleaned.replace("_", ": ", 1) if "_" in cleaned else cleaned
    return cleaned


def get_pipeline_parts(model):
    """
    Finds the preprocessing step and logistic regression step inside a sklearn Pipeline.
    This is written defensively because step names may differ across notebooks.
    """
    if not hasattr(model, "named_steps"):
        return None, None

    preprocessor = None
    classifier = None

    for _, step in model.named_steps.items():
        if hasattr(step, "transform") and hasattr(step, "get_feature_names_out"):
            preprocessor = step

        if hasattr(step, "coef_"):
            classifier = step

    return preprocessor, classifier


def get_coefficient_dataframe(model) -> pd.DataFrame:
    preprocessor, classifier = get_pipeline_parts(model)

    if preprocessor is None or classifier is None:
        return pd.DataFrame()

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": [clean_feature_name(name) for name in feature_names],
            "coefficient": coefficients,
        }
    )

    coef_df["direction"] = np.where(
        coef_df["coefficient"] >= 0,
        "Increases churn probability",
        "Reduces churn probability",
    )

    return coef_df


def build_local_explanation(model, input_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    preprocessor, classifier = get_pipeline_parts(model)

    if preprocessor is None or classifier is None:
        return pd.DataFrame()

    transformed_input = preprocessor.transform(input_df)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    values = transformed_input[0]

    contributions = values * coefficients

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": contributions,
        }
    )

    explanation_df["feature"] = explanation_df["feature"].apply(clean_feature_name)
    explanation_df["impact_direction"] = np.where(
        explanation_df["contribution"] >= 0,
        "Increases churn risk",
        "Reduces churn risk",
    )

    explanation_df = explanation_df.reindex(
        explanation_df["contribution"].abs().sort_values(ascending=False).index
    ).head(top_n)

    return explanation_df


def show_explanation_summary(explanation_df: pd.DataFrame):
    if explanation_df.empty:
        st.info("Model explanation is not available for this pipeline structure.")
        return

    increasing = (
        explanation_df[explanation_df["contribution"] > 0]
        .sort_values("contribution", ascending=False)
        .head(3)
    )

    reducing = (
        explanation_df[explanation_df["contribution"] < 0]
        .sort_values("contribution", ascending=True)
        .head(3)
    )

    st.markdown("#### Local explanation summary")
    st.caption(
        "This section explains this specific prediction. Positive factors push the customer closer to churn, "
        "while negative factors reduce the churn probability. These are model contributions for this customer, "
        "not general business rules."
    )

    col_inc, col_red = st.columns(2)

    with col_inc:
        st.markdown("##### 🔴 Factors increasing churn risk")
        if increasing.empty:
            st.write("No strong risk-increasing factor was detected.")
        else:
            for _, row in increasing.iterrows():
                st.markdown(f"- **{row['feature']}** `+{row['contribution']:.3f}`")

    with col_red:
        st.markdown("##### 🟢 Factors reducing churn risk")
        if reducing.empty:
            st.write("No strong risk-reducing factor was detected.")
        else:
            for _, row in reducing.iterrows():
                st.markdown(f"- **{row['feature']}** `{row['contribution']:.3f}`")


def show_explanation_chart(explanation_df: pd.DataFrame):
    if explanation_df.empty:
        st.info("Model explanation is not available for this pipeline structure.")
        return

    plot_df = explanation_df.copy().sort_values("contribution")

    fig = px.bar(
        plot_df,
        x="contribution",
        y="feature",
        orientation="h",
        color="impact_direction",
        title="Feature Contributions for This Customer",
        labels={
            "contribution": "Contribution to churn prediction",
            "feature": "Feature",
            "impact_direction": "Impact direction",
        },
        color_discrete_map={
            "Increases churn risk": "#EF4444",
            "Reduces churn risk": "#22C55E",
        },
        template="plotly_dark",
    )

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")

    fig.update_layout(
        height=480,
        legend_title_text="Impact",
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Interpretation note: these values show how the model used this customer's inputs for this prediction. "
        "They should be read as model contributions, not as direct causal effects."
    )


def show_retention_suggestions(input_row: dict, probability: float):
    st.subheader("💡 Recommended Retention Actions")

    suggestions = []

    if input_row["Contract"] == "Month-to-Month":
        suggestions.append(
            "**Contract:** This customer is on a month-to-month contract. "
            "Offer a small discount or bonus benefit for switching to a one-year or two-year contract."
        )

    if input_row["Monthly Charge"] >= 75:
        suggestions.append(
            "**Monthly Charge:** The monthly charge is relatively high. "
            "Review the customer's usage and suggest a more suitable plan if possible."
        )

    if input_row["Tenure in Months"] <= 6:
        suggestions.append(
            "**Tenure:** This is a relatively new customer. "
            "Use onboarding support, welcome benefits, or early loyalty incentives."
        )

    if input_row["Internet Type"] == "Fiber Optic":
        suggestions.append(
            "**Internet Type:** Fiber optic service should be reviewed carefully because it appeared "
            "as an important churn-related segment during the analysis. "
            "Check service quality, speed complaints, and support history."
        )

    if input_row["Premium Tech Support"] == "No":
        suggestions.append(
            "**Tech Support:** The customer does not have premium tech support. "
            "Consider offering a trial support package if service issues are likely."
        )

    if probability < FINAL_THRESHOLD:
        st.success("This customer is currently low risk. Standard loyalty communication may be enough.")
    elif suggestions:
        for suggestion in suggestions[:3]:
            st.info(suggestion)
    else:
        st.info("Review this customer manually and consider a general retention campaign.")



# -----------------------------
# Load model
# -----------------------------
model = load_model()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Telco Churn")
page = st.sidebar.radio(
    "Navigation",
    [
        "Single Prediction",
        "Existing Customer Lookup",
        "Bulk Prediction",
        "Model Explainability",
        "About Project",
    ]
)

st.sidebar.divider()
st.sidebar.caption("Model version")
st.sidebar.write("Logistic Regression v2.0")
st.sidebar.caption("Without Satisfaction Score and Total Charges")
st.sidebar.caption("Decision threshold")
st.sidebar.write("0.30 optimized for F1")
st.sidebar.divider()
st.sidebar.caption("Author")
st.sidebar.markdown(
    """
    **Gulsare Mirzayeva**  
    Data Science Trainee · Div Academy
    """
)
st.sidebar.markdown(f"[LinkedIn]({LINKEDIN_URL})")


# -----------------------------
# Hero section
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">Business-facing churn prediction tool</div>
        <div class="main-title">Telco Churn Prediction Dashboard</div>
        <div class="subtitle">
            Estimate customer churn risk, understand the model decision, and support retention actions.
        </div>
        <div class="small-note">
            Final model: Logistic Regression v2 · without Satisfaction Score and Total Charges · threshold = 0.30
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Single Prediction Page
# -----------------------------
if page == "Single Prediction":
    st.subheader("Single Customer Prediction")
    st.write("Enter customer information and estimate churn probability.")

    if "single_prediction_result" not in st.session_state:
        st.session_state.single_prediction_result = None

    form_expanded = st.session_state.single_prediction_result is None

    with st.expander("Customer inputs", expanded=form_expanded):
        with st.form("single_prediction_form"):
            st.markdown("### Core Customer Information")
            st.caption(
                "Only the most business-relevant inputs are shown here. "
                "Additional fields are available under Advanced inputs."
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                tenure = st.number_input("Tenure in Months", min_value=0, max_value=100, value=12)
                contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
                monthly_charge = st.number_input("Monthly Charge", min_value=0.0, value=70.0)

            with col2:
                internet_type = st.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "No Internet"])
                payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
                number_of_referrals = st.number_input("Number of Referrals", min_value=0, max_value=20, value=0)

            with col3:
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                tech_support = st.selectbox("Premium Tech Support", ["No", "Yes"])

                estimated_total_charges = tenure * monthly_charge
                st.metric("Estimated Total Charges", f"{estimated_total_charges:,.2f}")
                st.caption("Shown for business context only. This value is not used by the final model.")

            with st.expander("Advanced inputs"):
                adv1, adv2, adv3 = st.columns(3)

                with adv1:
                    age = st.number_input("Age", min_value=18, max_value=100, value=35)
                    gender = st.selectbox("Gender", ["Female", "Male"])
                    married = st.selectbox("Married", ["No", "Yes"])
                    number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
                    population = st.number_input("Population", min_value=0, value=10000)
                    offer = st.selectbox("Offer", ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])

                with adv2:
                    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
                    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                    device_protection = st.selectbox("Device Protection Plan", ["No", "Yes"])
                    unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
                    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

                with adv3:
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
                    streaming_music = st.selectbox("Streaming Music", ["No", "Yes"])
                    avg_long_distance = st.number_input(
                        "Avg Monthly Long Distance Charges",
                        min_value=0.0,
                        value=10.0,
                    )
                    avg_gb_download = st.number_input("Avg Monthly GB Download", min_value=0.0, value=20.0)
                    total_refunds = st.number_input("Total Refunds", min_value=0.0, value=0.0)
                    total_extra_data = st.number_input("Total Extra Data Charges", min_value=0.0, value=0.0)
                    total_long_distance = st.number_input("Total Long Distance Charges", min_value=0.0, value=100.0)

            submitted = st.form_submit_button("Predict Churn Probability")

    internet_service = "No" if internet_type == "No Internet" else "Yes"

    input_row = {
        "Age": age,
        "Number of Dependents": number_of_dependents,
        "Population": population,
        "Number of Referrals": number_of_referrals,
        "Tenure in Months": tenure,
        "Avg Monthly Long Distance Charges": avg_long_distance,
        "Avg Monthly GB Download": avg_gb_download,
        "Monthly Charge": monthly_charge,
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
    }

    input_data = pd.DataFrame([input_row])

    if submitted:
        validation_warnings = validate_customer_input(input_row)

        prediction_probability = model.predict_proba(input_data)[0][1]
        prediction_class = int(prediction_probability >= FINAL_THRESHOLD)
        risk_label = classify_risk(prediction_probability)
        explanation_df = build_local_explanation(model, input_data, top_n=10)

        st.session_state.single_prediction_result = {
            "input_row": input_row,
            "input_data": input_data,
            "validation_warnings": validation_warnings,
            "prediction_probability": prediction_probability,
            "prediction_class": prediction_class,
            "risk_label": risk_label,
            "explanation_df": explanation_df,
        }

        st.rerun()

    if st.session_state.single_prediction_result is not None:
        result = st.session_state.single_prediction_result

        for warning in result["validation_warnings"]:
            st.warning(warning)

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric("Churn Probability", f"{result['prediction_probability']:.1%}")
        result_col2.metric("Predicted Churn", "Yes" if result["prediction_class"] == 1 else "No")
        result_col3.metric("Risk Level", result["risk_label"])

        st.progress(int(result["prediction_probability"] * 100))
        show_risk_message(result["risk_label"])

        with st.expander("Why this prediction?", expanded=True):
            show_explanation_summary(result["explanation_df"])

            st.divider()
            show_explanation_chart(result["explanation_df"])

            if not result["explanation_df"].empty:
                top_increasing = (
                    result["explanation_df"][result["explanation_df"]["contribution"] > 0]
                    .sort_values("contribution", ascending=False)
                    .head(1)
                )

                top_reducing = (
                    result["explanation_df"][result["explanation_df"]["contribution"] < 0]
                    .sort_values("contribution", ascending=True)
                    .head(1)
                )

                st.markdown("#### Plain-language interpretation")

                if not top_increasing.empty:
                    st.write(
                        f"The largest risk-increasing model contribution for this specific customer is "
                        f"**{top_increasing.iloc[0]['feature']}**."
                    )

                if not top_reducing.empty:
                    st.write(
                        f"The largest risk-reducing model contribution for this specific customer is "
                        f"**{top_reducing.iloc[0]['feature']}**."
                    )

                st.caption(
                    "This explanation is local to the selected customer. "
                    "It does not mean that one feature always has the same effect for every customer."
                )

        st.divider()
        show_retention_suggestions(result["input_row"], result["prediction_probability"])

        if st.button("Clear result and enter a new customer"):
            st.session_state.single_prediction_result = None
            st.rerun()

# -----------------------------
# Existing Customer Lookup Page
# -----------------------------
elif page == "Existing Customer Lookup":
    st.subheader("Existing Customer Lookup")
    st.write(
        "Select an existing customer from the cloud PostgreSQL database and generate a churn prediction."
    )

    st.caption(
        "In a real business setting, this Customer ID would usually come from a CRM or customer support screen. "
        "In this capstone demo, the customer profile is retrieved from Neon PostgreSQL."
    )

    try:
        customer_options = load_customer_options()

        selected_label = st.selectbox(
            "Select customer",
            customer_options["customer_label"].tolist()
        )

        selected_customer_id = customer_options.loc[
            customer_options["customer_label"] == selected_label,
            "customer_id"
        ].iloc[0]

        if st.button("Load Customer and Predict"):
            customer_db_df = get_customer_by_id(selected_customer_id)

            if customer_db_df.empty:
                st.error("No customer found with this Customer ID.")
            else:
                st.success(f"Customer found: {selected_customer_id}")

                preview_columns = [
                    "customer_id",
                    "contract",
                    "internet_type",
                    "payment_method",
                    "tenure_in_months",
                    "monthly_charge",
                    "churn_label",
                ]

                available_preview_columns = [
                    col for col in preview_columns if col in customer_db_df.columns
                ]

                st.markdown("#### Customer profile from PostgreSQL")
                st.dataframe(customer_db_df[available_preview_columns])

                feature_df = convert_db_customer_to_model_features(customer_db_df)

                prediction_probability = model.predict_proba(feature_df)[0][1]
                prediction_class = int(prediction_probability >= FINAL_THRESHOLD)
                risk_label = classify_risk(prediction_probability)
                explanation_df = build_local_explanation(model, feature_df, top_n=10)

                st.divider()
                st.subheader("Prediction Result")

                result_col1, result_col2, result_col3 = st.columns(3)

                result_col1.metric("Churn Probability", f"{prediction_probability:.1%}")
                result_col2.metric("Predicted Churn", "Yes" if prediction_class == 1 else "No")
                result_col3.metric("Risk Level", risk_label)

                st.progress(int(prediction_probability * 100))
                show_risk_message(risk_label)

                actual_churn = customer_db_df["churn_label"].iloc[0]

                st.caption(
                    f"Actual churn label in the dataset: **{actual_churn}**. "
                    "This is shown for evaluation context and is not used as a model input."
                )

                with st.expander("Why this prediction?", expanded=True):
                    show_explanation_summary(explanation_df)

                    st.divider()

                    show_explanation_chart(explanation_df)

                    if not explanation_df.empty:
                        top_increasing = (
                            explanation_df[explanation_df["contribution"] > 0]
                            .sort_values("contribution", ascending=False)
                            .head(1)
                        )

                        top_reducing = (
                            explanation_df[explanation_df["contribution"] < 0]
                            .sort_values("contribution", ascending=True)
                            .head(1)
                        )

                        st.markdown("#### Plain-language interpretation")

                        if not top_increasing.empty:
                            st.write(
                                f"The largest risk-increasing model contribution for this specific customer is "
                                f"**{top_increasing.iloc[0]['feature']}**."
                            )

                        if not top_reducing.empty:
                            st.write(
                                f"The largest risk-reducing model contribution for this specific customer is "
                                f"**{top_reducing.iloc[0]['feature']}**."
                            )

                        st.caption(
                            "This explanation is local to the selected customer. "
                            "It does not mean that one feature always has the same effect for every customer."
                        )

                st.divider()
                show_retention_suggestions(
                    feature_df.iloc[0].to_dict(),
                    prediction_probability
                )

    except Exception as error:
        st.error("Customer lookup failed.")
        st.write(error)

        

# -----------------------------
# Bulk Prediction Page
# -----------------------------
elif page == "Bulk Prediction":
    st.subheader("Bulk CSV Prediction")
    st.write(
        "Upload a CSV file with the required customer feature columns. "
        "The app will return churn probability, predicted churn label, and risk level."
    )

    st.caption(
        "`Total Charges` may exist in the uploaded CSV, but the final v2 model does not use it. "
        "The app will ignore extra columns and use only the required model features."
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
            predictions = (probabilities >= FINAL_THRESHOLD).astype(int)

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
                mime="text/csv",
            )

        except Exception as error:
            st.error("Prediction failed. Please check whether your CSV contains all required columns.")
            st.write(error)


# -----------------------------
# Model Explainability Page
# -----------------------------
elif page == "Model Explainability":
    st.subheader("Model Explainability")
    st.write(
        "This page explains how the final Logistic Regression v2 model behaves globally. "
        "It is different from the single-customer explanation, which is local to one prediction."
    )

    st.markdown(
        """
        <div class="soft-card">
            <b>Final model design</b><br>
            Logistic Regression v2 removes <code>Satisfaction Score</code> to reduce leakage sensitivity
            and removes <code>Total Charges</code> to improve coefficient interpretability.
            The final classification threshold is <b>0.30</b>, selected after threshold tuning for the best F1 score.
        </div>
        """,
        unsafe_allow_html=True,
    )

    coef_df = get_coefficient_dataframe(model)

    if not coef_df.empty:
        top_coef_df = coef_df.reindex(
            coef_df["coefficient"].abs().sort_values(ascending=False).index
        ).head(15)

        inc_df = (
            coef_df[coef_df["coefficient"] > 0]
            .sort_values("coefficient", ascending=False)
            .head(5)
        )
        red_df = (
            coef_df[coef_df["coefficient"] < 0]
            .sort_values("coefficient", ascending=True)
            .head(5)
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔴 Strongest churn-increasing signals")
            st.caption("Positive coefficients push predictions toward churn.")
            for _, row in inc_df.iterrows():
                st.markdown(f"- **{row['feature']}** `{row['coefficient']:.3f}`")

        with col2:
            st.markdown("#### 🟢 Strongest churn-reducing signals")
            st.caption("Negative coefficients push predictions away from churn.")
            for _, row in red_df.iterrows():
                st.markdown(f"- **{row['feature']}** `{row['coefficient']:.3f}`")

        st.divider()

        fig = px.bar(
            top_coef_df.sort_values("coefficient"),
            x="coefficient",
            y="feature",
            orientation="h",
            color="direction",
            title="Top Logistic Regression Coefficients",
            labels={
                "coefficient": "Coefficient value",
                "feature": "Feature",
                "direction": "Direction",
            },
            color_discrete_map={
                "Increases churn probability": "#EF4444",
                "Reduces churn probability": "#22C55E",
            },
            template="plotly_dark",
        )

        fig.add_vline(
            x=0,
            line_width=1,
            line_dash="dash",
            line_color="gray",
        )

        fig.update_layout(
            height=560,
            legend_title_text="Coefficient direction",
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Important: these are global model coefficients, not direct causal effects. "
            "For one-hot encoded categorical variables, each coefficient is interpreted relative to the model's encoded baseline."
        )
    else:
        st.info("Coefficient explanation is not available for this pipeline structure.")


# -----------------------------
# About Page
# -----------------------------
elif page == "About Project":
    st.subheader("About This Project")

    st.write(
        """
        This project predicts customer churn for a telecommunications company using a structured data science workflow.

        Main stages:

        - data audit
        - SQL validation
        - exploratory data analysis
        - preprocessing and feature decisions
        - model comparison and tuning
        - leakage-aware final model selection
        - threshold tuning
        - Streamlit deployment
        """
    )

    st.write(
        """
        The final business-facing model is **Logistic Regression v2** without `Satisfaction Score` and `Total Charges`.

        `Satisfaction Score` was removed to reduce leakage sensitivity.
        `Total Charges` was removed to reduce multicollinearity risk and improve coefficient interpretability.

        The final classification threshold is set to **0.30** because it achieved the best F1 score during threshold tuning.
        This threshold supports a balanced retention strategy by capturing more potential churners while keeping campaign cost reasonable.
        """
    )

    st.markdown("**Author:** Gulsare Mirzayeva")
    st.markdown(f"[LinkedIn]({LINKEDIN_URL})")
