import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"

MODEL_PATH = PROJECT_ROOT / "models" / "conservative_logistic_regression_v2_pipeline.joblib"
TABLEAU_DIR = PROJECT_ROOT / "tableau"
TABLEAU_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_TABLE = "telco_customer_lookup"
DASHBOARD_TABLE = "telco_churn_dashboard"
OUTPUT_CSV = TABLEAU_DIR / "telco_churn_tableau_dashboard_data.csv"

FINAL_THRESHOLD = 0.30


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


MODEL_FEATURES = list(DB_TO_MODEL_COLUMN_MAP.values())


def load_database_url() -> str:
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            "Missing .streamlit/secrets.toml. DATABASE_URL is required."
        )

    with open(SECRETS_PATH, "rb") as file:
        secrets = tomllib.load(file)

    database_url = secrets.get("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL was not found in .streamlit/secrets.toml")

    return database_url


def create_db_engine():
    database_url = load_database_url()

    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 20},
    )


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    df[object_cols] = df[object_cols].replace("None", np.nan)
    df[object_cols] = df[object_cols].replace(r"^\s*$", np.nan, regex=True)

    return df


def build_model_input(customer_df: pd.DataFrame) -> pd.DataFrame:
    model_input = customer_df.rename(columns=DB_TO_MODEL_COLUMN_MAP).copy()

    model_input = normalize_missing_values(model_input)

    model_input["Offer"] = model_input["Offer"].fillna("No Offer")
    model_input["Internet Type"] = model_input["Internet Type"].fillna("No Internet")

    return model_input[MODEL_FEATURES]


def assign_risk_level(probability: pd.Series) -> pd.Series:
    return np.select(
        [
            probability >= 0.70,
            probability >= FINAL_THRESHOLD,
        ],
        [
            "High Risk",
            "Medium Risk",
        ],
        default="Low Risk",
    )


def create_action_segment(df: pd.DataFrame) -> pd.Series:
    return np.select(
        [
            (df["risk_level"] == "High Risk") & (df["monthly_charge"] >= 70),
            (df["risk_level"] == "High Risk"),
            (df["risk_level"] == "Medium Risk") & (df["contract"] == "Month-to-Month"),
            (df["risk_level"] == "Medium Risk"),
        ],
        [
            "Priority: High Value + High Risk",
            "High Risk",
            "Monitor: Month-to-Month",
            "Monitor",
        ],
        default="Low Priority",
    )


def create_recommendation(row: pd.Series) -> str:
    recommendations = []

    if row.get("contract") == "Month-to-Month":
        recommendations.append("Offer an incentive to switch to a one-year or two-year contract")

    if row.get("internet_type") == "Fiber Optic":
        recommendations.append("Review service quality, speed complaints, and support history")

    if row.get("premium_tech_support") == "No":
        recommendations.append("Offer a trial premium support package")

    if row.get("payment_method") in ["Bank Withdrawal", "Mailed Check"]:
        recommendations.append("Encourage a more stable or convenient payment method")

    if row.get("tenure_in_months", 0) <= 12:
        recommendations.append("Apply early-life retention follow-up")

    if not recommendations:
        return "No urgent action. Continue regular customer monitoring"

    return " | ".join(recommendations)


def main():
    print("Connecting to Neon PostgreSQL...")
    engine = create_db_engine()

    print(f"Reading source table: {SOURCE_TABLE}")
    customer_df = pd.read_sql_query(
        text(f'SELECT * FROM "{SOURCE_TABLE}"'),
        engine,
    )

    print(f"Rows loaded: {len(customer_df)}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print("Loading final Logistic Regression v2 model...")
    model = joblib.load(MODEL_PATH)

    print("Preparing model input...")
    model_input = build_model_input(customer_df)

    print("Generating churn predictions...")
    churn_probability = model.predict_proba(model_input)[:, 1]
    predicted_churn = np.where(churn_probability >= FINAL_THRESHOLD, "Yes", "No")

    dashboard_df = customer_df.copy()

    dashboard_df["churn_probability"] = churn_probability
    dashboard_df["predicted_churn"] = predicted_churn
    dashboard_df["risk_level"] = assign_risk_level(dashboard_df["churn_probability"])
    dashboard_df["churn_flag"] = dashboard_df["churn_label"].map({"Yes": 1, "No": 0})
    dashboard_df["predicted_churn_flag"] = dashboard_df["predicted_churn"].map({"Yes": 1, "No": 0})

    dashboard_df["prediction_result_type"] = np.select(
        [
            (dashboard_df["churn_flag"] == 1) & (dashboard_df["predicted_churn_flag"] == 1),
            (dashboard_df["churn_flag"] == 0) & (dashboard_df["predicted_churn_flag"] == 1),
            (dashboard_df["churn_flag"] == 1) & (dashboard_df["predicted_churn_flag"] == 0),
            (dashboard_df["churn_flag"] == 0) & (dashboard_df["predicted_churn_flag"] == 0),
        ],
        [
            "True Positive",
            "False Positive",
            "False Negative",
            "True Negative",
        ],
        default="Unknown",
    )

    dashboard_df["revenue_at_risk"] = dashboard_df["churn_probability"] * dashboard_df["monthly_charge"]
    dashboard_df["expected_annual_loss"] = dashboard_df["revenue_at_risk"] * 12

    dashboard_df["high_risk_revenue"] = np.where(
        dashboard_df["risk_level"] == "High Risk",
        dashboard_df["monthly_charge"],
        0,
    )

    dashboard_df["tenure_group"] = pd.cut(
        dashboard_df["tenure_in_months"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49-72 months"],
    ).astype(str)

    dashboard_df["monthly_charge_group"] = pd.cut(
        dashboard_df["monthly_charge"],
        bins=[-1, 35, 65, 95, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)

    dashboard_df["senior_label"] = dashboard_df["senior_citizen"].map({
        0: "Non-Senior",
        1: "Senior",
    })

    dashboard_df["internet_contract_segment"] = (
        dashboard_df["internet_type"].astype(str) + " + " + dashboard_df["contract"].astype(str)
    )

    dashboard_df["senior_payment_segment"] = (
        dashboard_df["senior_label"].astype(str) + " + " + dashboard_df["payment_method"].astype(str)
    )

    dashboard_df["action_segment"] = create_action_segment(dashboard_df)
    
    dashboard_df["recommended_action"] = dashboard_df.apply(create_recommendation, axis=1)

    dashboard_columns = [
        "customer_id",
        "gender",
        "age",
        "senior_citizen",
        "senior_label",
        "married",
        "number_of_dependents",
        "number_of_referrals",
        "tenure_in_months",
        "tenure_group",
        "offer",
        "customer_status",
        "city",
        "state",
        "zip_code",
        "latitude",
        "longitude",
        "churn_category",
        "churn_reason",
        "cltv",
        "internet_service",
        "internet_type",
        "contract",
        "payment_method",
        "paperless_billing",
        "monthly_charge",
        "monthly_charge_group",
        "total_charges",
        "total_revenue",
        "satisfaction_score",
        "churn_label",
        "churn_flag",
        "churn_probability",
        "predicted_churn",
        "predicted_churn_flag",
        "risk_level",
        "prediction_result_type",
        "revenue_at_risk",
        "expected_annual_loss",
        "high_risk_revenue",
        "internet_contract_segment",
        "senior_payment_segment",
        "action_segment",
        "recommended_action",
    ]

    dashboard_df = dashboard_df[dashboard_columns].copy()


    print(f"Writing dashboard table to PostgreSQL: {DASHBOARD_TABLE}")
    dashboard_df.to_sql(
        DASHBOARD_TABLE,
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=500,
    )

    print(f"Exporting dashboard CSV: {OUTPUT_CSV}")
    dashboard_df.to_csv(OUTPUT_CSV, index=False)

    with engine.connect() as connection:
        row_count = connection.execute(
            text(f'SELECT COUNT(*) FROM "{DASHBOARD_TABLE}"')
        ).scalar()

    print(f"Dashboard table created: {DASHBOARD_TABLE}")
    print(f"Rows written: {row_count}")
    print("Tableau dashboard dataset is ready.")


if __name__ == "__main__":
    main()