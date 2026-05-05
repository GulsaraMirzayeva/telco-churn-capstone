# 📊 Telco Customer Churn Capstone Project

## 🎯 Project Overview

This is my end-to-end Data Science capstone project on Telco Customer Churn prediction.

The goal of the project is not only to build a model that predicts churn, but also to understand the business problem behind churn and turn the model into something usable.

I worked with a dataset of **7,043 customers** and followed a full data science workflow:

- data audit
- SQL validation
- exploratory data analysis
- preprocessing and feature decisions
- model comparison and tuning
- leakage-aware final model selection
- threshold tuning
- Streamlit deployment
- cloud PostgreSQL customer lookup

One of the main lessons of this project was that the model with the highest score is not always the best final answer. For the final recommendation, I focused on a model that is easier to explain, safer to defend, and more realistic for business use.

---

## 🔗 Live Demo

Streamlit App: [Open App](https://telco-churn-capstone-dd4vehvfvqbd5uy5vfch5p.streamlit.app/)

The deployed app supports:

- single customer churn prediction
- existing customer lookup from Neon PostgreSQL
- bulk CSV churn prediction
- churn probability output
- predicted churn label
- risk-level classification
- local model explanation
- recommended retention actions

The app is designed as a simple business-facing tool. A user can manually enter customer information, select an existing customer from the database, or upload a CSV file for bulk predictions.

---

## 📸 App Preview

The Streamlit app turns the final model into a business-facing prediction tool.  
It supports manual prediction, PostgreSQL customer lookup, bulk CSV prediction, model explanation, and recommended retention actions.

### 1. App Overview

![Streamlit App Overview](reports/images/streamlit_app_overview.png)

### 2. Single Prediction Result

![Prediction Result](reports/images/prediction_result.png)

### 3. Local Model Explanation Summary

![Model Explainability Summary](reports/images/model_explainability1.png)

### 4. Feature Contribution Chart

![Feature Contribution Chart](reports/images/model_explainability2.png)

### 5. Detailed Explainability View

![Detailed Explainability View](reports/images/model_explainability3.png)

### 6. Existing Customer Lookup from PostgreSQL

![Customer Lookup from PostgreSQL](reports/images/customer_lookup_postgresql1.png)

### 7. Customer Lookup Prediction Result

![Customer Lookup Prediction Result](reports/images/customer_lookup_postgresql2.png)

### 8. Recommended Retention Actions

![Recommended Retention Actions](reports/images/recommended_retention_actions.png)

--- 

## 📊 Tableau Dashboard Preview

I also created Tableau dashboards to present the churn analysis from a business decision-making perspective.

While the Streamlit app focuses on prediction and customer-level interaction, the Tableau dashboards are used as a reporting and monitoring layer.

The Tableau dashboards focus on:

- overall churn and retention patterns
- customer status and contract distribution
- churn reason patterns
- predicted high-risk customers
- revenue at risk
- geographic risk concentration
- action-oriented customer prioritization

The Tableau dataset was prepared from the PostgreSQL customer data and enriched with final model prediction outputs such as churn probability, predicted churn label, risk level, and revenue at risk.

### 1. Executive Overview

This dashboard summarizes the overall churn situation, including customer status, churn rate, retention rate, contract distribution, gender distribution, age distribution, and churn reason patterns.

![Tableau Executive Overview](reports/images/tableau_executive_overview.png)

### 2. Risk & Geo Insights

This dashboard focuses on predicted customer risk, revenue exposure, geographic churn hotspots, risk deciles, and the highest-risk customers requiring retention action.

![Tableau Risk and Geo Insights](reports/images/tableau_risk_geo_insights.png)

### Tableau Public Dashboard

The final Tableau dashboard is also available on Tableau Public:

[Open Tableau Dashboard](https://public.tableau.com/app/profile/gulsara.mirzayeva/viz/TelcoChurnExecutiveDashboard/Overview?publish=yes)

---

## 🛠 Tech Stack

- **Data Storage & Validation:** PostgreSQL, Neon PostgreSQL, SQLAlchemy
- **Data Processing:** Python, Pandas, NumPy
- **Visualization & Statistics:** Matplotlib, Seaborn, SciPy, Tableau
- **Dashboarding:** Tableau Public
- **Modeling:** scikit-learn, XGBoost
- **App & Deployment:** Streamlit, Streamlit Cloud, Plotly
- **Workflow:** Jupyter Notebooks, GitHub

---

## 📂 Project Workflow & Key Findings

### 1. Data Audit (`01_data_audit.ipynb`)

The first step was to understand the dataset structure before doing any modeling.

I checked:

- dataset shape
- column names and data types
- duplicate records
- target column
- churn-related columns
- missing-value patterns

The dataset contains **7,043 customer records**.

One early observation was that some missing values were not represented in the same way across the dataset. This was important because inconsistent missing-value representation can later affect both SQL validation and Python preprocessing.

---

### 2. SQL Validation (`02_sql_validation.ipynb`)

After the initial audit, I moved the raw data into PostgreSQL and used SQL for validation.

The goal was not to clean the full dataset in SQL, but to check whether the imported raw data was reliable.

Important checks included:

- row count validation
- previewing imported records
- checking missing-value representation
- comparing SQL results with Python results

One important finding was:

- `Offer` and `Internet Type` contained literal `"None"` strings
- `Churn Category` and `Churn Reason` used proper SQL NULL values

This helped me decide that SQL would be used as a validation layer, while modeling-related cleaning would be handled in Python.

---

### 3. Exploratory Data Analysis (`03_eda.ipynb`)

The EDA focused on understanding which customer groups are more likely to churn.

I checked churn patterns by:

- contract type
- internet type
- payment method
- tenure
- monthly charge
- senior citizen status
- combined customer segments

Some important patterns were:

- Month-to-month contract customers showed higher churn risk.
- Fiber optic customers appeared as an important churn-related segment.
- Customers with shorter tenure were more likely to churn.
- Customers with higher monthly charges showed higher churn tendency.
- Bank withdrawal and mailed check customers appeared riskier than credit card customers.
- Gender did not show a meaningful churn difference.

I also used a Chi-Square test for `Gender` vs `Churn Label`.

The p-value was **0.4866**, so I did not treat gender as a strong churn signal in this dataset.

Two important business segments were:

- **Fiber Optic + Month-to-Month**
- **Senior Citizen + Bank Withdrawal**

---

### 4. Preprocessing & Feature Decisions (`04_preprocessing_and_feature_decisions.ipynb`)

In this stage, I prepared the data for machine learning.

The preprocessing included:

- defining the target variable
- removing leakage-sensitive columns
- separating numerical and categorical features
- imputing missing values
- scaling numerical features
- one-hot encoding categorical features
- building a reproducible pipeline with `ColumnTransformer`

Some columns were removed because they were too close to the churn outcome or directly described churn after it happened.

Examples:

- `Customer Status`
- `Churn Category`
- `Churn Reason`
- `Churn Score`
- `Churn Label` as target, not feature

This step was important because the model should learn from customer behavior and profile data, not from columns that already explain the churn result.

---

### 5. Modeling & Evaluation (`05_modeling_and_evaluation.ipynb`)

I compared several classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

At this stage, I wanted to see how different models behave on the same preprocessing pipeline.

The models gave strong results, especially when high-impact features were included. However, I did not want to make the final decision only based on the highest score. I also considered interpretability and whether the model could be explained to a business user.

---

### 6. Model Tuning & Final Selection (`06_model_tuning_and_final_selection.ipynb`)

In the final modeling stage, I tuned the strongest candidate models and reviewed the final model decision more carefully.

Models tuned:

- Logistic Regression
- Random Forest
- XGBoost

One important finding was that `Satisfaction Score` had a very strong impact on model performance.

However, I treated `Satisfaction Score` carefully because it may not be safely available before churn in a real business setting. Because of that, I removed it from the final conservative model.

I also removed `Total Charges` from the final model because it is strongly related to:

- `Tenure in Months`
- `Monthly Charge`

Keeping all three features together can make Logistic Regression coefficients harder to interpret.

The final selected model is:

> **Logistic Regression v2 without `Satisfaction Score` and `Total Charges`**

The final model was selected because it gives a better balance between performance, interpretability, and business defensibility.

---

### 7. Tableau Business Dashboard

After deploying the Streamlit app, I created Tableau dashboards as a business-facing reporting layer.

The goal of the Tableau dashboards is different from the Streamlit app:

- Streamlit is used for churn prediction and customer-level interaction.
- Tableau is used for monitoring churn patterns, risk concentration, geographic hotspots, and revenue exposure.

The Tableau dashboards include:

- an executive overview of churn and customer segments
- a risk and geography dashboard based on model prediction outputs
- high-risk customer prioritization for retention actions

This helped connect the machine learning output with a clearer business decision-making view.

---

## Final Modeling View

This project does not end with “the model with the highest score wins.”

During the modeling stage, I saw that the expanded XGBoost model achieved the strongest raw performance when `Satisfaction Score` was included. But I did not want to make the final recommendation only based on that result.

`Satisfaction Score` may be too close to the churn decision or may not be available early enough in a real prediction setting. Because of that, I treated it as leakage-sensitive and removed it from the final conservative model.

I also removed `Total Charges` from the final model. This feature is strongly related to `Tenure in Months` and `Monthly Charge`, and keeping all three together can make Logistic Regression coefficients harder to interpret.

Final selected model:

> **Logistic Regression v2 without `Satisfaction Score` and `Total Charges`**

Final test performance:

| Metric | Value |
|---|---:|
| Best CV F1 | 0.6914 |
| Test Accuracy | 0.8417 |
| Test Precision | 0.7254 |
| Test Recall | 0.6497 |
| Test F1 | 0.6855 |
| Test ROC-AUC | 0.9016 |

After threshold tuning, I selected:

> **Final threshold = 0.30**

At this threshold:

| Metric | Value |
|---|---:|
| Precision | 0.6019 |
| Recall | 0.8449 |
| F1-score | 0.7030 |

I selected this threshold because churn prediction is a retention problem. Missing too many real churn customers can be costly, but campaign cost also matters.

Threshold `0.30` gave the best F1-score and a better balance between catching potential churners and avoiding unnecessary retention actions.

For this reason, the final model is selected not only because of performance, but because it is more interpretable, more defensible, and more realistic for business-facing use.

---

## 🚀 Streamlit App

The Streamlit app turns the final model into a simple business-facing prediction tool.

It includes three main workflows:

### 1. Single Prediction

The user can manually enter customer information and get:

- churn probability
- predicted churn label
- risk level
- model explanation
- recommended retention actions

### 2. Existing Customer Lookup

The app connects to **Neon PostgreSQL** and allows the user to select an existing customer from the database.

The workflow is:

1. select customer from PostgreSQL database
2. load customer profile
3. send customer features to the model
4. generate churn probability
5. show explanation and recommended action

In this workflow, `Customer ID` is not used as a model feature. It is only used to retrieve the customer profile from the database.

This makes the app closer to a real business workflow where an operator or retention team member looks up an existing customer from a CRM or database.

### 3. Bulk Prediction

The user can upload a CSV file with multiple customer records.

The app returns:

- churn probability
- predicted churn label
- risk level

The result can also be downloaded as a CSV file.

---

## 🧠 Model Explainability

Since the final model is Logistic Regression, I used coefficient-based explanation.

The app includes:

- global coefficient explanation
- local explanation for a selected customer
- factors increasing churn risk
- factors reducing churn risk

The explanation is presented carefully as model contribution, not direct causality.

This is important because a feature can behave differently in one specific prediction depending on the full encoded feature set. For that reason, the app explains that the contribution values are local to the selected customer.

---

## 📁 Project Structure

```text
telco-churn-capstone/
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
│       └── telco.csv
├── models/
│   ├── conservative_logistic_regression_pipeline.joblib
│   ├── conservative_logistic_regression_v2_pipeline.joblib
│   └── expanded_xgboost_pipeline.joblib
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_sql_validation.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_preprocessing_and_feature_decisions.ipynb
│   ├── 05_modeling_and_evaluation.ipynb
│   └── 06_model_tuning_and_final_selection.ipynb
├── reports/
│   └── images/
│       ├── streamlit_app_overview.png
│       ├── prediction_result.png
│       ├── model_explainability1.png
│       ├── model_explainability2.png
│       ├── model_explainability3.png
│       ├── customer_lookup_postgresql1.png
│       ├── customer_lookup_postgresql2.png
│       ├── recommended_retention_actions.png
│       ├── tableau_executive_overview.png
│       └── tableau_risk_geo_insights.png
├── scripts/
│   ├── upload_customer_lookup_to_neon.py
│   ├── check_neon_customer_table.py
│   └── build_tableau_dashboard_table.py
├── tableau/
│   ├── telco_churn_dashboard.twbx
│   └── telco_churn_tableau_dashboard_data.csv
├── requirements.txt
├── README.md
└── model_documentation.md
```

---

## 🚀 How to Run the Streamlit App Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

For the PostgreSQL customer lookup feature, a valid `DATABASE_URL` must be added to Streamlit secrets.

Local secrets should be stored in:

```text
.streamlit/secrets.toml
```

Example format:

```toml
DATABASE_URL = "postgresql+psycopg2://USER:PASSWORD@HOST/DATABASE?sslmode=require"
```

This file should not be pushed to GitHub.

---

## 📌 Current Project Status

Completed:

- [x] Data Audit & Initial Inspection
- [x] SQL Migration & Validation
- [x] Exploratory Data Analysis
- [x] Preprocessing & Feature Decisions
- [x] Baseline Model Comparison
- [x] Model Tuning
- [x] Leakage Sensitivity Analysis
- [x] Final Model Selection
- [x] Threshold Tuning
- [x] Model Packaging & Saving
- [x] Streamlit App Deployment
- [x] Neon PostgreSQL Customer Lookup
- [x] Single Prediction
- [x] Bulk CSV Prediction
- [x] Model Explainability in App
- [x] Model Documentation Summary
- [x] Tableau Executive Overview Dashboard
- [x] Tableau Risk & Geo Insights Dashboard
- [x] Tableau Reporting Dataset
- [x] Tableau Public Dashboard

Remaining:

- [ ] Final Presentation
- [ ] Final Notebook Cleanup
---

## ⏭ Next Steps

1. **Final Presentation:** Prepare a concise project story from business problem to deployed solution.
2. **Notebook Cleanup:** Review notebook markdowns and make sure the explanations are clear and not repetitive.
3. **Final Dashboard Review:** Review the Tableau Public dashboard and screenshots before final submission.
4. **Future Monitoring Note:** Add a short note about possible model monitoring, such as data drift and performance tracking.

---

## About Me

**Gulsare Mirzayeva**  
Data Science Trainee at Div Academy  
Aspiring Data Scientist | Python | SQL | Machine Learning  

[LinkedIn](https://www.linkedin.com/in/gulsara-mirzayeva/) | [Email](mailto:mirzayevagulsare@gmail.com)