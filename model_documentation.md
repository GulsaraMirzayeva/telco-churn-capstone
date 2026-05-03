# Telco Customer Churn Model Documentation

## Project Overview

This project is a Telco Customer Churn prediction capstone project.

The main goal is to predict whether a customer is likely to churn and to turn that prediction into a business-facing decision support tool.

I used the Telco customer dataset with 7,043 customer records. The project was not built only as a modeling notebook. I tried to follow a full data science workflow:

- data audit
- SQL validation
- exploratory data analysis
- preprocessing and feature decisions
- model comparison
- model tuning
- leakage-aware model selection
- threshold tuning
- Streamlit deployment
- cloud PostgreSQL customer lookup

The final model is used inside a Streamlit app where users can make single predictions, bulk CSV predictions, and existing customer lookup predictions from a cloud PostgreSQL database.

---

## Business Problem

Customer churn is a business risk for telecom companies.

If a company can identify customers who are likely to leave, it can take action earlier. For example, the company can offer contract upgrades, review service quality, suggest better plans, or create retention campaigns.

For this reason, the project focuses not only on predicting churn, but also on making the result understandable for a business user.

The main question is:

> Which customers are more likely to churn, and what can the business do about it?

---

## Dataset

The dataset contains 7,043 customer records from a telecom company.

The target variable is:

- `Churn Label`

It is a binary classification problem:

- `Yes` = customer churned
- `No` = customer did not churn

The dataset includes customer profile, contract, service, billing, internet usage, and churn-related columns.

---

## SQL Storage and Validation

The raw dataset was first moved into PostgreSQL.

SQL was used mainly for raw data storage and validation, not for full preprocessing.

The main checks included:

- row count validation
- previewing imported records
- checking missing-value representations
- comparing SQL results with Python results

One important data quality observation was that missing values were represented differently across columns:

- `Offer` and `Internet Type` contained literal `"None"` values
- `Churn Category` and `Churn Reason` used proper SQL NULL values

After this check, I decided to keep SQL as the validation layer and handle modeling-related cleaning in Python.

Later, I also added Neon PostgreSQL as a cloud database for Streamlit customer lookup. This makes the deployed app closer to a real business workflow.

---

## Exploratory Data Analysis

The EDA focused on understanding churn patterns before modeling.

Main areas checked:

- churn distribution
- churn by contract type
- churn by internet type
- churn by payment method
- churn by tenure
- churn by monthly charge
- segment-level churn patterns

### Main Findings

Some important churn patterns were:

- Month-to-month contract customers showed higher churn risk.
- Fiber optic customers appeared as an important churn-related segment.
- Customers with shorter tenure were more likely to churn.
- Customers with higher monthly charges showed higher churn tendency.
- Bank withdrawal and mailed check customers appeared riskier than credit card customers.
- Gender did not show a meaningful churn difference.

A Chi-Square test was also used for `Gender` vs `Churn Label`. The p-value was 0.4866, so I did not treat gender as a strong churn signal.

The strongest business segment observed during EDA was:

- Fiber Optic + Month-to-Month

Another important segment was:

- Senior Citizen + Bank Withdrawal

---

## Preprocessing and Feature Decisions

The preprocessing stage focused on building a clean and reproducible machine learning pipeline.

### Main preprocessing steps

- defined the target variable
- normalized missing-like values
- separated numerical and categorical features
- used median imputation for numerical columns
- used most frequent imputation for categorical columns
- applied scaling to numerical columns
- applied one-hot encoding to categorical columns
- built the preprocessing with `ColumnTransformer`

### Leakage-sensitive columns removed

Some columns were removed because they are too close to the churn outcome or directly describe churn after it happened.

Examples:

- `Customer Status`
- `Churn Category`
- `Churn Reason`
- `Churn Score`
- `Churn Label` as target, not feature

I also reviewed `Satisfaction Score` separately because it had a very strong impact on model performance. Since it may not be safely available before churn in a real business setting, I treated it as leakage-sensitive and removed it from the final conservative model.

---

## Model Selection

I compared several classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The goal was not only to find the highest score. I also considered whether the final model could be explained and defended in a business setting.

The expanded XGBoost model achieved the strongest raw performance when `Satisfaction Score` was included.

However, for the final business-facing recommendation, I selected a conservative Logistic Regression model.

The reason is that Logistic Regression is easier to interpret, easier to explain, and more suitable for a cautious deployment-oriented conclusion.

---

## Final Model Decision

The final selected model is:

> Logistic Regression v2 without `Satisfaction Score` and `Total Charges`

### Why `Satisfaction Score` was removed

`Satisfaction Score` had a very strong impact on model performance.

However, it may be too close to the churn decision or may not be available early enough in a real prediction setting. Because of that, I removed it from the final conservative model to reduce leakage risk.

### Why `Total Charges` was removed

`Total Charges` is strongly related to:

- `Tenure in Months`
- `Monthly Charge`

Keeping all three features together can make Logistic Regression coefficients harder to interpret.

Since the goal of the final model is interpretability and defensibility, I removed `Total Charges` from the final v2 model.

In the Streamlit app, `Total Charges` is shown only as a calculated business context value:

> Estimated Total Charges = Tenure × Monthly Charge

It is not used as a model input.

---

## Final Model Performance

The final Logistic Regression v2 model achieved:

| Metric | Value |
|---|---:|
| Best CV F1 | 0.6914 |
| Test Accuracy | 0.8417 |
| Test Precision | 0.7254 |
| Test Recall | 0.6497 |
| Test F1 | 0.6855 |
| Test ROC-AUC | 0.9016 |

The ROC-AUC remained around 0.90, which shows that the model still separates churn and non-churn customers well.

Although the F1 score slightly decreased after removing `Total Charges`, the model became easier to explain and defend.

---

## Threshold Selection

The default classification threshold of 0.50 was not assumed to be the best business decision point.

Since churn prediction is a retention problem, I tested different thresholds and compared precision, recall, and F1-score.

The selected final threshold is:

> 0.30

At this threshold:

| Metric | Value |
|---|---:|
| Precision | 0.6019 |
| Recall | 0.8449 |
| F1-score | 0.7030 |

I selected threshold 0.30 because it gave the best F1-score and captured more potential churn customers.

This is suitable for a balanced retention strategy because missing too many real churn customers can be costly, but precision still matters because retention campaigns also have a cost.

---

## Deployment

The final model is deployed using Streamlit Cloud.

The app supports:

- single customer prediction
- existing customer lookup from Neon PostgreSQL
- bulk CSV prediction
- churn probability output
- predicted churn label
- risk level classification
- local model explanation
- recommended retention actions

The app uses the saved Logistic Regression v2 pipeline from the `models/` folder.

The deployed app also connects to Neon PostgreSQL for existing customer lookup. In this workflow, Customer ID is not used as a model feature. It is only used to retrieve the customer profile from the database.

---

## Streamlit App Features

### 1. Single Prediction

The user can manually enter the most important customer information.

The app returns:

- churn probability
- predicted churn label
- risk level
- explanation of the prediction
- retention suggestions

### 2. Existing Customer Lookup

The app connects to Neon PostgreSQL and allows the user to select an existing customer.

The flow is:

1. select customer from database
2. load customer profile
3. send customer features to the model
4. generate churn probability
5. show explanation and recommended action

This simulates a real business workflow where an operator or retention team member looks up a customer from a CRM or database.

### 3. Bulk Prediction

The user can upload a CSV file with multiple customers.

The app returns:

- churn probability
- predicted churn
- risk level

The user can also download the prediction results as a CSV file.

---

## Model Explainability

Since the final model is Logistic Regression, I used coefficient-based explanation.

The app shows:

- global feature coefficient chart
- local explanation for a selected customer
- factors increasing churn risk
- factors reducing churn risk

The explanation is presented carefully as model contribution, not direct causality.

For example, a feature may reduce risk in one specific prediction because of how it interacts with the full encoded feature set. Therefore, the app explains that these values should be read as model contributions for the selected customer.

---

## Project Structure

```text
telco-churn-capstone/
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
│       └── telco.csv
├── models/
│   └── conservative_logistic_regression_v2_pipeline.joblib
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_sql_validation.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_preprocessing_and_feature_decisions.ipynb
│   ├── 05_modeling_and_evaluation.ipynb
│   └── 06_model_tuning_and_final_selection.ipynb
├── scripts/
│   ├── upload_customer_lookup_to_neon.py
│   └── check_neon_customer_table.py
├── requirements.txt
├── README.md
└── model_documentation.md

```
---

## How to Run the App Locally

To run the project locally, first install the required libraries:

```bash
pip install -r requirements.txt
```

Then start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

The app will open in the browser and can be used for:

- single customer prediction
- existing customer lookup
- bulk CSV prediction
- model explanation

For the PostgreSQL customer lookup feature, a valid `DATABASE_URL` is required.

Locally, this should be stored in:

```text
.streamlit/secrets.toml
```

Example format:

```toml
DATABASE_URL = "postgresql+psycopg2://USER:PASSWORD@HOST/DATABASE?sslmode=require"
```

This file is intentionally ignored by Git and should not be pushed to GitHub.

---

## Key Dependencies

The main libraries used in this project are:

- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- SQLAlchemy
- psycopg2-binary
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- SciPy

The full dependency list is available in `requirements.txt`.

---

## Key Decisions Summary

| Decision | Why I made this decision |
|---|---|
| Used SQL for raw validation | I wanted to first check whether the imported data was consistent before moving to modeling. |
| Used Python for preprocessing and modeling | This made the machine learning pipeline easier to reproduce and update. |
| Removed churn outcome columns | These columns describe churn after it happened, so using them would create leakage. |
| Removed `Satisfaction Score` from the final model | It improved performance a lot, but I treated it carefully because it may not be safely available before churn. |
| Removed `Total Charges` from the final model | It is strongly related to tenure and monthly charge, so I removed it to make Logistic Regression easier to interpret. |
| Selected Logistic Regression v2 | I preferred a model that is easier to explain and defend, not only the model with the highest raw score. |
| Used threshold `0.30` | This threshold gave the best F1-score and helped catch more potential churn customers. |
| Deployed with Streamlit | I wanted the model to work as an actual user-facing tool, not only inside a notebook. |
| Added Neon PostgreSQL lookup | This made the app closer to a real business workflow where existing customers are retrieved from a database. |

---

## Current Status

At this stage, the main data science and deployment parts of the project are completed.

Completed work:

- data audit
- SQL validation
- exploratory data analysis
- preprocessing and feature decisions
- baseline model comparison
- model tuning
- final model selection
- threshold tuning
- model saving
- Streamlit deployment
- Neon PostgreSQL customer lookup
- single prediction
- bulk prediction
- model explanation inside the app

Remaining work:

- Tableau dashboard
- final presentation
- README final polish
- notebook markdown cleanup
- final screenshots and demo materials

---

## Limitations and Next Steps

The final model is intentionally conservative.

I did not simply choose the model or feature setup with the highest score. I tried to choose the version that would be easier to explain in a business setting.

`Satisfaction Score` improved model performance, but I removed it from the final model because it may create leakage risk. In a real company, such a score may not always be available before the customer actually decides to leave.

I also removed `Total Charges` because it overlaps strongly with `Tenure in Months` and `Monthly Charge`. Since the final model is Logistic Regression, keeping the coefficients understandable was important.

The next steps are:

1. Build a Tableau dashboard for churn segment analysis.
2. Add final screenshots and app link to the README.
3. Review notebook markdowns and remove repeated or overly generic explanations.
4. Prepare the final presentation.
5. Add a short note about possible future monitoring, such as data drift and performance tracking.

---

## Final Reflection

This project helped me understand that model performance alone is not enough.

At first, it was tempting to focus only on the strongest model score. But during the project, I saw that a model also needs to be explainable, defendable, and realistic for business use.

That is why I kept two views in the project:

- the expanded model view, which shows how strong performance can become when high-impact features are included
- the conservative final model view, which is safer for explanation and deployment

My final choice was Logistic Regression v2 because it gives a reasonable balance between performance, interpretability, and business usefulness.

The main lesson for me was that a data science project is not only about building a model. It is also about making clear decisions, explaining trade-offs, and turning the result into something that another person can actually use.