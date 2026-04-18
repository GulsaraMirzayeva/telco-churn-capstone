# 📊 Telco Customer Churn Capstone Project



## 🎯 Project Overview

This project is an end-to-end data science initiative aimed at predicting customer churn for a telecommunications company. By analyzing a dataset of **7,043 customers**, this project demonstrates a full-stack workflow—ranging from initial data auditing and SQL-based validation to exploratory data analysis (EDA) and predictive modeling.



## 🛠 Tech Stack

* **Data Engineering:** SQL (PostgreSQL), Python (Pandas), SQLAlchemy

* **Analysis & Stats:** SciPy (Chi-Square Testing), Seaborn, Matplotlib

* **Pipeline:** Jupyter Notebooks

* **Deployment (Upcoming):** Streamlit Cloud



## 📂 Project Workflow & Key Discoveries



### 1. Data Audit (`01_data_audit.ipynb`)

Initial inspection of the dataset structure.

* **Key Action:** Verified dataset dimensions (7,043 rows) and identified target variables vs. potential leakage columns.

* **Observation:** Noted inconsistencies in missing data representations across different categorical columns.



### 2. SQL Validation (`02_sql_validation.ipynb`)

Ensuring data integrity by migrating the dataset to a PostgreSQL database.

* **Key Discovery (The "None" vs. NULL issue):** Identified a critical nuance:

    * `offer` and `internet_type` columns contain literal **"None"** text strings.

    * `churn_category` and `churn_reason` use standard **SQL NULL** values.

* **Outcome:** Confirmed that Python-based audits and SQL query results match perfectly (5,174 "No" vs. 1,869 "Yes"), ensuring the pipeline is reliable.



### 3. Exploratory Data Analysis (`03_eda.ipynb`)

Deep-dive statistical analysis into factors influencing churn.

* **Key Insight:** There is a strong correlation between **Senior Citizen status + Bank Withdrawal payment method**, with a churn risk of approximately **45%**.

* **Statistical Significance:** Conducted a Chi-Square test on `Gender` vs. `Churn`. With a **p-value of 0.4866**, we statistically rejected the hypothesis that gender influences churn in this dataset.



---



## 🚀 Project Status

- [x] Data Audit & Initial Inspection

- [x] SQL Migration & Data Validation

- [x] Exploratory Data Analysis (EDA)

- [ ] Preprocessing (ColumnTransformer & Imputation)

- [ ] Machine Learning Pipeline (Random Forest/XGBoost)

- [ ] Model Evaluation & Deployment

- [ ] Streamlit App Deployment



## ⏭ Next Steps

1.  **Preprocessing:** Implement `ColumnTransformer` to normalize "None" strings to actual nulls and encode categorical features.

2.  **Feature Engineering:** Create interaction features (e.g., `Senior_Payment_Group`) to capture the risk factors identified in the EDA phase.

3.  **Modeling:** Train and evaluate classification models to optimize for recall in high-risk segments.



---

*Created by Gulsara*



***



