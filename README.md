# Telco Churn Capstone Project

## Project Overview
[cite_start]This project aims to predict customer churn for a telecommunications company using a dataset of 7,043 customers[cite: 102]. [cite_start]The project demonstrates a full-stack data science workflow, from data engineering in SQL to model deployment via Streamlit[cite: 30, 31, 26].

## Tech Stack
- [cite_start]**Database:** PostgreSQL (Data storage and initial validation) [cite: 66, 30]
- [cite_start]**Analysis & Modeling:** Python (Pandas, Scikit-Learn) [cite: 73, 19]
- [cite_start]**Deployment:** Streamlit Cloud [cite: 26]
- [cite_start]**Visualization:** Tableau (Bonus) and Seaborn/Matplotlib [cite: 62, 19]

## Project Progress & Key Discoveries

### Stage 1: Data Engineering & SQL Validation
[cite_start]The raw dataset was first pushed to a PostgreSQL database to ensure data integrity and demonstrate database management skills[cite: 30, 66].

**Key Validation Findings:**
- [cite_start]**Row Count:** Verified that all 7,043 rows were successfully imported, matching the initial pandas audit[cite: 102].
- [cite_start]**Missing Value Inconsistency:** A critical discovery was made regarding how missing values are represented[cite: 324]:
    - [cite_start]`offer` and `internet_type` columns contain literal **"None"** text strings[cite: 326].
    - [cite_start]`churn_category` and `churn_reason` columns use standard **SQL NULL** values[cite: 326].
- [cite_start]**Target Variable:** The `churn_label` was validated as the primary target for modeling, with a distribution of 5,174 "No" and 1,869 "Yes" cases[cite: 321, 319, 320].

### Stage 2: Preprocessing Strategy (Planned)
[cite_start]Based on the SQL audit, the preprocessing pipeline will implement column-specific handling[cite: 327]:
- [cite_start]Convert "None" strings to actual null values[cite: 298].
- [cite_start]Implement a `ColumnTransformer` to handle different imputation needs for categorical features[cite: 19].
- [cite_start]Remove potential data leakage variables (e.g., specific IDs or redundant churn indicators)[cite: 19].

## Current Status
- [x] [cite_start]Dataset Selection & SQL Integration [cite: 30]
- [x] [cite_start]SQL Data Audit & Validation [cite: 323]
- [ ] Exploratory Data Analysis (EDA)
- [ ] Machine Learning Pipeline Development
- [ ] Model Evaluation & Hyperparameter Tuning
- [ ] Streamlit App Deployment

---
