# 📊 Telco Customer Churn Capstone Project

## 🎯 Project Overview

This project is an end-to-end data science capstone focused on predicting customer churn for a telecommunications company. Using a dataset of **7,043 customers**, the project follows a structured workflow including data auditing, SQL-based validation, exploratory data analysis, preprocessing, model comparison, tuning, leakage-aware sensitivity analysis, and deployment preparation.

The goal is not only to build a predictive model, but also to demonstrate a clear, reproducible, and business-oriented data science process.

---

## 🔗 Live Demo

Streamlit App: [Open App](https://telco-churn-capstone-dd4vehvfvqbd5uy5vfch5p.streamlit.app/)

The deployed app supports:

- single customer churn prediction
- bulk CSV churn prediction
- churn probability output
- risk-level classification

---

## 🛠 Tech Stack

- **Data Storage & Validation:** PostgreSQL, SQLAlchemy
- **Data Processing:** Python, Pandas, NumPy
- **Visualization & Statistics:** Matplotlib, Seaborn, SciPy
- **Modeling:** scikit-learn, XGBoost
- **Workflow:** Jupyter Notebooks
- **Deployment:** Streamlit Cloud

---

## 📂 Project Workflow & Key Findings

### 1. Data Audit (`01_data_audit.ipynb`)
Initial inspection of the dataset structure and column quality.

- **Key Action:** Verified dataset dimensions (7,043 rows), reviewed data types, checked duplicates, and profiled churn-related columns.
- **Observation:** Missing-value representations were inconsistent across some categorical variables.

### 2. SQL Validation (`02_sql_validation.ipynb`)
Validated the raw dataset after moving it into PostgreSQL.

- **Key Discovery:**  
  - `Offer` and `Internet Type` contained literal **"None"** strings  
  - `Churn Category` and `Churn Reason` used proper **SQL NULL**
- **Outcome:** Python and SQL validation results matched, confirming a reliable raw-data pipeline.

### 3. Exploratory Data Analysis (`03_eda.ipynb`)
Explored the main business and behavioral patterns associated with churn.

- **Key Insight:** Higher-risk churn segments included **Fiber Optic + Month-to-Month** and **Senior Citizen + Bank Withdrawal**.
- **Statistical Check:** A Chi-Square test was conducted for `Gender` vs `Churn`. With a **p-value of 0.4866**, no statistically significant association was found, suggesting that gender is not a strong churn signal in this dataset.

### 4. Preprocessing & Feature Decisions (`04_preprocessing_and_feature_decisions.ipynb`)
Prepared the baseline modeling dataset and documented preprocessing choices.

- **Key Action:** Defined the target variable, normalized missing-like values, and separated feature groups for preprocessing.
- **Leakage Review:** High-risk leakage-sensitive columns such as `Customer Status`, `Churn Category`, `Churn Reason`, and `Churn Score` were excluded from the modeling dataset.

### 5. Modeling & Evaluation (`05_modeling_and_evaluation.ipynb`)
Built the first reproducible modeling pipeline and compared multiple classification models.

- **Models Compared:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Result:** Strong baseline performance was observed, with Random Forest leading in cross-validation and Logistic Regression showing strong balance on the test set.

### 6. Model Tuning & Final Comparison (`06_model_tuning_and_final_selection.ipynb`)
Tuned the strongest candidate models and performed an additional sensitivity review.

- **Models Tuned:** Logistic Regression, Random Forest, XGBoost
- **Key Finding:** `Satisfaction Score` had a dominant effect on model performance.
- **Sensitivity Result:**  
  - With `Satisfaction Score`, tuned XGBoost delivered the strongest predictive performance  
  - Without `Satisfaction Score`, tuned Logistic Regression was selected as the recommended conservative model because it offers a better balance between performance, interpretability, and defensibility.
- **Project Interpretation:** The project reports both:
  - an **expanded high-performance view**
  - a **conservative leakage-aware view**

---

## 🚀 How to Run the Streamlit App Locally

After cloning the repository and installing the required packages, run the Streamlit app from the project root:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 🚀 Current Project Status

- [x] Data Audit & Initial Inspection
- [x] SQL Migration & Validation
- [x] Exploratory Data Analysis (EDA)
- [x] Preprocessing & Feature Decisions
- [x] Baseline Model Comparison
- [x] Model Tuning
- [x] Leakage Sensitivity Analysis
- [x] Model Packaging & Saving
- [x] Streamlit App Deployment
- [ ] README Final Polish
- [ ] Tableau Dashboard
- [ ] Final Presentation

---

## ⏭ Next Steps

1. **Tableau Dashboard:** Build a dashboard to summarize churn patterns and key business insights.
2. **Model Interpretation in App:** Add selected model metrics or visual outputs to the Streamlit app.
3. **README & Portfolio Polish:** Finalize documentation, screenshots, and project presentation materials.
4. **Final Presentation:** Prepare a concise project story from business problem to deployed solution.

---

## Final Modeling View

This project does not reduce the outcome to a single oversimplified final claim.

Instead, it presents two final model views:

- **Expanded Model:** tuned XGBoost with `Satisfaction Score`, offering stronger predictive performance
- **Conservative Model:** tuned Logistic Regression without `Satisfaction Score`, offering a more defensible leakage-aware result

For real deployment and business-facing reporting, the **conservative model** is treated as the primary recommendation until the timing and deployability of `Satisfaction Score` can be fully verified.

---

## About Me

**Gulsare Mirzayeva**  
Aspiring Data Scientist | Python | SQL | Machine Learning  
[LinkedIn](...) | [Email](mailto:...)

---