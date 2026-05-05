# Final Capstone Presentation Outline

## 1. Business Problem
Customer churn reduces recurring revenue. The goal is to identify customers likely to churn and support earlier retention actions.

## 2. Data
The project uses 7,043 Telco customer records.

## 3. SQL Validation
Raw data was stored and validated in PostgreSQL. Row count and missing-value patterns were checked.

## 4. EDA Findings
- Month-to-month contracts showed higher churn.
- Fiber optic customers appeared as an important churn-related segment.
- Short-tenure customers were more likely to churn.
- Payment method and senior status created important customer segments.

## 5. Modeling
Multiple models were compared: Logistic Regression, Decision Tree, Random Forest, and XGBoost.

## 6. Final Model
Final model: Logistic Regression v2 without Satisfaction Score and Total Charges.
Final threshold: 0.30.

## 7. Deployment
Streamlit app supports manual prediction, PostgreSQL lookup, bulk prediction, explainability, and retention recommendations.

## 8. Tableau Dashboard
Tableau provides executive churn overview, revenue-at-risk view, geographic risk, and high-risk customer prioritization.

## 9. Final Recommendation
Use the model as a retention support tool, not as an automatic decision-maker.