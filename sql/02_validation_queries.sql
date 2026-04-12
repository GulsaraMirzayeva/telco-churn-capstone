SELECT COUNT(*) AS total_rows
FROM telco_customers;

SELECT *
FROM telco_customers
LIMIT 5;

SELECT
    COUNT(*) AS total_rows,
    COUNT(NULLIF(offer, 'None')) AS valid_offer_count,
    COUNT(NULLIF(internet_type, 'None')) AS valid_internet_type_count,
    COUNT(NULLIF(churn_category, 'None')) AS valid_churn_category_count,
    COUNT(NULLIF(churn_reason, 'None')) AS valid_churn_reason_count
FROM telco_customers;


-- During SQL validation, some missing categorical values were found to be stored as the literal string `'None'` rather than as SQL `NULL`.  
-- Because of this, validation queries used `NULLIF(column, 'None')` to count only meaningful non-missing values.


SELECT
    churn_label,
    COUNT(*) AS customer_count
FROM telco_customers
GROUP BY churn_label
ORDER BY customer_count DESC;