# Customer Credit Risk Prediction

A customer-level machine learning project that combines demographic/application information, historical account behaviour, payment-history patterns, and credit-enquiry activity to identify customers associated with bad credit history.

> **Project status:** Exploratory analysis, feature engineering, preprocessing, and model evaluation are documented in the notebook. The selected Random Forest is a candidate decision-support model, not a production-ready credit decision system.

## 1. Business Problem

The objective is to help **Bank GoodCredit** identify customers who may be more likely to develop a poor credit history. The project frames this as a binary classification problem at the customer level.

The target variable is `Bad_label`:

| Value | Meaning |
|---|---|
| `0` | Good credit history |
| `1` | Bad credit history (30+ DPD bucket) |

The intended use is to support risk-based customer monitoring and prioritise cases for further review. Model scores should not independently determine credit approval, credit limits, or customer restrictions.

## 2. Data Sources

The notebook retrieves three source tables from a MySQL database:

| Source table | Records | Columns | Grain | Role |
|---|---:|---:|---|---|
| `Cust_Demographics` | 23,896 | 83 | One row per customer | Base dataset; contains `Bad_label` |
| `Cust_Account` | 186,329 | 21 | Multiple records per customer | Historical account and payment information |
| `Cust_Enquiry` | 413,188 | 6 | Multiple records per customer | Historical credit-enquiry information |

The three tables share `customer_no`. The notebook verifies that the account and enquiry tables cover the same 23,896 customers as the demographic table. Since account and enquiry tables have multiple records per customer, their records are aggregated before they are combined with the customer-level base dataset.

### Target distribution

The dataset is highly imbalanced:

| Class | Customers | Share |
|---|---:|---:|
| Good credit (`0`) | 22,892 | 95.80% |
| Bad credit (`1`) | 1,004 | 4.20% |

Because the positive class is rare, accuracy alone is not a sufficient measure of model quality. The notebook also evaluates Gini, ROC-AUC, PR-AUC, precision, recall, F1-score, and risk ranking.

## 3. Project Workflow

### A. MySQL extraction and data understanding

- Connect to MySQL using SQLAlchemy and PyMySQL.
- Inspect the source tables, column types, row counts, and customer identifiers.
- Export source tables to local CSV files under `data/raw/` for analysis.
- Load the exported tables into Pandas for subsequent analysis.

### B. Exploratory Data Analysis

The EDA examines data quality, feature distributions, relationships with `Bad_label`, and the structure of the account and enquiry records.

Areas investigated include:

- Target-class imbalance.
- Missing-value patterns and duplicate records.
- Numerical, categorical, coded, and date-related variables.
- Account types, balances, credit limits, and account duration.
- Payment-history strings containing three-character codes.
- Enquiry purpose, enquiry amounts, enquiry frequency, and enquiry recency.
- Potentially informative features, redundant columns, and possible leakage.

Some feature names are anonymised (`feature_1`, `feature_2`, and so on). Their underlying business definitions are not available in the project data, so interpretations are kept descriptive rather than assumed.

### C. Data preparation and preprocessing

The modelling dataset uses one row per customer, with `demographic_df` as the base table and `Bad_label` as the target.

Preprocessing decisions documented in the notebook include:

- Investigating missingness before choosing imputation strategies.
- Excluding extremely sparse or unsuitable features, including `feature_61`, `feature_18`, `feature_10`, `feature_17`, `feature_9`, `feature_8`, `feature_57`, and `feature_73`.
- Excluding `feature_49` and `feature_74` because they contain approximately 99.56% and 99.93% missing values, respectively.
- Removing a constant feature and redundant duplicate features.
- Excluding date-related fields from the initial model because their prediction-time availability and business meaning could not be established reliably.
- Removing `customer_no` from the predictors as an identifier.
- Retaining investigated outliers rather than automatically removing or capping them.

The preprocessing pipeline uses `ColumnTransformer` and Scikit-learn `Pipeline`:

| Feature type | Treatment |
|---|---|
| Continuous numerical | Median imputation with missingness indicators, followed by `RobustScaler` |
| Numerical categorical | Most-frequent imputation with missingness indicators |
| Categorical | Dedicated missing-value category and `OneHotEncoder` with infrequent-category grouping (`min_frequency=0.01`) |

Preprocessing is fitted on training data and then applied to held-out data to reduce data leakage. The notebook reports 374 transformed features after preprocessing.

### D. Customer-level feature engineering

Account and enquiry information is summarised to customer level before integration.

**Account features** include:

- Total number of accounts per customer.
- Number of distinct account types.
- Account counts by account type.
- Account counts by ownership indicator.
- Payment-history-derived features, including late-payment and DPD-related summaries.
- Balance-to-high-credit ratios.

The account-count aggregation produces 37 customer-level features and is validated to ensure that there is one record per customer.

**Enquiry features** include:

- Total enquiry count and distinct enquiry-purpose count.
- Total, mean, median, and maximum enquiry amounts.
- Enquiry frequency over 90-day and 365-day windows.
- Mean difference between account-opening and enquiry dates.
- An indicator for whether valid enquiry details are available.

The enquiry table contains 13,827 exact duplicate rows. These are removed for the enquiry feature-engineering step, reducing the record count from 413,188 to 399,361. The original source table is preserved.

After integrating demographic, account, enquiry, payment-history, and derived balance information, the engineered customer-level dataset contains **23,896 rows and 146 columns**, with one record per customer and no missing target values. Feature selection reduces the candidate predictor set to 132 columns.

## 4. Model Development

The notebook evaluates the following classifiers:

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- K-Nearest Neighbours (KNN)
- Support Vector Machine (SVM)

The modelling process includes class-imbalance-aware configurations, preprocessing pipelines, validation, threshold assessment, and a final held-out test evaluation.

A stratified 80/20 split is used with `random_state=42`:

| Split | Customers |
|---|---:|
| Training | 19,116 |
| Testing | 4,780 |

The Random Forest uses class weighting. A probability threshold of **0.40** is selected based on validation threshold analysis, then applied to the final test predictions.

## 5. Final Model and Test Results

The selected candidate is a **Random Forest Classifier**. Its final held-out test results are:

| Metric | Test result |
|---|---:|
| Accuracy | 81.36% |
| Precision | 8.73% |
| Recall | 36.32% |
| F1-score | 14.08% |
| ROC-AUC | 0.6924 |
| PR-AUC | 0.0725 |
| **Gini** | **38.49** |
| Project Gini benchmark | 37.90 |

Gini is calculated as `(2 × ROC-AUC − 1) × 100`. The final test Gini of **38.49** is 0.59 points above the project benchmark of 37.90.

### Confusion matrix

Rows are actual classes (`0`, `1`) and columns are predicted classes (`0`, `1`):

```text
[[3816  763]
 [ 128   73]]
```

### How to interpret the result

The model's ranking performance is better than random and its Gini exceeds the stated project benchmark. However, its classification performance remains limited:

- Precision of 8.73% means that only a small share of customers classified as bad-credit customers are actually in the bad-credit class.
- Recall of 36.32% means that the model identifies only a portion of the bad-credit customers and misses many of them.
- The target imbalance means that accuracy can appear high even when positive-class detection is weak.

The Random Forest should therefore be treated as a candidate for further validation, not as an operational credit decision system.

### Validation risk-ranking observation

A separate validation rank-ordering analysis grouped customers into predicted-risk deciles. The highest-risk decile had a 10.18% bad rate, and the top two deciles together captured 40.37% of validation bad-credit customers. These are validation-set ranking observations and should not be confused with the final test metrics above.

## 6. Business Insights

The analysis highlights several areas that may support risk assessment, subject to additional validation:

- **Account behaviour:** customers can have multiple accounts, so account exposure and account-type structure need to be summarised at customer level.
- **Payment history:** payment-history strings were converted into customer-level summaries such as late-payment counts, DPD counts, maximum DPD, and mean DPD. Code definitions should be confirmed against authoritative data documentation before operational interpretation.
- **Enquiry behaviour:** enquiry counts, recency, and amounts provide additional context about customer credit activity. Enquiry frequency and total enquiry amount had a strong positive Spearman correlation (0.7740) in the EDA.
- **Data quality:** missing financial information, unusual balances, date inconsistencies, and duplicate enquiry records require ongoing monitoring.

The notebook's feature-importance analysis identifies `feature_7` as the leading permutation-importance feature in the reported test analysis, followed by missingness in `feature_51`, `feature_52`, account-type-10 count, and `feature_12` category information. Feature importance indicates model dependence, not causation or confirmed business risk drivers.

## 7. Limitations and Further Work

1. **Class imbalance:** only 4.20% of customers are in the bad-credit class.
2. **Limited positive-class precision and recall:** the final model produces many false positives and still misses a substantial proportion of bad-credit customers.
3. **Benchmark margin:** the final Gini is only slightly above the project benchmark.
4. **Model comparison caveat:** not all candidate models in the notebook were evaluated on the same split. Their reported scores should not be treated as a strict apples-to-apples comparison. A consistent evaluation protocol is needed before making definitive superiority claims.
5. **Anonymised predictors:** the business meaning of many demographic features is unavailable.
6. **Temporal leakage risk:** some date and payment-history fields have uncertain prediction-time availability; their use requires a clearly defined prediction cutoff.
7. **Data quality:** missing values, unusual values, and the meaning of some coded values require further source-system documentation.
8. **Deployment readiness:** further validation, calibration, fairness assessment, explainability, stability monitoring, and governance would be required before any real-world use.

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SQLAlchemy
- PyMySQL
- python-dotenv
- MySQL
- Jupyter Notebook

The notebook records the following environment versions during execution: Pandas 2.2.3, NumPy 2.1.3, and Scikit-learn 1.6.1.

## 9. Setup and Execution

### Prerequisites

- Python installed.
- Access to the MySQL database containing the three project tables.
- Database permissions to read `Cust_Account`, `Cust_Enquiry`, and `Cust_Demographics`.

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn sqlalchemy pymysql python-dotenv jupyter
```

### Configure database access

Create a local `.env` file in the project working directory with your own database connection details:

```env
DB_HOST=your_host
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database
```

The notebook loads these variables with `python-dotenv` and connects to MySQL through SQLAlchemy/PyMySQL.

> **Security:** Never commit your real `.env` file, database password, credentials, or private customer-level data to a public repository. Keep `.env` out of Git (for example, add it to `.gitignore`). Only publish datasets if you have explicit permission and have confirmed that they contain no confidential or personally identifiable information.

### Run the notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `PR-0015_Credit_Risk_Prediction.ipynb` and run the cells in order. The initial database extraction section reads the three MySQL tables and saves local CSV copies under `data/raw/` before loading them for analysis.

## 10. Suggested Repository Structure

```text
PR-0015_Credit_Risk_Prediction/
├── PR-0015_Credit_Risk_Prediction.ipynb
├── README.md
├── .gitignore
└── data/
    └── raw/                 # Created locally by the notebook; do not publish private data
        ├── Cust_Account.csv
        ├── Cust_Enquiry.csv
        └── Cust_Demographics.csv
```

The `.env` file is intentionally not shown in this structure because it should remain local and untracked.

## 11. Conclusion

This project builds a customer-level credit-risk modelling workflow by combining demographic/application information with aggregated account, payment-history, and enquiry behaviour. The final Random Forest candidate achieves a test Gini of **38.49**, slightly exceeding the project benchmark of **37.90**.

The result demonstrates some ability to rank customers by risk, but the low precision and moderate recall show that the model is not ready to make independent credit decisions. Further consistent model comparison, temporal validation, model explainability, data-quality review, and governance are needed before operational use.
