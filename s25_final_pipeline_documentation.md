# Credit Card Approvals Data Pipeline Documentation

## Pipeline Overview

This pipeline preprocesses the Credit Card Approvals dataset, focusing on numerical features to prepare them for machine learning modeling. It applies outlier detection and treatment, followed by feature scaling. Categorical features like 'Gender', 'PriorDefault', 'Employed', and 'DriversLicense' are assumed to be already in a suitable binary (0/1) format and are not processed by this specific pipeline. Similarly, this pipeline does not perform missing value imputation, as indicated by the dataset's state at this stage.

```python
approvals_transformer = Pipeline(steps=[
    # Gender: already categorical 0 or 1
    # Age: numerical, so we might transform to normalize it then apply tukey for outliers
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('scale_age', CustomRobustTransformer(target_column='Age')),
    # Debt: numerical, so normalize and apply tukey
    ('tukey_debt', CustomTukeyTransformer(target_column='Debt', fence='outer')),
    ('scale_debt', CustomRobustTransformer(target_column='Debt')),
    # YearsEmployed numerical
    ('tukey_years_employed', CustomTukeyTransformer(target_column='YearsEmployed', fence='outer')),
    ('scale_years_employed', CustomRobustTransformer(target_column='YearsEmployed')),
    # PriorDefault is already categorical 0 or 1
    # Employed is already categorical 0 or 1
    # CreditScore is numerical
    ('tukey_credit_score', CustomTukeyTransformer(target_column='CreditScore', fence='outer')),
    ('scale_credit_score', CustomRobustTransformer(target_column='CreditScore')),
    # DriversLicense is already categorical 0 or 1
    # Income is numerical
    ('tukey_income', CustomTukeyTransformer(target_column='Income', fence='outer')),
    ('scale_income', CustomRobustTransformer(target_column='Income')),

    # There are no missing values to impute
    ], verbose=True)
```

## Step-by-Step Design Choices

The pipeline processes the following numerical features: 'Age', 'Debt', 'YearsEmployed', 'CreditScore', and 'Income'. For each, it applies a Tukey Transformer for outlier treatment and then a Robust Transformer for scaling.

### 1. Age: Outlier Treatment & Scaling

- **Transformer 1.1:** `CustomTukeyTransformer(target_column='Age', fence='outer')`
  - **Design Choice:** Tukey method with 'outer' fence.
  - **Rationale:** To identify and cap extreme outliers in the 'Age' column. The 'outer' fence (Q1 - 3*IQR, Q3 + 3*IQR) is chosen to be less aggressive, targeting only the most extreme values, which is suitable for a feature like age that can have a wide but legitimate range.
- **Transformer 1.2:** `CustomRobustTransformer(target_column='Age')`
  - **Design Choice:** Robust scaling.
  - **Rationale:** To scale the 'Age' feature using statistics that are robust to outliers (median and interquartile range). This is preferred over standard scaling, especially when the data may not be normally distributed or when outliers might still influence mean/standard deviation.

### 2. Debt: Outlier Treatment & Scaling

- **Transformer 2.1:** `CustomTukeyTransformer(target_column='Debt', fence='outer')`
  - **Design Choice:** Tukey method with 'outer' fence.
  - **Rationale:** To manage extreme values in the 'Debt' column, which can vary significantly. The 'outer' fence helps in normalizing the distribution without overly constricting the data.
- **Transformer 2.2:** `CustomRobustTransformer(target_column='Debt')`
  - **Design Choice:** Robust scaling.
  - **Rationale:** To scale the 'Debt' feature robustly, minimizing the impact of potential outliers on the scaling process and preparing it for models sensitive to feature magnitudes.

### 3. YearsEmployed: Outlier Treatment & Scaling

- **Transformer 3.1:** `CustomTukeyTransformer(target_column='YearsEmployed', fence='outer')`
  - **Design Choice:** Tukey method with 'outer' fence.
  - **Rationale:** Addresses extreme outliers in 'YearsEmployed'. This feature can have a right-skewed distribution, and the outer fence helps cap very high values.
- **Transformer 3.2:** `CustomRobustTransformer(target_column='YearsEmployed')`
  - **Design Choice:** Robust scaling.
  - **Rationale:** Provides robust scaling for 'YearsEmployed', making it comparable with other features and suitable for various machine learning algorithms.

### 4. CreditScore: Outlier Treatment & Scaling

- **Transformer 4.1:** `CustomTukeyTransformer(target_column='CreditScore', fence='outer')`
  - **Design Choice:** Tukey method with 'outer' fence.
  - **Rationale:** Manages extreme outliers in 'CreditScore'. Credit scores can have a defined range but outliers might still exist due to data entry errors or unusual cases.
- **Transformer 4.2:** `CustomRobustTransformer(target_column='CreditScore')`
  - **Design Choice:** Robust scaling.
  - **Rationale:** Scales 'CreditScore' using median and IQR, which is appropriate for scores that might not follow a perfect normal distribution.

### 5. Income: Outlier Treatment & Scaling

- **Transformer 5.1:** `CustomTukeyTransformer(target_column='Income', fence='outer')`
  - **Design Choice:** Tukey method with 'outer' fence.
  - **Rationale:** Handles extreme outliers in the 'Income' feature, which is often highly skewed. The outer fence helps in mitigating the influence of very high incomes.
- **Transformer 5.2:** `CustomRobustTransformer(target_column='Income')`
  - **Design Choice:** Robust scaling.
  - **Rationale:** Robustly scales the 'Income' feature, making it more suitable for modeling, especially given its potential skewness and the presence of outliers.

## Pipeline Execution Order Rationale

1.  **Outlier Treatment Before Scaling:** For each numerical feature, `CustomTukeyTransformer` is applied before `CustomRobustTransformer`. This order is crucial because the parameters for scaling (like median and IQR for RobustScaler) can be skewed by the presence of extreme outliers. By treating outliers first, the subsequent scaling is based on a more representative distribution of the data.
2.  **Sequential Processing per Feature:** The pipeline processes each numerical feature with its pair of transformations (Tukey then Robust) sequentially.

## Performance Considerations

- **Robustness to Outliers:** The choice of `CustomTukeyTransformer` (especially with `fence='outer'`) and `CustomRobustTransformer` throughout the pipeline emphasizes a strategy that is robust to outliers. This helps in creating a more stable and reliable preprocessing pipeline, particularly for datasets where outliers are common or their impact needs to be minimized.
- **No Imputation Step:** The pipeline explicitly notes, "There are no missing values to impute." This simplifies the pipeline for this specific dataset context, assuming missing values have been handled prior or are not present in the `reduced_df` that this pipeline processes.
- **Pre-processed Categorical Features:** Features like 'Gender', 'PriorDefault', 'Employed', and 'DriversLicense' are commented as "already categorical 0 or 1". This implies they do not require encoding within this pipeline, streamlining its focus on numerical transformations.
