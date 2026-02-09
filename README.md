# ML Regression Assignment

This project implements various linear regression techniques on a car dataset to predict prices.

## Dataset
- **File:** `cleaned_global_cars_dataset.csv`
- **Size:** 300 samples
- **Features:** horsepower, engine_cc, mileage_km_per_l, manufacture_year, brand, body_type, fuel_type, transmission, manufacturing_country
- **Target:** price_usd


## Part A: Exploratory Data Analysis (EDA)

**Objective:** Analyze and understand the dataset before applying linear regression models.

### Implementation
- **Dataset Loaded:** cleaned_global_cars_dataset.csv
- **Basic Information:** Displayed first 5 rows, dataset shape, column names, and data types
- **Summary Statistics:** Calculated mean, standard deviation, minimum, maximum, and quartile values for numerical features
- **Missing Values Check:** Verified absence of null values in all columns
- **Outlier Detection:** Identified potential outliers using the IQR (Interquartile Range) method
- **Notebook:** `part_a_eda.ipynb`

### Results
- The dataset contains multiple numerical and categorical features suitable for regression analysis.
- No missing values were found in the dataset.
- No significant outliers were detected in numerical features.
- Summary statistics show reasonable distribution of values across features.

### Visualization
- Histograms plotted to observe distribution of numerical features.
- Correlation heatmap generated to analyze relationships between numerical variables.
- Horsepower shows relatively stronger positive correlation with price_usd compared to other features.


## Part B: Simple Linear Regression

**Objective:** Build a simple linear regression model using one feature to predict car prices.

### Implementation
- **Feature Selected:** horsepower
- **Model:** Linear Regression
- **Notebook:** `part_b_simple_linear_regression.ipynb`

### Results
- **Model Equation:** price = 23.59 × horsepower + 53101.69
- **Interpretation:**
  - **Slope (β1):** 23.59 - For every 1 unit increase in horsepower, the price increases by $23.59
  - **Intercept (β0):** 53101.69 - Base price when horsepower is 0

### Visualization
- Scatter plot showing relationship between horsepower and price
- Red regression line fitted to the data points



