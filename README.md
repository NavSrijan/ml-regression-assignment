# ML Regression Assignment

This project implements various linear regression techniques on a car dataset to predict prices.

## Dataset
- **File:** `cleaned_global_cars_dataset.csv`
- **Size:** 300 samples
- **Features:** horsepower, engine_cc, mileage_km_per_l, manufacture_year, brand, body_type, fuel_type, transmission, manufacturing_country
- **Target:** price_usd

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

