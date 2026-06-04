# Car Price Prediction using Linear Regression

This report shows how we built regression models to predict car prices from scratch. We implement everything using gradient descent and validate our results mathematically.

---

## Part A: Understanding the Data

We work with the CarDekho dataset containing used car information—engine size, year, mileage, and selling price.

### Data Cleaning

First, we load the data and handle basic issues:
- Convert `max_power` to numeric format
- Drop rows with missing values  
- Remove the `name` column (car names don't help predictions)
- Use **one-hot encoding** for categorical features (fuel type, transmission) to convert text into numbers

### Outlier Detection

We use the **IQR (Interquartile Range) method** to detect outliers mathematically:

Given data points sorted in order, we calculate:
- $Q_1$ = 25th percentile (first quartile)
- $Q_3$ = 75th percentile (third quartile)
- $IQR = Q_3 - Q_1$ (the middle 50% spread)

Then we define outlier boundaries:
$$\text{Lower Bound} = Q_1 - 1.5 \times IQR$$
$$\text{Upper Bound} = Q_3 + 1.5 \times IQR$$

Any data point $x$ where $x < \text{Lower Bound}$ or $x > \text{Upper Bound}$ is flagged as an outlier.

**Example:** If engine sizes have $Q_1 = 1200$ cc and $Q_3 = 1800$ cc:
- $IQR = 1800 - 1200 = 600$
- Lower = $1200 - 1.5(600) = 300$ cc
- Upper = $1800 + 1.5(600) = 2700$ cc
- Any car with engine < 300 cc or > 2700 cc is an outlier

We keep outliers since they might be legitimate luxury or budget cars.

### Correlation Analysis

We compute the **Pearson correlation coefficient** between each feature and price:

$$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Where $r_{xy}$ ranges from -1 to 1.

---

## Part B: Simple Linear Regression

We predict car prices using just **one feature: engine size**. The goal is to find the best-fitting straight line through the data.

### The Linear Model

We want to find a line of the form:

$$\hat{y} = \theta_0 + \theta_1 x$$

Where:
- $\hat{y}$ = predicted price
- $x$ = engine size
- $\theta_0$ = intercept (base price when engine = 0)
- $\theta_1$ = slope (price change per unit engine size increase)

After training on our dataset of 7906 cars, the model learned $\theta_0 = -0.042$ and $\theta_1 = 0.265$ (on normalized scale).

### Feature Scaling (Normalization)

We normalize data to the range [0, 1]:

$$x_{scaled} = \frac{x}{x_{max}}$$

**Why?** Gradient descent converges faster when features are on similar scales.

For instance, in our dataset the maximum engine size determines the scaling factor for all engine values.

### Matrix Formulation

To handle the intercept $\theta_0$ elegantly, we add a column of 1's to $X$. This way:
- $X$ becomes an $m \times 2$ matrix: first column is all 1's, second column is engine sizes
- $\theta$ is a $2 \times 1$ vector: $[\theta_0, \theta_1]$
- $y$ is an $m \times 1$ vector of prices

Now our model becomes: $\hat{y} = X\theta$

This matrix multiplication gives us predictions for all examples at once. Each row computes: $\hat{y}_i = 1 \times \theta_0 + x_i \times \theta_1 = \theta_0 + \theta_1 x_i$

### Cost Function (Mean Squared Error)

To measure how good our line is, we calculate the average squared error:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
- $m$ = number of training examples
- $\hat{y}^{(i)}$ = prediction for example $i$
- $y^{(i)}$ = actual price for example $i$
- Factor of $\frac{1}{2}$ simplifies derivative calculation

**In matrix form:**
$$J(\theta) = \frac{1}{2m}(X\theta - y)^T(X\theta - y)$$

**Example with 3 data points:**
If predictions are [8000, 9000, 7500] and actuals are [8200, 8800, 7700]:
- Errors: [−200, 200, −200]
- Squared errors: [40000, 40000, 40000]
- $J = \frac{1}{6}(40000 + 40000 + 40000) = 20000$

### Gradient Descent Algorithm

We minimize the cost function by iteratively adjusting $\theta$ in the direction that reduces $J$.

**Update rule:**
$$\theta := \theta - \alpha \nabla J(\theta)$$

Where:
- $\alpha$ = learning rate (step size) = 0.01
- $\nabla J(\theta)$ = gradient (direction of steepest increase)

**Computing the gradient:**

The partial derivatives of $J(\theta)$ with respect to $\theta$ are:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$

**In matrix form (much cleaner):**
$$\nabla J(\theta) = \frac{1}{m} X^T(X\theta - y)$$

**Step-by-step in each iteration (epoch):**

1. **Compute predictions:** 
   $$h = X\theta$$

2. **Compute errors:**
   $$error = h - y$$

3. **Compute cost:**
   $$J = \frac{1}{2m}\sum(error^2)$$

4. **Compute gradient:**
   $$\nabla = \frac{1}{m}X^T \cdot error$$

5. **Update parameters:**
   $$\theta := \theta - \alpha \nabla$$

We run this for **80,000 epochs**. The cost $J$ decreases each iteration until convergence. We plot cost vs epoch number to verify convergence.

---

## Part C: Multiple Linear Regression

Now we use **all features** simultaneously to make better predictions. Car prices depend on engine size, year, mileage, and more.

### The Extended Model

With $n$ features, our model becomes:

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + ... + \theta_n x_n$$

Our model uses 17 parameters (1 intercept + 16 features from one-hot encoded data) trained on 6324 training samples and tested on 1582 samples.

### Matrix Formulation

For $m$ examples and $n$ features, we construct:
- $X$: an $m \times (n+1)$ matrix with first column of 1's, then all features
- $\theta$: an $(n+1) \times 1$ vector of parameters
- $y$: an $m \times 1$ vector of target values

The model remains: $\hat{y} = X\theta$ (now with more columns in $X$)

### Train-Test Split

We randomly split data:
- **Training set (80%)**: Used to learn $\theta$ values
- **Test set (20%)**: Used to evaluate generalization

```python
np.random.seed(42)  # Reproducible randomness
p = np.random.permutation(len(y))  # Shuffle indices
split = int(0.8 * len(y))
X_train, X_test = X[p[:split]], X[p[split:]]
y_train, y_test = y[p[:split]], y[p[split:]]
```

### Gradient Descent

The math is identical to Part B, just with more dimensions:

1. Initialize: $\theta$ as vector of zeros (one zero per feature + intercept)

2. For each epoch (500 times):
   - $h = X_{train}\theta$ (predictions on training data)
   - $error = h - y_{train}$
   - $J = \frac{1}{2m}\sum error^2$ (cost on training data)
   - $\nabla = \frac{1}{m}X_{train}^T \cdot error$
   - $\theta := \theta - 0.01 \cdot \nabla$

### Evaluation Metrics

After training on training set, we evaluate on **both** training and test sets.

#### 1. Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Step-by-step calculation:**
1. For each example, compute error: $e_i = y_i - \hat{y}_i$
2. Square each error: $e_i^2$
3. Average all squared errors: $MSE = \frac{1}{n}\sum e_i^2$
4. Take square root: $RMSE = \sqrt{MSE}$

In our implementation, the test set achieved RMSE = 683,756 rupees, meaning predictions deviate by approximately 6.8 lakh rupees on average.

#### 2. R-Squared ($R^2$) — Coefficient of Determination

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Where:
- $SS_{res}$ = Residual Sum of Squares (our model's error)
- $SS_{tot}$ = Total Sum of Squares (variance in data)
- $\bar{y}$ = mean of actual values

**Breaking it down mathematically:**

1. **Calculate mean:** 
   $$\bar{y} = \frac{1}{n}\sum_{i=1}^{n}y_i$$

2. **Calculate $SS_{tot}$ (total variance to explain):**
   $$SS_{tot} = \sum_{i=1}^{n}(y_i - \bar{y})^2$$
   This measures how much prices vary from the average.

3. **Calculate $SS_{res}$ (our model's remaining error):**
   $$SS_{res} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
   This measures how much our predictions miss the actual values.

4. **Calculate $R^2$:**
   $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

In our multiple linear regression model:
- Training R²: 0.3416 (explained 34.16% of price variance)
- Test R²: 0.3308 (explained 33.08% of price variance)
- Generalization gap: only 1.09%, indicating good generalization

### Checking for Overfitting

We compare $R^2$ on training and test sets. Large gap between train and test $R^2$ indicates overfitting.

**Our results:**
- Training: RMSE = 655,480, R² = 0.3416
- Test: RMSE = 683,756, R² = 0.3308
- Gap: 1.09% (minimal overfitting)

---

## Part D: Polynomial Regression

Linear models assume straight-line relationships. But what if the relationship curves? We add **polynomial features** to capture non-linear patterns.

### Feature Engineering

We create new features by squaring existing ones:

$$X_{new} = [x_1, x_2, x_3, x_4, x_1^2, x_2^2, x_3^2, x_4^2]$$

For 4 base features (year, engine, max_power, km_driven), we add their squares.

**Example transformation:**
- Original: [2020, 1500, 120, 50000]
- Squared features: [2020², 1500², 120², 50000²] = [4,080,400, 2,250,000, 14,400, 2,500,000,000]
- Combined: [2020, 1500, 120, 50000, 4,080,400, 2,250,000, 14,400, 2,500,000,000]

### The Polynomial Model

With original features + squared features:

$$price = \theta_0 + \theta_1(year) + \theta_2(engine) + \theta_3(power) + \theta_4(km)$$
$$+ \theta_5(year^2) + \theta_6(engine^2) + \theta_7(power^2) + \theta_8(km^2)$$

**Why this captures curves:**

For example, to fit a curved relationship:

$$price = \theta_0 + \theta_1 \cdot engine + \theta_2 \cdot engine^2$$

If $\theta_2$ is negative, the relationship curves downward (diminishing returns). If positive, it accelerates upward.

Even though the relationship with engine is non-linear, it's still linear in $\theta$, so we can use the same gradient descent algorithm.

### Training

Identical process to Part C:
1. Scale all features (including squared ones)
2. Add intercept column  
3. Run gradient descent for 1000 epochs
4. Evaluate on train and test sets

**Our results with polynomial features (21 parameters):**
- Training: RMSE = 578,477, R² = 0.4872
- Test: RMSE = 603,859, R² = 0.4780
- Gap: 0.92% (excellent generalization)
- Improvement: R² increased from 33% to 48% by adding squared features

### Interpreting Coefficients

For feature $x_i$ with linear coefficient $\theta_i$ and quadratic coefficient $\theta_{i+n}$:

- If $\theta_{i+n} > 0$: Accelerating relationship (curve bends upward)
- If $\theta_{i+n} < 0$: Decelerating relationship (curve bends downward)  
- If $\theta_{i+n} \approx 0$: Relationship is mostly linear

---

## Part E: Regularization

We introduce regularization techniques to prevent overfitting and perform feature selection by penalizing large model coefficients. We implement both Ridge (L2) and Lasso (L1) regression from scratch.

### 1. Ridge Regression (L2 Regularization)

Ridge regression adds a penalty proportional to the sum of squared coefficients to the Mean Squared Error (MSE) cost function:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

Where:
- $\lambda$ (lambda) is the regularization strength.
- We do **not** regularize the intercept term ($\theta_0$).

#### Gradient descent update rule:
$$\theta_0 := \theta_0 - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \right)$$
$$\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} + \frac{\lambda}{m} \theta_j \right) \quad (\text{for } j \ge 1)$$

In matrix form:
$$\theta := \theta - \alpha \left( \frac{1}{m} X^T(X\theta - y) + \frac{\lambda}{m} \theta_{reg} \right)$$
Where $\theta_{reg} = [0, \theta_1, \theta_2, \dots, \theta_n]^T$.

---

### 2. Lasso Regression (L1 Regularization)

Lasso regression adds a penalty proportional to the sum of absolute values of the coefficients:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|$$

Since the absolute value function $|\theta_j|$ is not differentiable at $\theta_j = 0$, we use subgradient descent. The subgradient of $|\theta_j|$ is the sign function:

$$\text{sign}(\theta_j) = \begin{cases} 1 & \text{if } \theta_j > 0 \\ -1 & \text{if } \theta_j < 0 \\ 0 & \text{if } \theta_j = 0 \end{cases}$$

#### Subgradient descent update rule:
$$\theta_0 := \theta_0 - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \right)$$
$$\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} + \frac{\lambda}{m} \text{sign}(\theta_j) \right) \quad (\text{for } j \ge 1)$$

---

### 3. Ridge vs. Lasso: Feature Selection Comparison

- **Ridge Regression (L2)**: Shrinks the coefficients close to zero but keeps all features. It is ideal for handling collinearity/multicollinearity.
- **Lasso Regression (L1)**: Shrinks some coefficients to **exactly zero**, thereby performing automatic **feature selection** and leaving a simpler, more interpretable model.

Our training results with $\lambda = 1.0$ show that Lasso reduces the coefficient values of several categorical features (like specific brands or body types) to exactly 0, whereas Ridge retains them all with small values.

---


## Part F: Model Diagnostics

We validate our models by checking if they satisfy regression assumptions. Violating these can make our model unreliable even with good $R^2$.

### Three Models to Compare

1. **Simple Linear**: $price = \theta_0 + \theta_1(max\_power)$
2. **Polynomial**: $price = \theta_0 + \theta_1(max\_power) + \theta_2(max\_power^2)$
3. **Multiple Linear**: $price = \theta_0 + \theta_1(year) + \theta_2(engine) + ... + \theta_5(mileage)$

### Normal Equation (Closed-Form Solution)

Instead of gradient descent, we solve directly using linear algebra:

$$\theta = (X^TX)^{-1}X^Ty$$

**Matrix derivation:**

We want to minimize: $J(\theta) = \frac{1}{2m}(X\theta - y)^T(X\theta - y)$

Taking derivative and setting to zero:
$$\frac{\partial J}{\partial \theta} = \frac{1}{m}X^T(X\theta - y) = 0$$
$$X^TX\theta = X^Ty$$
$$\theta = (X^TX)^{-1}X^Ty$$

This gives optimal $\theta$ in one calculation. We add regularization $10^{-8}I$ to prevent numerical issues:
$$\theta = (X^TX + 10^{-8}I)^{-1}X^Ty$$

### Assumption 1: Linearity

Plot actual vs predicted prices. Points should scatter around the diagonal line $y = x$ with no systematic curve or pattern.

### Assumption 2: Normality of Residuals

Residuals are prediction errors: $e_i = y_i - \hat{y}_i$

Plot histogram of residuals. Should be bell-shaped (Gaussian) and centered at zero:
$$\frac{1}{n}\sum_{i=1}^{n}e_i \approx 0$$

### Assumption 3: Homoscedasticity (Constant Variance)

Plot residuals vs predicted values. Residuals should have even spread across all prediction values with no "funnel" shape and centered at zero.

### Assumption 4: Independence (Durbin-Watson Test)

**Test:** Check if consecutive residuals are correlated

$$DW = \frac{\sum_{i=2}^{n}(e_i - e_{i-1})^2}{\sum_{i=1}^{n}e_i^2}$$

**Calculating step-by-step:**

1. **Compute differences between consecutive residuals:**
   $$d_i = e_i - e_{i-1}$$ for $i = 2, 3, ..., n$

2. **Square and sum differences:**
   $$\text{Numerator} = \sum_{i=2}^{n}d_i^2 = \sum_{i=2}^{n}(e_i - e_{i-1})^2$$

3. **Square and sum residuals:**
   $$\text{Denominator} = \sum_{i=1}^{n}e_i^2$$

4. **Divide:**
   $$DW = \frac{\text{Numerator}}{\text{Denominator}}$$

**Interpretation:**
- DW ≈ 2: No autocorrelation
- DW < 1.5: Positive autocorrelation
- DW > 2.5: Negative autocorrelation

**Calculation example:**
For a sequence of residuals $e = [e_1, e_2, ..., e_n]$, we compute differences between consecutive residuals, square them, sum them up, and divide by the sum of squared residuals.

### Model Comparison

We calculate $R^2$ for all three models on both train and test sets to compare performance and check for overfitting.

---

## Part G: Classification Algorithms

We extend the project to classification tasks by implementing several core classifiers from scratch using NumPy and validating them on synthetic datasets.

### 1. Linear Discriminant Analysis (LDA)
LDA projects data into a lower-dimensional space to maximize class separability by finding the projection vector $w$ that maximizes the ratio of between-class variance ($S_B$) to within-class variance ($S_W$):
$$J(w) = \frac{w^T S_B w}{w^T S_W w}$$

### 2. Logistic Regression (Binary and Multinomial)
- **Binary Logistic Regression**: Maps inputs to probabilities using the Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ and minimizes cross-entropy loss using gradient descent.
- **Multinomial Logistic Regression (Softmax)**: Extends classification to $K$ classes using the Softmax activation function:
$$P(y = c | x) = \frac{e^{w_c^T x}}{\sum_{j=1}^{K} e^{w_j^T x}}$$

### 3. Naive Bayes Classifier (Gaussian)
Based on Bayes' Theorem, Gaussian Naive Bayes assumes that features are conditionally independent given the class label and follow a normal distribution:
$$P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\\sigma_y^2}\right)$$

### 4. Support Vector Machine (SVM)
We formulate SVM using Hinge Loss and RBF (Gaussian) kernels, optimized using subgradient descent:
$$J(w, b) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^{m} \max(0, 1 - y^{(i)}(w^T \Phi(x^{(i)}) + b))$$

---

## Part H: Decision Trees and Neural Networks

We implement advanced supervised models from scratch:

### 1. Decision Trees & Regression Trees
- **Decision Trees**: Classifies data by splitting nodes to maximize Information Gain based on Entropy:
$$\text{Entropy}(y) = -\sum p_i \log_2(p_i)$$
- **Regression Trees**: Predicts continuous values by splitting nodes to maximize Variance Reduction.
- **Stopping & Pruning**: We implement stopping criteria including `max_depth` and `min_samples_split`.

### 2. Artificial Neural Networks (ANN)
We implement a Multi-Layer Perceptron (MLP) from scratch:
- **Perceptron Learning Algorithm (PLA)**: Formulates weights for binary linearly separable data.
- **Backpropagation**: Calculates partial derivatives of the loss function with respect to weights using the chain rule and updates weights using gradient descent.
- **Initialization & Validation**: Implements Xavier/He weight initialization and tracks performance on a validation split.

---

## Part I: Unsupervised Learning & Association Mining

### 1. K-Means Clustering (Partitional)
Partitions data into $K$ distinct clusters by minimizing the within-cluster sum of squared distances to cluster centroids.

### 2. Agglomerative Hierarchical Clustering
Implements bottom-up hierarchical clustering using the single-linkage distance metric to merge clusters:
$$d(A, B) = \min \{ d(x, y) : x \in A, y \in B \}$$

### 3. Principal Component Analysis (PCA)
Performs dimensionality reduction by calculating the covariance matrix of centered features, finding its eigenvectors/eigenvalues, and projecting the data onto the top principal components.

### 4. Apriori Association Rule Mining
Mines frequent itemsets and generates association rules from transaction databases using:
- **Support**: $P(A \cup B)$
- **Confidence**: $P(B | A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$
- **Lift**: $\frac{\text{Support}(A \cup B)}{\text{Support}(A) \times \text{Support}(B)}$

---

## Part J: Evaluation Measures & Hypothesis Testing

### 1. Classification Metrics
Based on the Confusion Matrix (TP, FP, TN, FN), we compute:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

### 2. ROC (Receiver Operating Characteristic) Curve
Plots the True Positive Rate (TPR/Recall) against the False Positive Rate (FPR) at various decision thresholds:
$$\text{FPR} = \frac{FP}{FP + TN}$$

### 3. K-Fold Cross Validation
Splits the dataset into $K$ equal folds, iteratively training on $K-1$ folds and validating on the remaining fold to ensure robust generalization.

### 4. Bootstrapping
Generates new training sets of the same size as the original data by sampling with replacement to estimate model uncertainty.