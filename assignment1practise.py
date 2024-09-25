#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

power_data=pd.read_csv("Energydata export 18-09-2024 20-05-47.csv",parse_dates=['ts'])
power_data.set_index('ts', inplace=True)

weather_data=pd.read_csv("Energydata export 19-09-2024 12-41-31.csv",parse_dates=['ts'])
weather_data.set_index('ts', inplace=True)

power_data = power_data.resample('h').ffill()
weather_data = weather_data.resample('h').ffill()

power_data.reset_index('ts', inplace=True)
weather_data.reset_index('ts', inplace=True)



# In[ ]:





# # Random Forest

# In[15]:


power_data = power_data[['ts', 'Snorrebakken sterled Active Power | sno_ost_effekt | 804131']]
df = pd.merge(power_data, weather_data, on='ts', how='inner')

# Step 2: Set timestamp as the index and resample to hourly frequency (ensures consistent hourly data)
df.set_index('ts', inplace=True)
df = df.resample('h').ffill()  # Forward fill any missing timestamps

# Step 3: Feature Engineering - Lag and Rolling Features
df['prev_day_wind_power1'] = df['Snorrebakken sterled Active Power | sno_ost_effekt | 804131'].shift(24)
df['prev_hour_wind_power'] = df['Snorrebakken sterled Active Power | sno_ost_effekt | 804131'].shift(1)
df['wind_power_05_quantile'] = df['Snorrebakken sterled Active Power | sno_ost_effekt | 804131'].rolling(window=168).quantile(0.05).shift(24)
df['wind_power_50_quantile'] = df['Snorrebakken sterled Active Power | sno_ost_effekt | 804131'].rolling(window=168).quantile(0.50).shift(24)
df['wind_power_95_quantile'] = df['Snorrebakken sterled Active Power | sno_ost_effekt | 804131'].rolling(window=168).quantile(0.95).shift(24)

# Step 4: Time-Based Features (day of week, month, hour) with cyclical transformations
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['hour'] = df.index.hour

# Convert time features into cyclical format
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Step 5: Drop NaN values caused by shifts and rolling windows
df.dropna(inplace=True)

# Step 6: Define features (X) and target (y)
X = df[['prev_day_wind_power1', 'prev_hour_wind_power',
        'wind_power_05_quantile', 'wind_power_50_quantile', 'wind_power_95_quantile',
        'Observed mean humidity past hour at Hammer Odde Fyr - DMI station 06193 | 9F7P/7Q/XC/DMI/metObs/humidity_past1h/06193 | 406576',
        'Observed mean temperature past hour at Hammer Odde Fyr - DMI station 06193 | 9F7P/7Q/XC/DMI/metObs/temp_mean_past1h/06193 | 406560',
        'Observed mean wind direction the past hour at Hammer Odde Fyr - DMI station 06193 | 9F7P/7Q/XC/DMI/metObs/wind_dir_past1h/06193 | 406624',
        'Observed mean wind speed the past hour at Hammer Odde Fyr - DMI station 06193 | 9F7P/7Q/XC/DMI/metObs/wind_speed_past1h/06193 | 406640',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']]

y = df['Snorrebakken sterled Active Power | sno_ost_effekt | 804131']

# Step 7: Train-Test Split (no shuffling for time series data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Step 8: Pipeline - Standardization and RandomForest Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),   # Standardize the features
    ('regressor', RandomForestRegressor(random_state=42))  # RandomForest model
])

# Step 9: Define GridSearchCV parameters
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 15],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='r2')

# Step 10: Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Step 11: Output best parameters
print("Best parameters found:", grid_search.best_params_)

# Step 12: Make predictions using the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# Step 13: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

# Step 14: Plot Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit Line')
plt.xlabel('Actual Wind Power')
plt.ylabel('Predicted Wind Power')
plt.title('Actual vs. Predicted Wind Power')
plt.legend()
plt.grid(True)
plt.show()


# In[35]:


import matplotlib.pyplot as plt

# Assuming X is your dataset (for example, a list or numpy array)
# X = [your dataset here]

# Creating the histogram
plt.hist(y, bins=30, edgecolor='black')

# Adding title and labels
plt.title('Histogram of Dataset X')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Display the plot
plt.show()



# # check for overfitting

# In[17]:


# Assuming your pipeline is already defined and trained on X_train, y_train

# Step 1: Make predictions on both the training and test sets
y_train_pred = grid_search.best_estimator_.predict(X_train)  # Predictions on the training set
y_test_pred = grid_search.best_estimator_.predict(X_test)    # Predictions on the test set

# Step 2: Evaluate the model's performance on the training set
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

print("Training Set Performance:")
print("Mean Absolute Error (MAE):", train_mae)
print("Mean Squared Error (MSE):", train_mse)
print("Root Mean Squared Error (RMSE):", train_rmse)
print("R-squared (R2) Score:", train_r2)

# Step 3: Evaluate the model's performance on the test set
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTest Set Performance:")
print("Mean Absolute Error (MAE):", test_mae)
print("Mean Squared Error (MSE):", test_mse)
print("Root Mean Squared Error (RMSE):", test_rmse)
print("R-squared (R2) Score:", test_r2)


# In[ ]:


X_intercept = np.c_[np.ones(X_train.shape[0]), X_train]  # Add intercept term
XtX = X_intercept.T @ X_intercept
condition_number = np.linalg.cond(XtX)
print("Condition number of X^T X:", condition_number)


# # Linear Regression Using Residual Sum of squares Normal Equation
# 

# In[19]:


from sklearn.base import BaseEstimator, RegressorMixin

# Step 1: Define the custom Normal Equation-based Linear Regression class
class NormalEquationLinearRegression(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        # Add intercept term (column of ones) to X
        X_intercept = np.c_[np.ones(X.shape[0]), X]
        # Normal Equation: Beta = (X^T * X)^(-1) * X^T * y
        self.beta_ = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ y
        return self
    
    def predict(self, X):
        # Add intercept term (column of ones) to X
        X_intercept = np.c_[np.ones(X.shape[0]), X]
        return X_intercept @ self.beta_

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

# Step 3: Build the pipeline with StandardScaler and Normal Equation Linear Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('normal_eq_lr', NormalEquationLinearRegression())  # Normal Equation Linear Regression
])

# Step 4: Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Step 5: Make predictions using the trained model
y_pred = pipeline.predict(X_test)

# Step 6: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)


# # Linear Regression Using Gradient
# 

# In[21]:


# Step 1: Gradient Descent Implementation
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m, n = X.shape
    # Initialize parameters (weights) to zero
    beta = np.zeros(n + 1)  # n features + 1 intercept term
    X_intercept = np.c_[np.ones(m), X]  # Add intercept term (column of ones)

    for iteration in range(n_iterations):
        # Compute predictions
        y_pred = X_intercept @ beta
        # Compute the error
        error = y_pred - y
        # Compute the gradient
        gradient = (1 / m) * X_intercept.T @ error
        # Update the coefficients using gradient descent
        beta = beta - learning_rate * gradient
    
    return beta

# Step 2: Predict using learned beta values
def predict(X, beta):
    X_intercept = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
    return X_intercept @ beta





# Split the dataset into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

# Step 4: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Apply Gradient Descent on Training Data
learning_rate = 0.1
n_iterations = 10000
beta = gradient_descent(X_train_scaled, y_train, learning_rate, n_iterations)

# Step 6: Make predictions using the trained model
y_pred = predict(X_test_scaled, beta)

# Step 7: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

# Print the learned coefficients (intercept + slopes)
print("Learned coefficients (beta):", beta)
print('Predicted values',y_pred)


# In[ ]:





# In[ ]:





# # Normal Equation Using Ridge Regularization

# In[25]:


# Step 1: Define the Ridge-Regularized Normal Equation-based Linear Regression class
class RidgeNormalEquationLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):  # alpha is the regularization parameter (λ)
        self.alpha = alpha
    
    def fit(self, X, y):
        m, n = X.shape
        X_intercept = np.c_[np.ones(m), X]  # Add intercept term (column of ones)
        
        # Ridge Normal Equation: Beta = (X^T * X + α * I)^(-1) * X^T * y
        # I is the identity matrix, α is the regularization parameter
        I = np.eye(X_intercept.shape[1])  # Identity matrix (size = number of features + 1)
        I[0, 0] = 0  # Do not regularize the intercept term

        # Compute the ridge-regularized coefficients (β)
        self.beta_ = np.linalg.inv(X_intercept.T @ X_intercept + self.alpha * I) @ X_intercept.T @ y
        return self
    
    def predict(self, X):
        X_intercept = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
        return X_intercept @ self.beta_


# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

# Step 4: Build the pipeline with StandardScaler and Ridge-Regularized Normal Equation Linear Regression
alpha_value = 1.0  # Regularization strength (λ)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('ridge_normal_eq_lr', RidgeNormalEquationLinearRegression(alpha=alpha_value))  # Ridge Normal Equation Linear Regression
])

# Step 5: Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Step 6: Make predictions using the trained model
y_pred = pipeline.predict(X_test)

# Step 7: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

# Print the learned coefficients (intercept + slopes)
ridge_normal_eq_model = pipeline.named_steps['ridge_normal_eq_lr']
print("Learned coefficients (beta):", ridge_normal_eq_model.beta_)

print('Predicted values',y_pred)

X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
XtX = np.dot(X_train_scaled.T, X_train_scaled)

# Create identity matrix with the same shape as XtX
I = np.eye(XtX.shape[0])

# Add the regularization term (alpha * I) to XtX
XtX_regularized = XtX + alpha_value * I

# Step 9: Calculate the condition number of the regularized matrix
condition_number_regularized = np.linalg.cond(XtX_regularized)

print("Condition Number of regularized X^T X after applying Ridge:", condition_number_regularized)


# In[ ]:





# # HeatMap to view multicolinearities

# In[45]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your pandas DataFrame
X = df[['prev_day_wind_power1', 'prev_hour_wind_power',
        'wind_power_05_quantile', 'wind_power_50_quantile', 'wind_power_95_quantile',
        'Observed mean humidity past hour at Hammer Odde Fyr - DMI station 06193 | 9F7P/7Q/XC/DMI/metObs/humidity_past1h/06193 | 406576',
        'Observed mean temperature past hour at Hammer Odde Fyr - DMI station 06193 | 9F7P/7Q/XC/DMI/metObs/temp_mean_past1h/06193 | 406560',
        'Observed mean wind direction the past hour at Hammer Odde Fyr - DMI station 06193 | 9F7P/7Q/XC/DMI/metObs/wind_dir_past1h/06193 | 406624',
        'Observed mean wind speed the past hour at Hammer Odde Fyr - DMI station 06193 | 9F7P/7Q/XC/DMI/metObs/wind_speed_past1h/06193 | 406640',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']]

y = df['Snorrebakken sterled Active Power | sno_ost_effekt | 804131']

# Concatenating X and y into a single DataFrame
data = pd.concat([X, y], axis=1)

# Calculating the correlation matrix
corr_matrix = data.corr()

# Creating the correlation heatmap
plt.figure(figsize=(12, 8))  # Adjust the size to fit all columns
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Adding title
plt.title('Correlation Heatmap between Features and Target')

# Display the plot
plt.show()


# # Normal Equation Using Lasso Regularization

# In[27]:


from sklearn.linear_model import Lasso


# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

# Step 3: Build the pipeline with StandardScaler and Lasso Regression
alpha_value = 0.001  # Regularization strength (λ)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('lasso', Lasso(alpha=alpha_value, max_iter=10000, random_state=42))  # Lasso Regression with L1 regularization
])

# Step 4: Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Step 5: Make predictions using the trained model
y_pred = pipeline.predict(X_test)

# Step 6: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Lasso Regression Performance:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

# Step 7: Print the learned coefficients
lasso_model = pipeline.named_steps['lasso']
print("Learned coefficients (Lasso):", lasso_model.coef_)


# In[74]:


# Step 1: Apply the scaler (StandardScaler) to the training data



# # Finding optimal alpha using cross validation with GridSearch

# In[29]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

# Step 2: Define the Lasso pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('lasso', Lasso(max_iter=10000, random_state=42))  # Lasso model
])

# Step 3: Define a range of alpha values to search
param_grid = {
    'lasso__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]  # Test a range of alpha values
}

# Step 4: Use GridSearchCV to search for the best alpha value
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Step 5: Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Step 6: Output the best alpha and the corresponding performance
print("Best alpha value:", grid_search.best_params_)
print("Best score (MSE):", -grid_search.best_score_)  # Use negative because scikit-learn maximizes by default

# Step 7: Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model's performance on the test set
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Best Lasso Model Performance on Test Set:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)


# # Checking train and test metrics to evaluate for overfitting

# In[31]:


# Assuming your pipeline is already defined and trained on X_train, y_train

# Step 1: Make predictions on both the training and test sets
y_train_pred = best_model.predict(X_train)  # Predictions on the training set
y_test_pred = best_model.predict(X_test)    # Predictions on the test set

# Step 2: Evaluate the model's performance on the training set
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

print("Training Set Performance:")
print("Mean Absolute Error (MAE):", train_mae)
print("Mean Squared Error (MSE):", train_mse)
print("Root Mean Squared Error (RMSE):", train_rmse)
print("R-squared (R2) Score:", train_r2)

# Step 3: Evaluate the model's performance on the test set
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTest Set Performance:")
print("Mean Absolute Error (MAE):", test_mae)
print("Mean Squared Error (MSE):", test_mse)
print("Root Mean Squared Error (RMSE):", test_rmse)
print("R-squared (R2) Score:", test_r2)


# # using polynomial features

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

# Step 3: Define the pipeline with PolynomialFeatures, StandardScaler, and LinearRegression
degree = 2  # Degree of the polynomial, you can adjust this
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),  # Generate polynomial features
    ('scaler', StandardScaler()),  # Standardize the features
    ('linear_regression', LinearRegression())  # Apply Linear Regression on transformed features
])

# Step 4: Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Step 6: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Polynomial Regression Performance:")
print(f"Polynomial Degree: {degree}")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)


# In[ ]:


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit Line')
plt.xlabel('Actual Wind Power')
plt.ylabel('Predicted Wind Power')
plt.title(f'Actual vs. Predicted Wind Power (Polynomial Degree {degree})')
plt.legend()
plt.grid(True)
plt.show()


# # adding weights to target data <1

# In[ ]:


# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

# Step 3: Define the polynomial degree and pipeline
degree = 2  # Degree of the polynomial, you can adjust this
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),  # Generate polynomial features
    ('scaler', StandardScaler()),  # Standardize the features
    ('linear_regression', LinearRegression())  # Apply Linear Regression on transformed features
])

# Step 4: Create sample weights (this is an example, adjust based on your needs)
# For instance, you can give more weight to the samples where wind power is above 1.0
sample_weights = np.where(y_train > 0, 10, 1)  # Assign a weight of 2 to samples with target > 1, and 1 to others

# Step 5: Fit the pipeline with sample weights
pipeline.fit(X_train, y_train, linear_regression__sample_weight=sample_weights)

# Step 6: Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Step 7: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Polynomial Regression with Weights Performance:")
print(f"Polynomial Degree: {degree}")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)


# In[ ]:


plt.figure(figsize=(8, 6))
sns.histplot(y, bins=30, kde=True, color='blue')  # kde=True adds a density curve
plt.title('Distribution of the Target Variable')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[ ]:


import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



# Step 1: Define the target variable (y)
y = df['Snorrebakken sterled Active Power | sno_ost_effekt | 804131']

# Step 2: Create bins based on the target value ranges
bins = [0, 0.1, 0.5, 1, 1.5, 2.72]  # Define bins based on the observed distribution
weights = np.array([0.5, 1.0, 1.5, 2.0, 3.0])  # Define corresponding weights for each bin

# Step 3: Assign weights to each data point based on which bin the target value falls into
bin_indices = np.digitize(y, bins, right=True)  # Assign each target value to a bin (returns bin index)
sample_weights = weights[bin_indices - 1]  # Map the bin index to the corresponding weight


# Step 5: Split the data into training and test sets (both X and y)
X_train, X_test, y_train, y_test, sample_weights_train, sample_weights_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42,shuffle=False)

# Step 6: Define the polynomial degree and pipeline
degree = 2  # Degree of the polynomial, you can adjust this
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),  # Generate polynomial features
    ('scaler', StandardScaler()),  # Standardize the features
    ('linear_regression', LinearRegression())  # Apply Linear Regression on transformed features
])

# Step 7: Fit the pipeline with sample weights (use the correct target `y_train` and pass weights separately)
pipeline.fit(X_train, y_train, linear_regression__sample_weight=sample_weights_train)

# Step 8: Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Step 9: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Polynomial Regression with Weights Performance:")
print(f"Polynomial Degree: {degree}")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)


# In[ ]:


y.shape


# In[ ]:





# In[56]:


lasso_model = pipeline.named_steps['lasso']  # Access the trained Lasso model
print("Lasso coefficients:", lasso_model.coef_)
print("Number of non-zero coefficients in Lasso:", np.sum(lasso_model.coef_ != 0))


# In[50]:


ridge_normal_eq_model = pipeline.named_steps['ridge_normal_eq_lr']
print("ridge coef:", ridge_normal_eq_model.beta_[1:])


# In[ ]:





# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
power_data=pd.read_csv("Energydata export 18-09-2024 20-05-47.csv",parse_dates=['ts'])


# In[ ]:


power_data.isna().sum()


# In[7]:


power_data.shape


# In[ ]:





# In[19]:


power_data=pd.read_csv("Energydata export 18-09-2024 20-05-47.csv",parse_dates=['ts'])


# In[ ]:


power_data[power_data['ts'] >= '2023-12-01 00:00:00'].head(100)


# In[ ]:


# Plotting 'ts' (index) against the target y (Wind Power)
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Snorrebakken sterled Active Power | sno_ost_effekt | 804131'], label='Wind Power')

# Labeling the axes and title
plt.xlabel('Timestamp (ts)')
plt.ylabel('Snorrebakken sterled Active Power (Wind Power)')
plt.title('Wind Power over Time')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()


# In[ ]:




