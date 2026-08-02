# ============================================
# AI Internship - Day 10
# Topic: Model Evaluation using MAE, MSE & R² Score
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# --------------------------------------------
# Step 1: Load Dataset
# --------------------------------------------
data = pd.read_csv("student_scores.csv")

print("First 5 Rows:\n")
print(data.head())

# --------------------------------------------
# Step 2: Split Features & Target
# --------------------------------------------
X = data[['Hours']]
y = data['Scores']

# --------------------------------------------
# Step 3: Train-Test Split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------
# Step 4: Train Model
# --------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------
# Step 5: Prediction
# --------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------
# Step 6: Evaluate Model
# --------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# --------------------------------------------
# Step 7: Display Predictions
# --------------------------------------------
result = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": y_pred
})

print("\nPrediction Comparison:")
print(result)

print("\nModel Evaluation Completed Successfully!")