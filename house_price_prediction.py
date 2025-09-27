# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the California housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  # Convert to DataFrame
print("✅ Dataset loaded successfully!")

# Step 2: Explore the data
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Step 3: Clean the data
# (This dataset is already clean, but we check for nulls and handle them if present)
df = df.dropna()

# Step 4: Select features and target
X = df.drop("MedHouseVal", axis=1)  # Features
y = df["MedHouseVal"]               # Target (median house value)

# Step 5: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set:", X_train.shape)
print("Test set:", X_test.shape)

# Step 6: Feature scaling (important for regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train the model (Linear Regression)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("\n✅ Model training complete!")

# Step 8: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 9: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# Step 10: Visualize predictions
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
