import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("irrigation_data_1.csv")  # Use your uploaded dataset

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Drop duplicates and handle missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Remove "crop" column
if "crop" in df.columns:
    df.drop(columns=["crop"], inplace=True)

# Define Features and Target
X = df.drop(columns=["time"])  # Features (without target variable)
y = df["time"]  # Target (Pump ON time in minutes)

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Regressor Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred = model.predict(X_test_scaled)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Evaluation:\nMAE: {mae:.2f} | MSE: {mse:.2f} | R² Score: {r2:.2f}")

# Save Model and Scaler
pickle.dump(model, open("pump_time_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model training complete! Files saved:")
print("- pump_time_model.pkl")
print("- scaler.pkl")