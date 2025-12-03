"""
Train a simple ML model for NYC Taxi trip duration prediction.
This is a lightweight alternative to Apache Submarine for resource-constrained environments.
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("Loading data...")
# Read a subset of the training data
df = pd.read_csv('data/taxi_subset.csv')

print(f"Loaded {len(df)} records")
print(f"Columns: {list(df.columns)}")

# Feature engineering
print("\nPreparing features...")

# Convert datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# Extract time features
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['month'] = df['pickup_datetime'].dt.month

# Calculate distance (haversine approximation using lat/lon)
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate approximate distance in km"""
    R = 6371  # Earth's radius in kilometers
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine_distance(
    df['pickup_latitude'], df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

# Select features for the model
features = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 
            'dropoff_longitude', 'dropoff_latitude', 'hour', 'day_of_week', 
            'month', 'distance_km']

X = df[features]
y = df['trip_duration']

# Handle any NaN values
X = X.fillna(X.median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train model
print("\nTraining RandomForest model...")
model = RandomForestRegressor(
    n_estimators=50,  # Keep small for fast training
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f} seconds")
print(f"R² Score: {r2:.4f}")

# Feature importance
print("\nFeature Importance:")
for feature, importance in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {importance:.4f}")

# Save model
model_path = 'model/taxi_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'features': features,
        'rmse': rmse,
        'r2': r2
    }, f)

print(f"\n✓ Model saved to {model_path}")
print("Ready for Phase 2 & 3!")
