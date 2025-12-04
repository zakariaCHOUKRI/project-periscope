"""
Train multiple ML models for NYC Taxi trip duration prediction.
Compare models and save the best one based on RMSE.
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

print("Loading data...")
# Read the full training data instead of the small subset
df = pd.read_csv('data/train.csv')

print(f"Loaded {len(df)} records")

# Remove outliers - trips that are too short (<1 min) or too long (>2 hours)
# These are likely data errors or unusual circumstances
original_len = len(df)
df = df[(df['trip_duration'] >= 60) & (df['trip_duration'] <= 7200)]
print(f"After removing outliers: {len(df)} records ({original_len - len(df)} removed)")

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

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to train (increased complexity for larger dataset)
models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    ),
    'DecisionTree': DecisionTreeRegressor(
        max_depth=15,
        min_samples_split=5,
        random_state=42
    ),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=1.0, random_state=42),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    'KNeighbors': KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
    'AdaBoost': AdaBoostRegressor(
        n_estimators=100,
        random_state=42
    ),
}

# Models that need scaled features
scaled_models = {'Ridge', 'Lasso', 'ElasticNet', 'KNeighbors'}

# Train and evaluate all models
print("\n" + "="*60)
print("TRAINING AND EVALUATING MULTIPLE MODELS")
print("="*60)

results = []

for name, model in models.items():
    print(f"\n{'─'*40}")
    print(f"Training {name}...")
    
    # Use scaled features for linear models and KNN
    if name in scaled_models:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    # Train model
    model.fit(X_tr, y_train)
    
    # Predict
    y_pred = model.predict(X_te)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results.append({
        'name': name,
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'needs_scaling': name in scaled_models
    })
    
    print(f"  RMSE: {rmse:.2f} seconds")
    print(f"  MAE:  {mae:.2f} seconds")
    print(f"  R²:   {r2:.4f}")

# Sort by RMSE (lower is better)
results.sort(key=lambda x: x['rmse'])

# Print comparison table
print("\n" + "="*60)
print("MODEL COMPARISON (sorted by RMSE)")
print("="*60)
print(f"{'Model':<20} {'RMSE':>12} {'MAE':>12} {'R²':>10}")
print("─"*60)
for r in results:
    print(f"{r['name']:<20} {r['rmse']:>12.2f} {r['mae']:>12.2f} {r['r2']:>10.4f}")

# Select best model
best = results[0]
print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best['name']}")
print(f"   RMSE: {best['rmse']:.2f} seconds")
print(f"   MAE:  {best['mae']:.2f} seconds")
print(f"   R²:   {best['r2']:.4f}")
print("="*60)

# Feature importance (if available)
if hasattr(best['model'], 'feature_importances_'):
    print("\nFeature Importance:")
    for feature, importance in sorted(zip(features, best['model'].feature_importances_), 
                                       key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")

# Save best model
model_path = 'model/taxi_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': best['model'],
        'model_name': best['name'],
        'features': features,
        'rmse': best['rmse'],
        'mae': best['mae'],
        'r2': best['r2'],
        'needs_scaling': best['needs_scaling'],
        'scaler': scaler if best['needs_scaling'] else None,
        'all_results': [{k: v for k, v in r.items() if k != 'model'} for r in results]
    }, f)

print(f"\n✓ Best model ({best['name']}) saved to {model_path}")
print("Ready for Phase 2 & 3!")
