"""
Train ML models for NYC Taxi trip duration prediction using Apache Submarine.
Apache Submarine provides a unified ML platform for training, serving, and managing models at scale.
This implementation uses Submarine's experiment tracking and distributed training capabilities.
"""

import pandas as pd
import pickle
import numpy as np
import os
import json
from datetime import datetime

# Submarine imports for experiment tracking and model management
try:
    from submarine import tracking
    SUBMARINE_AVAILABLE = True
except ImportError:
    SUBMARINE_AVAILABLE = False
    print("Warning: Apache Submarine SDK not available. Running in standalone mode.")

# ML framework imports - using PyTorch for Submarine compatibility
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuration for Submarine
SUBMARINE_SERVER = os.getenv('SUBMARINE_SERVER', 'http://localhost:8080')
EXPERIMENT_NAME = "nyc-taxi-duration-prediction"
MODEL_NAME = "taxi-duration-model"


class TaxiDurationModel(nn.Module):
    """
    Deep Neural Network for taxi trip duration prediction.
    Architecture designed for distributed training with Submarine.
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32]):
        super(TaxiDurationModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class GradientBoostingWrapper:
    """
    Wrapper for Gradient Boosting to provide consistent interface.
    Used as baseline comparison with deep learning model.
    """
    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate approximate distance in km using Haversine formula."""
    R = 6371  # Earth's radius in kilometers
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def prepare_features(df):
    """Prepare features for the model."""
    # Convert datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # Extract time features
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    
    # Calculate distance
    df['distance_km'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    return df


def log_params_to_submarine(params: dict, experiment_id: str = None):
    """Log parameters to Submarine experiment tracking."""
    if SUBMARINE_AVAILABLE:
        try:
            for key, value in params.items():
                tracking.log_param(key, value)
            print(f"  ✓ Parameters logged to Submarine")
        except Exception as e:
            print(f"  Warning: Could not log params to Submarine: {e}")


def log_metrics_to_submarine(metrics: dict, step: int = None):
    """Log metrics to Submarine experiment tracking."""
    if SUBMARINE_AVAILABLE:
        try:
            for key, value in metrics.items():
                tracking.log_metric(key, value, step=step)
            print(f"  ✓ Metrics logged to Submarine")
        except Exception as e:
            print(f"  Warning: Could not log metrics to Submarine: {e}")


def register_model_to_submarine(model_path: str, model_name: str, metrics: dict):
    """Register trained model to Submarine Model Registry."""
    if SUBMARINE_AVAILABLE:
        try:
            # Log model artifact to tracking
            tracking.log_artifact(model_path)
            print(f"  ✓ Model registered to Submarine Model Registry")
        except Exception as e:
            print(f"  Warning: Could not register model to Submarine: {e}")


def train_pytorch_model(X_train, y_train, X_test, y_test, device, epochs=50, batch_size=1024):
    """
    Train PyTorch model with Submarine experiment tracking.
    Designed for distributed training on Submarine/Kubernetes.
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).to(device)
    
    # Create DataLoader for batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = TaxiDurationModel(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training parameters for Submarine logging
    params = {
        'model_type': 'PyTorch_DNN',
        'hidden_dims': '[256, 128, 64, 32]',
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'weight_decay': 1e-5,
        'dropout': 0.2
    }
    log_params_to_submarine(params)
    
    best_rmse = float('inf')
    best_model_state = None
    
    print(f"\n  Training PyTorch DNN for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test_tensor)
            val_mse = criterion(val_predictions, y_test_tensor).item()
            val_rmse = np.sqrt(val_mse)
        
        scheduler.step(val_rmse)
        
        # Log metrics to Submarine every 10 epochs
        if (epoch + 1) % 10 == 0:
            metrics = {
                'train_loss': total_loss / len(train_loader),
                'val_rmse': val_rmse
            }
            log_metrics_to_submarine(metrics, step=epoch)
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Val RMSE: {val_rmse:.2f}")
        
        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model_state = model.state_dict().copy()
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    return model, best_rmse


def train_sklearn_baseline(X_train, y_train, X_test, y_test):
    """Train sklearn GradientBoosting as baseline comparison."""
    params = {
        'model_type': 'GradientBoosting',
        'n_estimators': 100,
        'max_depth': 8,
        'learning_rate': 0.1
    }
    log_params_to_submarine(params)
    
    model = GradientBoostingWrapper()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, rmse


def main():
    """Main training pipeline with Submarine integration."""
    
    print("="*70)
    print("  APACHE SUBMARINE ML TRAINING PIPELINE")
    print("  NYC Taxi Trip Duration Prediction")
    print("="*70)
    
    # Initialize Submarine experiment tracking
    if SUBMARINE_AVAILABLE:
        try:
            tracking.set_tracking_uri(SUBMARINE_SERVER)
            tracking.set_experiment(EXPERIMENT_NAME)
            print(f"\n✓ Connected to Submarine at {SUBMARINE_SERVER}")
        except Exception as e:
            print(f"\n⚠ Could not connect to Submarine: {e}")
            print("  Continuing in standalone mode...")
    else:
        print("\n⚠ Running in standalone mode (Submarine SDK not installed)")
        print("  Install with: pip install apache-submarine")
    
    # Check for GPU (for distributed training)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Training device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n" + "-"*70)
    print("  PHASE 1: Data Loading and Preprocessing")
    print("-"*70)
    
    df = pd.read_csv('data/train.csv')
    print(f"  Loaded {len(df):,} records")
    
    # Remove outliers
    original_len = len(df)
    df = df[(df['trip_duration'] >= 60) & (df['trip_duration'] <= 7200)]
    print(f"  After removing outliers: {len(df):,} records ({original_len - len(df):,} removed)")
    
    # Feature engineering
    print("\n  Preparing features...")
    df = prepare_features(df)
    
    # Select features
    features = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 
                'dropoff_longitude', 'dropoff_latitude', 'hour', 'day_of_week', 
                'month', 'distance_km']
    
    X = df[features].fillna(df[features].median())
    y = df['trip_duration']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training
    print("\n" + "-"*70)
    print("  PHASE 2: Model Training with Submarine")
    print("-"*70)
    
    results = []
    
    # Train PyTorch model (Submarine-compatible for distributed training)
    print("\n  [1/2] Training PyTorch Deep Neural Network...")
    pytorch_model, pytorch_rmse = train_pytorch_model(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        device, epochs=50, batch_size=1024
    )
    
    # Calculate full metrics for PyTorch model
    pytorch_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_pred_pytorch = pytorch_model(X_test_tensor).cpu().numpy()
    
    pytorch_mae = mean_absolute_error(y_test, y_pred_pytorch)
    pytorch_r2 = r2_score(y_test, y_pred_pytorch)
    
    results.append({
        'name': 'PyTorch_DNN',
        'model': pytorch_model,
        'rmse': pytorch_rmse,
        'mae': pytorch_mae,
        'r2': pytorch_r2,
        'type': 'pytorch'
    })
    
    print(f"\n    PyTorch DNN Results:")
    print(f"      RMSE: {pytorch_rmse:.2f} seconds")
    print(f"      MAE:  {pytorch_mae:.2f} seconds")
    print(f"      R²:   {pytorch_r2:.4f}")
    
    # Train sklearn baseline for comparison
    print("\n  [2/2] Training GradientBoosting baseline...")
    sklearn_model, sklearn_rmse = train_sklearn_baseline(
        X_train, y_train, X_test, y_test
    )
    
    y_pred_sklearn = sklearn_model.predict(X_test)
    sklearn_mae = mean_absolute_error(y_test, y_pred_sklearn)
    sklearn_r2 = r2_score(y_test, y_pred_sklearn)
    
    results.append({
        'name': 'GradientBoosting',
        'model': sklearn_model,
        'rmse': sklearn_rmse,
        'mae': sklearn_mae,
        'r2': sklearn_r2,
        'type': 'sklearn'
    })
    
    print(f"\n    GradientBoosting Results:")
    print(f"      RMSE: {sklearn_rmse:.2f} seconds")
    print(f"      MAE:  {sklearn_mae:.2f} seconds")
    print(f"      R²:   {sklearn_r2:.4f}")
    
    # Results comparison
    print("\n" + "-"*70)
    print("  PHASE 3: Model Comparison")
    print("-"*70)
    
    results.sort(key=lambda x: x['rmse'])
    
    print(f"\n  {'Model':<25} {'RMSE':>12} {'MAE':>12} {'R²':>10}")
    print("  " + "-"*60)
    for r in results:
        print(f"  {r['name']:<25} {r['rmse']:>12.2f} {r['mae']:>12.2f} {r['r2']:>10.4f}")
    
    best = results[0]
    
    print("\n" + "="*70)
    print(f"  🏆 BEST MODEL: {best['name']}")
    print(f"     RMSE: {best['rmse']:.2f} seconds (~{best['rmse']/60:.1f} minutes)")
    print(f"     MAE:  {best['mae']:.2f} seconds")
    print(f"     R²:   {best['r2']:.4f}")
    print("="*70)
    
    # Log final metrics to Submarine
    final_metrics = {
        'best_model': best['name'],
        'final_rmse': best['rmse'],
        'final_mae': best['mae'],
        'final_r2': best['r2']
    }
    log_metrics_to_submarine(final_metrics)
    
    # Save model
    print("\n  Saving model...")
    model_path = 'model/taxi_model.pkl'
    
    # Create model artifact for Submarine
    model_artifact = {
        'model_name': best['name'],
        'model_type': best['type'],
        'features': features,
        'rmse': best['rmse'],
        'mae': best['mae'],
        'r2': best['r2'],
        'scaler': scaler,
        'training_date': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'submarine_experiment': EXPERIMENT_NAME,
        'all_results': [{k: v for k, v in r.items() if k != 'model'} for r in results]
    }
    
    if best['type'] == 'pytorch':
        # Save PyTorch model state dict
        model_artifact['model_state_dict'] = best['model'].state_dict()
        model_artifact['model_architecture'] = {
            'input_dim': len(features),
            'hidden_dims': [256, 128, 64, 32]
        }
    else:
        model_artifact['model'] = best['model'].model
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_artifact, f)
    
    # Register model to Submarine Model Registry
    register_model_to_submarine(model_path, MODEL_NAME, final_metrics)
    
    print(f"\n  ✓ Model saved to {model_path}")
    
    # Save Submarine experiment metadata
    experiment_metadata = {
        'experiment_name': EXPERIMENT_NAME,
        'model_name': MODEL_NAME,
        'training_date': datetime.now().isoformat(),
        'best_model': best['name'],
        'metrics': final_metrics,
        'features': features,
        'submarine_server': SUBMARINE_SERVER
    }
    
    with open('model/experiment_metadata.json', 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    
    print(f"  ✓ Experiment metadata saved to model/experiment_metadata.json")
    
    print("\n" + "="*70)
    print("  Training complete! Model ready for deployment.")
    print("  Use Submarine to deploy as a REST API for serving predictions.")
    print("="*70)


if __name__ == "__main__":
    main()
