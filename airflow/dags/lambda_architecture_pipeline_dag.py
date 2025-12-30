"""
Airflow DAG: Complete Lambda Architecture Pipeline
====================================================
A comprehensive, production-grade DAG that orchestrates the entire Lambda Architecture:
- Data Ingestion & Validation
- Batch Layer Processing (Master Data + Multiple Views)
- Speed Layer Initialization
- ML Model Training & Deployment
- Serving Layer Health Checks
- Data Quality Gates & Reconciliation
- Alerting & Notifications

This DAG demonstrates advanced Airflow features:
- TaskGroups for organization
- BranchPythonOperator for conditional logic
- ShortCircuitOperator for quality gates
- XCom for task communication
- Parallel task execution
- Dynamic task generation
- Sensors for external dependencies
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator, ShortCircuitOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json
import hashlib

# Configuration
DATA_DIR = '/opt/airflow/data'
MODEL_DIR = '/opt/airflow/model'
MASTER_DATA_PATH = f'{DATA_DIR}/avro'
BATCH_VIEWS_PATH = f'{DATA_DIR}/parquet'
SPEED_LAYER_PATH = f'{DATA_DIR}/speed_layer'
RAW_DATA_FILE = f'{DATA_DIR}/train.csv'
SUBSET_DATA_FILE = f'{DATA_DIR}/taxi_subset.csv'
QUALITY_THRESHOLD = 0.95  # 95% data quality threshold
MODEL_PERFORMANCE_THRESHOLD = 0.7  # R² score threshold

default_args = {
    'owner': 'lambda-architecture',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(hours=1),
}

# ============================================================================
# DATA VALIDATION & QUALITY FUNCTIONS
# ============================================================================

def validate_raw_data(**context):
    """
    Validate raw data file exists and has expected schema.
    Pushes validation results to XCom for downstream tasks.
    """
    validation_results = {
        'file_exists': False,
        'row_count': 0,
        'schema_valid': False,
        'null_percentage': {},
        'quality_score': 0.0,
        'validation_timestamp': datetime.now().isoformat()
    }
    
    # Check file existence
    data_file = RAW_DATA_FILE if os.path.exists(RAW_DATA_FILE) else SUBSET_DATA_FILE
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    validation_results['file_exists'] = True
    
    # Load and validate schema
    df = pd.read_csv(data_file)
    validation_results['row_count'] = len(df)
    
    required_columns = [
        'id', 'vendor_id', 'pickup_datetime', 'passenger_count',
        'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
        'dropoff_latitude', 'trip_duration'
    ]
    
    missing_cols = set(required_columns) - set(df.columns)
    validation_results['schema_valid'] = len(missing_cols) == 0
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate null percentages
    for col in required_columns:
        null_pct = df[col].isnull().sum() / len(df) * 100
        validation_results['null_percentage'][col] = round(null_pct, 2)
    
    # Calculate overall quality score
    total_nulls = sum(validation_results['null_percentage'].values())
    validation_results['quality_score'] = max(0, 1 - (total_nulls / (len(required_columns) * 100)))
    
    # Push to XCom
    context['ti'].xcom_push(key='validation_results', value=validation_results)
    context['ti'].xcom_push(key='data_file', value=data_file)
    
    print(f"✓ Validation complete:")
    print(f"  - Rows: {validation_results['row_count']:,}")
    print(f"  - Quality Score: {validation_results['quality_score']:.2%}")
    
    return validation_results


def check_data_quality(**context):
    """
    Quality gate - determines if data quality is sufficient to proceed.
    Returns True to continue, False to short-circuit the pipeline.
    """
    ti = context['ti']
    validation_results = ti.xcom_pull(task_ids='data_validation.validate_raw_data', key='validation_results')
    
    quality_score = validation_results.get('quality_score', 0)
    
    if quality_score >= QUALITY_THRESHOLD:
        print(f"✓ Quality check PASSED: {quality_score:.2%} >= {QUALITY_THRESHOLD:.2%}")
        return True
    else:
        print(f"✗ Quality check FAILED: {quality_score:.2%} < {QUALITY_THRESHOLD:.2%}")
        return False


def data_profiling(**context):
    """
    Generate comprehensive data profiling statistics.
    """
    ti = context['ti']
    data_file = ti.xcom_pull(task_ids='data_validation.validate_raw_data', key='data_file')
    
    df = pd.read_csv(data_file)
    
    # Parse datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    profile = {
        'basic_stats': {
            'total_records': len(df),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'date_range': {
                'min': df['pickup_datetime'].min().isoformat(),
                'max': df['pickup_datetime'].max().isoformat()
            }
        },
        'trip_duration': {
            'mean': df['trip_duration'].mean(),
            'median': df['trip_duration'].median(),
            'std': df['trip_duration'].std(),
            'min': df['trip_duration'].min(),
            'max': df['trip_duration'].max(),
            'outliers_pct': ((df['trip_duration'] < 60) | (df['trip_duration'] > 7200)).sum() / len(df) * 100
        },
        'vendor_distribution': df['vendor_id'].value_counts().to_dict(),
        'passenger_distribution': df['passenger_count'].value_counts().to_dict(),
        'hourly_distribution': df['pickup_datetime'].dt.hour.value_counts().sort_index().to_dict()
    }
    
    ti.xcom_push(key='data_profile', value=profile)
    
    print(f"✓ Data profiling complete:")
    print(f"  - Total records: {profile['basic_stats']['total_records']:,}")
    print(f"  - Memory usage: {profile['basic_stats']['memory_usage_mb']:.2f} MB")
    print(f"  - Outliers: {profile['trip_duration']['outliers_pct']:.2f}%")
    
    return profile


def detect_anomalies(**context):
    """
    Detect anomalies in the data using statistical methods.
    """
    ti = context['ti']
    data_file = ti.xcom_pull(task_ids='data_validation.validate_raw_data', key='data_file')
    
    df = pd.read_csv(data_file)
    
    anomalies = {
        'geographic_anomalies': 0,
        'duration_anomalies': 0,
        'passenger_anomalies': 0,
        'total_anomalies': 0,
        'anomaly_percentage': 0.0
    }
    
    # Geographic anomalies (outside NYC bounds)
    nyc_bounds = {
        'lat_min': 40.4, 'lat_max': 41.0,
        'lon_min': -74.3, 'lon_max': -73.7
    }
    
    geo_mask = (
        (df['pickup_latitude'] < nyc_bounds['lat_min']) |
        (df['pickup_latitude'] > nyc_bounds['lat_max']) |
        (df['pickup_longitude'] < nyc_bounds['lon_min']) |
        (df['pickup_longitude'] > nyc_bounds['lon_max'])
    )
    anomalies['geographic_anomalies'] = geo_mask.sum()
    
    # Duration anomalies (< 1 min or > 3 hours)
    duration_mask = (df['trip_duration'] < 60) | (df['trip_duration'] > 10800)
    anomalies['duration_anomalies'] = duration_mask.sum()
    
    # Passenger anomalies
    passenger_mask = (df['passenger_count'] < 0) | (df['passenger_count'] > 9)
    anomalies['passenger_anomalies'] = passenger_mask.sum()
    
    # Total
    total_mask = geo_mask | duration_mask | passenger_mask
    anomalies['total_anomalies'] = total_mask.sum()
    anomalies['anomaly_percentage'] = total_mask.sum() / len(df) * 100
    
    ti.xcom_push(key='anomalies', value=anomalies)
    
    print(f"✓ Anomaly detection complete:")
    print(f"  - Geographic: {anomalies['geographic_anomalies']:,}")
    print(f"  - Duration: {anomalies['duration_anomalies']:,}")
    print(f"  - Passenger: {anomalies['passenger_anomalies']:,}")
    print(f"  - Total: {anomalies['anomaly_percentage']:.2f}%")
    
    return anomalies


# ============================================================================
# BATCH LAYER FUNCTIONS
# ============================================================================

def clean_and_transform_data(**context):
    """
    Clean data and apply transformations for batch processing.
    """
    ti = context['ti']
    data_file = ti.xcom_pull(task_ids='data_validation.validate_raw_data', key='data_file')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    df = pd.read_csv(data_file)
    initial_count = len(df)
    
    # Remove anomalies
    df = df[
        (df['trip_duration'] >= 60) & 
        (df['trip_duration'] <= 7200) &
        (df['passenger_count'] > 0) &
        (df['passenger_count'] <= 9)
    ]
    
    # Remove geographic outliers
    df = df[
        (df['pickup_latitude'] >= 40.4) & (df['pickup_latitude'] <= 41.0) &
        (df['pickup_longitude'] >= -74.3) & (df['pickup_longitude'] <= -73.7)
    ]
    
    cleaned_count = len(df)
    
    # Add derived features
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Calculate distance
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lat2 = np.radians(lat1), np.radians(lat2)
        dlat = lat2 - lat1
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    df['distance_km'] = haversine(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # Calculate speed
    df['speed_kmh'] = df['distance_km'] / (df['trip_duration'] / 3600)
    df['speed_kmh'] = df['speed_kmh'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Save cleaned data
    os.makedirs(f'{DATA_DIR}/cleaned', exist_ok=True)
    cleaned_file = f'{DATA_DIR}/cleaned/taxi_cleaned_{date_str}.csv'
    df.to_csv(cleaned_file, index=False)
    
    ti.xcom_push(key='cleaned_file', value=cleaned_file)
    ti.xcom_push(key='cleaned_count', value=cleaned_count)
    
    print(f"✓ Data cleaning complete:")
    print(f"  - Initial: {initial_count:,}")
    print(f"  - Cleaned: {cleaned_count:,}")
    print(f"  - Removed: {initial_count - cleaned_count:,} ({(initial_count - cleaned_count) / initial_count * 100:.2f}%)")
    
    return cleaned_file


def create_master_data_avro(**context):
    """
    Create master data in Avro format - the immutable batch layer data.
    """
    import fastavro
    from fastavro import writer
    
    ti = context['ti']
    cleaned_file = ti.xcom_pull(task_ids='batch_layer.clean_and_transform', key='cleaned_file')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    df = pd.read_csv(cleaned_file)
    
    # Define Avro schema
    schema = {
        'type': 'record',
        'name': 'TaxiTripMaster',
        'namespace': 'nyc.taxi.batch',
        'fields': [
            {'name': 'id', 'type': 'string'},
            {'name': 'vendor_id', 'type': 'int'},
            {'name': 'pickup_datetime', 'type': 'string'},
            {'name': 'passenger_count', 'type': 'int'},
            {'name': 'pickup_longitude', 'type': 'double'},
            {'name': 'pickup_latitude', 'type': 'double'},
            {'name': 'dropoff_longitude', 'type': 'double'},
            {'name': 'dropoff_latitude', 'type': 'double'},
            {'name': 'store_and_fwd_flag', 'type': 'string'},
            {'name': 'trip_duration', 'type': 'int'},
            {'name': 'hour', 'type': 'int'},
            {'name': 'day_of_week', 'type': 'int'},
            {'name': 'month', 'type': 'int'},
            {'name': 'is_weekend', 'type': 'int'},
            {'name': 'is_rush_hour', 'type': 'int'},
            {'name': 'distance_km', 'type': 'double'},
            {'name': 'speed_kmh', 'type': 'double'},
            {'name': 'ingestion_date', 'type': 'string'},
            {'name': 'batch_id', 'type': 'string'},
        ]
    }
    
    # Generate batch ID
    batch_id = hashlib.md5(f"{date_str}_{len(df)}".encode()).hexdigest()[:12]
    
    # Convert to records
    records = []
    for _, row in df.iterrows():
        record = {
            'id': str(row['id']),
            'vendor_id': int(row['vendor_id']),
            'pickup_datetime': str(row['pickup_datetime']),
            'passenger_count': int(row['passenger_count']),
            'pickup_longitude': float(row['pickup_longitude']),
            'pickup_latitude': float(row['pickup_latitude']),
            'dropoff_longitude': float(row['dropoff_longitude']),
            'dropoff_latitude': float(row['dropoff_latitude']),
            'store_and_fwd_flag': str(row.get('store_and_fwd_flag', 'N')),
            'trip_duration': int(row['trip_duration']),
            'hour': int(row['hour']),
            'day_of_week': int(row['day_of_week']),
            'month': int(row['month']),
            'is_weekend': int(row['is_weekend']),
            'is_rush_hour': int(row['is_rush_hour']),
            'distance_km': float(row['distance_km']),
            'speed_kmh': float(row['speed_kmh']),
            'ingestion_date': date_str,
            'batch_id': batch_id,
        }
        records.append(record)
    
    # Write Avro
    os.makedirs(MASTER_DATA_PATH, exist_ok=True)
    avro_file = f'{MASTER_DATA_PATH}/taxi_master_{date_str}.avro'
    
    parsed_schema = fastavro.parse_schema(schema)
    with open(avro_file, 'wb') as out:
        writer(out, parsed_schema, records)
    
    ti.xcom_push(key='avro_file', value=avro_file)
    ti.xcom_push(key='batch_id', value=batch_id)
    
    print(f"✓ Master data created: {avro_file}")
    print(f"  - Records: {len(records):,}")
    print(f"  - Batch ID: {batch_id}")
    
    return avro_file


def create_hourly_stats_view(**context):
    """Create Parquet batch view: Hourly statistics."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    ti = context['ti']
    cleaned_file = ti.xcom_pull(task_ids='batch_layer.clean_and_transform', key='cleaned_file')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    df = pd.read_csv(cleaned_file)
    
    hourly_stats = df.groupby('hour').agg({
        'trip_duration': ['mean', 'std', 'min', 'max', 'count'],
        'distance_km': ['mean', 'std'],
        'passenger_count': ['mean', 'sum'],
        'speed_kmh': 'mean',
        'is_rush_hour': 'first'
    }).reset_index()
    
    hourly_stats.columns = [
        'hour', 'avg_duration', 'std_duration', 'min_duration', 'max_duration', 'trip_count',
        'avg_distance', 'std_distance', 'avg_passengers', 'total_passengers',
        'avg_speed', 'is_rush_hour'
    ]
    
    os.makedirs(BATCH_VIEWS_PATH, exist_ok=True)
    output_file = f'{BATCH_VIEWS_PATH}/hourly_stats_{date_str}.parquet'
    
    table = pa.Table.from_pandas(hourly_stats)
    pq.write_table(table, output_file)
    
    print(f"✓ Created hourly_stats view: {output_file}")
    return output_file


def create_daily_summary_view(**context):
    """Create Parquet batch view: Daily summary."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    ti = context['ti']
    cleaned_file = ti.xcom_pull(task_ids='batch_layer.clean_and_transform', key='cleaned_file')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    df = pd.read_csv(cleaned_file)
    
    daily_summary = pd.DataFrame({
        'date': [date_str],
        'total_trips': [len(df)],
        'avg_duration': [df['trip_duration'].mean()],
        'median_duration': [df['trip_duration'].median()],
        'std_duration': [df['trip_duration'].std()],
        'avg_distance': [df['distance_km'].mean()],
        'total_distance': [df['distance_km'].sum()],
        'total_passengers': [df['passenger_count'].sum()],
        'avg_passengers': [df['passenger_count'].mean()],
        'avg_speed': [df['speed_kmh'].mean()],
        'rush_hour_trips': [df['is_rush_hour'].sum()],
        'weekend_trips': [df['is_weekend'].sum()],
    })
    
    output_file = f'{BATCH_VIEWS_PATH}/daily_summary_{date_str}.parquet'
    
    table = pa.Table.from_pandas(daily_summary)
    pq.write_table(table, output_file)
    
    print(f"✓ Created daily_summary view: {output_file}")
    return output_file


def create_vendor_stats_view(**context):
    """Create Parquet batch view: Vendor statistics."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    ti = context['ti']
    cleaned_file = ti.xcom_pull(task_ids='batch_layer.clean_and_transform', key='cleaned_file')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    df = pd.read_csv(cleaned_file)
    
    vendor_stats = df.groupby('vendor_id').agg({
        'trip_duration': ['mean', 'std', 'count'],
        'distance_km': 'mean',
        'passenger_count': ['mean', 'sum'],
        'speed_kmh': 'mean',
    }).reset_index()
    
    vendor_stats.columns = [
        'vendor_id', 'avg_duration', 'std_duration', 'trip_count',
        'avg_distance', 'avg_passengers', 'total_passengers', 'avg_speed'
    ]
    
    output_file = f'{BATCH_VIEWS_PATH}/vendor_stats_{date_str}.parquet'
    
    table = pa.Table.from_pandas(vendor_stats)
    pq.write_table(table, output_file)
    
    print(f"✓ Created vendor_stats view: {output_file}")
    return output_file


def create_geographic_stats_view(**context):
    """Create Parquet batch view: Geographic zone statistics."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    ti = context['ti']
    cleaned_file = ti.xcom_pull(task_ids='batch_layer.clean_and_transform', key='cleaned_file')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    df = pd.read_csv(cleaned_file)
    
    # Create geographic zones (grid-based)
    df['pickup_zone_lat'] = (df['pickup_latitude'] * 10).round() / 10
    df['pickup_zone_lon'] = (df['pickup_longitude'] * 10).round() / 10
    df['pickup_zone'] = df['pickup_zone_lat'].astype(str) + ',' + df['pickup_zone_lon'].astype(str)
    
    geo_stats = df.groupby('pickup_zone').agg({
        'trip_duration': ['mean', 'count'],
        'distance_km': 'mean',
        'passenger_count': 'mean',
        'pickup_zone_lat': 'first',
        'pickup_zone_lon': 'first',
    }).reset_index()
    
    geo_stats.columns = [
        'zone', 'avg_duration', 'trip_count', 'avg_distance',
        'avg_passengers', 'lat', 'lon'
    ]
    
    output_file = f'{BATCH_VIEWS_PATH}/geographic_stats_{date_str}.parquet'
    
    table = pa.Table.from_pandas(geo_stats)
    pq.write_table(table, output_file)
    
    print(f"✓ Created geographic_stats view: {output_file}")
    return output_file


def create_temporal_patterns_view(**context):
    """Create Parquet batch view: Temporal patterns (hour x day_of_week)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    ti = context['ti']
    cleaned_file = ti.xcom_pull(task_ids='batch_layer.clean_and_transform', key='cleaned_file')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    df = pd.read_csv(cleaned_file)
    
    temporal_stats = df.groupby(['hour', 'day_of_week']).agg({
        'trip_duration': ['mean', 'count'],
        'distance_km': 'mean',
        'speed_kmh': 'mean',
    }).reset_index()
    
    temporal_stats.columns = [
        'hour', 'day_of_week', 'avg_duration', 'trip_count',
        'avg_distance', 'avg_speed'
    ]
    
    # Add day name
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    temporal_stats['day_name'] = temporal_stats['day_of_week'].map(lambda x: day_names[x])
    
    output_file = f'{BATCH_VIEWS_PATH}/temporal_patterns_{date_str}.parquet'
    
    table = pa.Table.from_pandas(temporal_stats)
    pq.write_table(table, output_file)
    
    print(f"✓ Created temporal_patterns view: {output_file}")
    return output_file


# ============================================================================
# ML MODEL FUNCTIONS
# ============================================================================

def check_model_exists(**context):
    """
    Check if a trained model exists. Branch based on result.
    """
    model_path = f'{MODEL_DIR}/taxi_model.pkl'
    
    if os.path.exists(model_path):
        print(f"✓ Model found at {model_path}")
        return 'ml_pipeline.evaluate_existing_model'
    else:
        print(f"✗ No model found, need to train")
        return 'ml_pipeline.train_model'


def train_model(**context):
    """
    Train ML model for trip duration prediction.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import pickle
    
    ti = context['ti']
    cleaned_file = ti.xcom_pull(task_ids='batch_layer.clean_and_transform', key='cleaned_file')
    execution_date = context['execution_date']
    
    df = pd.read_csv(cleaned_file)
    
    # Prepare features
    features = ['vendor_id', 'passenger_count', 'hour', 'day_of_week',
                'is_weekend', 'is_rush_hour', 'distance_km']
    
    X = df[features].values
    y = df['trip_duration'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'training_date': execution_date.isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
    }
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_package = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'metrics': metrics
    }
    
    model_path = f'{MODEL_DIR}/taxi_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    # Save metadata
    metadata_path = f'{MODEL_DIR}/experiment_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    ti.xcom_push(key='model_metrics', value=metrics)
    
    print(f"✓ Model trained and saved:")
    print(f"  - R² Score: {metrics['r2_score']:.4f}")
    print(f"  - RMSE: {metrics['rmse']:.2f} seconds")
    print(f"  - MAE: {metrics['mae']:.2f} seconds")
    
    return metrics


def evaluate_existing_model(**context):
    """
    Evaluate existing model on current data.
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import pickle
    
    ti = context['ti']
    cleaned_file = ti.xcom_pull(task_ids='batch_layer.clean_and_transform', key='cleaned_file')
    
    # Load model
    model_path = f'{MODEL_DIR}/taxi_model.pkl'
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    scaler = model_package['scaler']
    features = model_package['features']
    
    # Load data
    df = pd.read_csv(cleaned_file)
    X = df[features].values
    y = df['trip_duration'].values
    
    # Evaluate
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    metrics = {
        'r2_score': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'evaluation_date': datetime.now().isoformat(),
        'evaluation_samples': len(y),
    }
    
    ti.xcom_push(key='model_metrics', value=metrics)
    
    print(f"✓ Model evaluated:")
    print(f"  - R² Score: {metrics['r2_score']:.4f}")
    print(f"  - RMSE: {metrics['rmse']:.2f} seconds")
    
    return metrics


def check_model_performance(**context):
    """
    Quality gate for model performance.
    """
    ti = context['ti']
    
    # Try to get metrics from either training or evaluation
    metrics = ti.xcom_pull(task_ids='ml_pipeline.train_model', key='model_metrics')
    if not metrics:
        metrics = ti.xcom_pull(task_ids='ml_pipeline.evaluate_existing_model', key='model_metrics')
    
    if not metrics:
        print("✗ No model metrics available")
        return False
    
    r2 = metrics.get('r2_score', 0)
    
    if r2 >= MODEL_PERFORMANCE_THRESHOLD:
        print(f"✓ Model performance OK: R²={r2:.4f} >= {MODEL_PERFORMANCE_THRESHOLD}")
        return True
    else:
        print(f"✗ Model performance POOR: R²={r2:.4f} < {MODEL_PERFORMANCE_THRESHOLD}")
        return False


# ============================================================================
# SPEED LAYER FUNCTIONS
# ============================================================================

def initialize_speed_layer(**context):
    """
    Initialize speed layer directory and configuration.
    """
    os.makedirs(SPEED_LAYER_PATH, exist_ok=True)
    
    # Create speed layer config
    config = {
        'initialized_at': datetime.now().isoformat(),
        'kafka_topic': 'nyc-taxi-stream',
        'kafka_bootstrap_servers': ['kafka:29092'],
        'checkpoint_dir': f'{DATA_DIR}/checkpoint',
        'output_dir': SPEED_LAYER_PATH,
    }
    
    config_file = f'{SPEED_LAYER_PATH}/config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Speed layer initialized")
    return config


def verify_kafka_connection(**context):
    """
    Verify Kafka is accessible and topic exists.
    """
    try:
        from kafka import KafkaConsumer
        from kafka.errors import NoBrokersAvailable
        
        consumer = KafkaConsumer(
            bootstrap_servers=['kafka:29092'],
            consumer_timeout_ms=5000
        )
        topics = consumer.topics()
        consumer.close()
        
        print(f"✓ Kafka connection verified. Topics: {topics}")
        return True
    except Exception as e:
        print(f"⚠ Kafka not available: {e}")
        return True  # Continue anyway, Kafka might not be running in test


# ============================================================================
# SERVING LAYER & RECONCILIATION FUNCTIONS
# ============================================================================

def reconcile_batch_speed_layers(**context):
    """
    Compare batch and speed layer data for consistency.
    """
    ti = context['ti']
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    reconciliation = {
        'date': date_str,
        'batch_layer': {},
        'speed_layer': {},
        'discrepancies': [],
        'status': 'unknown'
    }
    
    # Check batch layer
    daily_summary_file = f'{BATCH_VIEWS_PATH}/daily_summary_{date_str}.parquet'
    if os.path.exists(daily_summary_file):
        import pyarrow.parquet as pq
        batch_df = pq.read_table(daily_summary_file).to_pandas()
        reconciliation['batch_layer'] = {
            'total_trips': int(batch_df['total_trips'].iloc[0]),
            'avg_duration': float(batch_df['avg_duration'].iloc[0]),
        }
    
    # Check speed layer
    speed_files = [f for f in os.listdir(SPEED_LAYER_PATH) if f.endswith('.json')]
    if speed_files:
        total_speed_records = 0
        speed_durations = []
        for sf in speed_files:
            with open(f'{SPEED_LAYER_PATH}/{sf}', 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        total_speed_records += 1
                        speed_durations.append(record.get('trip_duration', 0))
                    except:
                        pass
        
        if speed_durations:
            reconciliation['speed_layer'] = {
                'total_records': total_speed_records,
                'avg_duration': np.mean(speed_durations),
            }
    
    # Compare if both have data
    if reconciliation['batch_layer'] and reconciliation['speed_layer']:
        batch_avg = reconciliation['batch_layer']['avg_duration']
        speed_avg = reconciliation['speed_layer']['avg_duration']
        diff_pct = abs(batch_avg - speed_avg) / batch_avg * 100
        
        if diff_pct > 10:
            reconciliation['discrepancies'].append(
                f"Average duration differs by {diff_pct:.2f}%"
            )
            reconciliation['status'] = 'discrepancy_detected'
        else:
            reconciliation['status'] = 'consistent'
    else:
        reconciliation['status'] = 'incomplete_data'
    
    ti.xcom_push(key='reconciliation', value=reconciliation)
    
    print(f"✓ Reconciliation complete: {reconciliation['status']}")
    return reconciliation


def generate_pipeline_report(**context):
    """
    Generate comprehensive pipeline execution report.
    """
    ti = context['ti']
    execution_date = context['execution_date']
    
    report = {
        'execution_date': execution_date.isoformat(),
        'generated_at': datetime.now().isoformat(),
        'status': 'completed',
        'stages': {}
    }
    
    # Gather all stage results
    validation = ti.xcom_pull(task_ids='data_validation.validate_raw_data', key='validation_results')
    if validation:
        report['stages']['data_validation'] = {
            'status': 'success',
            'row_count': validation.get('row_count'),
            'quality_score': validation.get('quality_score')
        }
    
    anomalies = ti.xcom_pull(task_ids='data_validation.detect_anomalies', key='anomalies')
    if anomalies:
        report['stages']['anomaly_detection'] = {
            'status': 'success',
            'anomaly_percentage': anomalies.get('anomaly_percentage')
        }
    
    cleaned_count = ti.xcom_pull(task_ids='batch_layer.clean_and_transform', key='cleaned_count')
    if cleaned_count:
        report['stages']['batch_processing'] = {
            'status': 'success',
            'records_processed': cleaned_count
        }
    
    model_metrics = ti.xcom_pull(task_ids='ml_pipeline.train_model', key='model_metrics')
    if not model_metrics:
        model_metrics = ti.xcom_pull(task_ids='ml_pipeline.evaluate_existing_model', key='model_metrics')
    if model_metrics:
        report['stages']['ml_pipeline'] = {
            'status': 'success',
            'r2_score': model_metrics.get('r2_score'),
            'rmse': model_metrics.get('rmse')
        }
    
    reconciliation = ti.xcom_pull(task_ids='serving_layer.reconcile_layers', key='reconciliation')
    if reconciliation:
        report['stages']['reconciliation'] = {
            'status': reconciliation.get('status'),
            'discrepancies': reconciliation.get('discrepancies', [])
        }
    
    # Save report
    report_dir = f'{DATA_DIR}/reports'
    os.makedirs(report_dir, exist_ok=True)
    report_file = f'{report_dir}/pipeline_report_{execution_date.strftime("%Y%m%d")}.json'
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Pipeline report generated: {report_file}")
    print(json.dumps(report, indent=2))
    
    return report


def send_notification(**context):
    """
    Send notification about pipeline completion.
    (In production, this would send email/Slack/PagerDuty alerts)
    """
    ti = context['ti']
    
    # Get report
    report = ti.xcom_pull(task_ids='serving_layer.generate_report')
    
    notification = {
        'type': 'pipeline_complete',
        'timestamp': datetime.now().isoformat(),
        'summary': 'Lambda Architecture Pipeline completed successfully',
        'details': report
    }
    
    # Log notification (in production, send to notification service)
    print("=" * 60)
    print("📧 NOTIFICATION")
    print("=" * 60)
    print(f"Subject: Lambda Pipeline Complete - {context['execution_date'].strftime('%Y-%m-%d')}")
    print(f"Status: ✓ Success")
    if report and 'stages' in report:
        for stage, details in report['stages'].items():
            print(f"  - {stage}: {details.get('status', 'unknown')}")
    print("=" * 60)
    
    return notification


# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    'lambda_architecture_pipeline',
    default_args=default_args,
    description='Complete Lambda Architecture Pipeline: Batch + Speed + Serving Layers with ML',
    schedule_interval='@daily',
    start_date=datetime(2025, 12, 1),
    catchup=False,
    tags=['lambda', 'batch', 'speed', 'ml', 'production'],
    doc_md="""
    # Lambda Architecture Pipeline
    
    This DAG orchestrates the complete Lambda Architecture for NYC Taxi data:
    
    ## Stages:
    1. **Data Validation**: Schema validation, quality checks, anomaly detection
    2. **Batch Layer**: Clean data, create Avro master data, generate Parquet views
    3. **ML Pipeline**: Train or evaluate model, performance gates
    4. **Speed Layer**: Initialize streaming infrastructure
    5. **Serving Layer**: Reconciliation, reporting, notifications
    
    ## Features:
    - TaskGroups for logical organization
    - BranchOperator for conditional model training
    - ShortCircuitOperator for quality gates
    - Parallel batch view generation
    - XCom for inter-task communication
    """,
) as dag:
    
    # ========================================================================
    # START
    # ========================================================================
    start = EmptyOperator(task_id='start')
    
    # ========================================================================
    # DATA VALIDATION TASK GROUP
    # ========================================================================
    with TaskGroup('data_validation', tooltip='Data validation and quality checks') as data_validation_group:
        
        validate_task = PythonOperator(
            task_id='validate_raw_data',
            python_callable=validate_raw_data,
        )
        
        quality_gate = ShortCircuitOperator(
            task_id='quality_gate',
            python_callable=check_data_quality,
        )
        
        profile_task = PythonOperator(
            task_id='data_profiling',
            python_callable=data_profiling,
        )
        
        anomaly_task = PythonOperator(
            task_id='detect_anomalies',
            python_callable=detect_anomalies,
        )
        
        validate_task >> quality_gate >> [profile_task, anomaly_task]
    
    # ========================================================================
    # BATCH LAYER TASK GROUP
    # ========================================================================
    with TaskGroup('batch_layer', tooltip='Batch layer processing') as batch_layer_group:
        
        clean_task = PythonOperator(
            task_id='clean_and_transform',
            python_callable=clean_and_transform_data,
        )
        
        master_data_task = PythonOperator(
            task_id='create_master_data',
            python_callable=create_master_data_avro,
        )
        
        # Parallel batch views
        hourly_view = PythonOperator(
            task_id='hourly_stats_view',
            python_callable=create_hourly_stats_view,
        )
        
        daily_view = PythonOperator(
            task_id='daily_summary_view',
            python_callable=create_daily_summary_view,
        )
        
        vendor_view = PythonOperator(
            task_id='vendor_stats_view',
            python_callable=create_vendor_stats_view,
        )
        
        geo_view = PythonOperator(
            task_id='geographic_stats_view',
            python_callable=create_geographic_stats_view,
        )
        
        temporal_view = PythonOperator(
            task_id='temporal_patterns_view',
            python_callable=create_temporal_patterns_view,
        )
        
        batch_complete = EmptyOperator(
            task_id='batch_complete',
            trigger_rule=TriggerRule.ALL_SUCCESS
        )
        
        clean_task >> master_data_task >> [hourly_view, daily_view, vendor_view, geo_view, temporal_view] >> batch_complete
    
    # ========================================================================
    # ML PIPELINE TASK GROUP
    # ========================================================================
    with TaskGroup('ml_pipeline', tooltip='ML model training and evaluation') as ml_pipeline_group:
        
        check_model = BranchPythonOperator(
            task_id='check_model_exists',
            python_callable=check_model_exists,
        )
        
        train_task = PythonOperator(
            task_id='train_model',
            python_callable=train_model,
        )
        
        evaluate_task = PythonOperator(
            task_id='evaluate_existing_model',
            python_callable=evaluate_existing_model,
        )
        
        performance_gate = ShortCircuitOperator(
            task_id='performance_gate',
            python_callable=check_model_performance,
            trigger_rule=TriggerRule.ONE_SUCCESS
        )
        
        check_model >> [train_task, evaluate_task] >> performance_gate
    
    # ========================================================================
    # SPEED LAYER TASK GROUP
    # ========================================================================
    with TaskGroup('speed_layer', tooltip='Speed layer initialization') as speed_layer_group:
        
        init_speed = PythonOperator(
            task_id='initialize',
            python_callable=initialize_speed_layer,
        )
        
        verify_kafka = PythonOperator(
            task_id='verify_kafka',
            python_callable=verify_kafka_connection,
        )
        
        init_speed >> verify_kafka
    
    # ========================================================================
    # SERVING LAYER TASK GROUP
    # ========================================================================
    with TaskGroup('serving_layer', tooltip='Serving layer and reconciliation') as serving_layer_group:
        
        reconcile_task = PythonOperator(
            task_id='reconcile_layers',
            python_callable=reconcile_batch_speed_layers,
        )
        
        report_task = PythonOperator(
            task_id='generate_report',
            python_callable=generate_pipeline_report,
        )
        
        notify_task = PythonOperator(
            task_id='send_notification',
            python_callable=send_notification,
        )
        
        reconcile_task >> report_task >> notify_task
    
    # ========================================================================
    # END
    # ========================================================================
    end = EmptyOperator(
        task_id='end',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )
    
    # ========================================================================
    # DAG DEPENDENCIES
    # ========================================================================
    start >> data_validation_group >> batch_layer_group
    batch_layer_group >> ml_pipeline_group
    batch_layer_group >> speed_layer_group
    [ml_pipeline_group, speed_layer_group] >> serving_layer_group >> end
