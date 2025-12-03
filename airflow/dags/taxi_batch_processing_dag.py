"""
Airflow DAG: Batch Processing Pipeline
Automates the read/write pipeline for the Batch Layer.
- Reads data from source
- Writes to Avro format (master data)
- Creates Parquet batch views
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os

# Configuration
DATA_DIR = '/opt/airflow/data'
MASTER_DATA_PATH = f'{DATA_DIR}/avro'
BATCH_VIEWS_PATH = f'{DATA_DIR}/parquet'
RAW_DATA_FILE = f'{DATA_DIR}/taxi_subset.csv'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def ingest_to_avro(**context):
    """Read raw data and write to Avro format for master data storage."""
    import fastavro
    from fastavro import writer
    
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    print(f"Reading raw data from {RAW_DATA_FILE}")
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Define Avro schema
    schema = {
        'type': 'record',
        'name': 'TaxiTrip',
        'namespace': 'nyc.taxi',
        'fields': [
            {'name': 'id', 'type': 'string'},
            {'name': 'vendor_id', 'type': 'int'},
            {'name': 'pickup_datetime', 'type': 'string'},
            {'name': 'dropoff_datetime', 'type': ['null', 'string'], 'default': None},
            {'name': 'passenger_count', 'type': 'int'},
            {'name': 'pickup_longitude', 'type': 'double'},
            {'name': 'pickup_latitude', 'type': 'double'},
            {'name': 'dropoff_longitude', 'type': 'double'},
            {'name': 'dropoff_latitude', 'type': 'double'},
            {'name': 'store_and_fwd_flag', 'type': 'string'},
            {'name': 'trip_duration', 'type': 'int'},
            {'name': 'ingestion_date', 'type': 'string'},
        ]
    }
    
    # Convert DataFrame to records
    records = []
    for _, row in df.iterrows():
        record = {
            'id': str(row['id']),
            'vendor_id': int(row['vendor_id']),
            'pickup_datetime': str(row['pickup_datetime']),
            'dropoff_datetime': str(row.get('dropoff_datetime', '')),
            'passenger_count': int(row['passenger_count']),
            'pickup_longitude': float(row['pickup_longitude']),
            'pickup_latitude': float(row['pickup_latitude']),
            'dropoff_longitude': float(row['dropoff_longitude']),
            'dropoff_latitude': float(row['dropoff_latitude']),
            'store_and_fwd_flag': str(row['store_and_fwd_flag']),
            'trip_duration': int(row['trip_duration']),
            'ingestion_date': date_str,
        }
        records.append(record)
    
    # Write to Avro
    os.makedirs(MASTER_DATA_PATH, exist_ok=True)
    avro_file = f'{MASTER_DATA_PATH}/taxi_master_{date_str}.avro'
    
    parsed_schema = fastavro.parse_schema(schema)
    with open(avro_file, 'wb') as out:
        writer(out, parsed_schema, records)
    
    print(f"✓ Wrote {len(records)} records to {avro_file}")
    return avro_file

def create_batch_views(**context):
    """Create Parquet batch views with aggregations."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y%m%d')
    
    print(f"Creating batch views for {date_str}")
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Parse datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    
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
    
    os.makedirs(BATCH_VIEWS_PATH, exist_ok=True)
    
    # View 1: Hourly aggregations
    hourly_stats = df.groupby('hour').agg({
        'trip_duration': ['mean', 'count'],
        'distance_km': 'mean',
        'passenger_count': 'mean'
    }).reset_index()
    hourly_stats.columns = ['hour', 'avg_duration', 'trip_count', 'avg_distance', 'avg_passengers']
    
    hourly_table = pa.Table.from_pandas(hourly_stats)
    pq.write_table(hourly_table, f'{BATCH_VIEWS_PATH}/hourly_stats_{date_str}.parquet')
    print(f"✓ Created hourly_stats view")
    
    # View 2: Daily summary
    daily_summary = pd.DataFrame({
        'date': [date_str],
        'total_trips': [len(df)],
        'avg_duration': [df['trip_duration'].mean()],
        'avg_distance': [df['distance_km'].mean()],
        'total_passengers': [df['passenger_count'].sum()],
        'avg_passengers': [df['passenger_count'].mean()],
    })
    
    daily_table = pa.Table.from_pandas(daily_summary)
    pq.write_table(daily_table, f'{BATCH_VIEWS_PATH}/daily_summary_{date_str}.parquet')
    print(f"✓ Created daily_summary view")
    
    # View 3: Vendor stats
    vendor_stats = df.groupby('vendor_id').agg({
        'trip_duration': 'mean',
        'distance_km': 'mean',
        'id': 'count'
    }).reset_index()
    vendor_stats.columns = ['vendor_id', 'avg_duration', 'avg_distance', 'trip_count']
    
    vendor_table = pa.Table.from_pandas(vendor_stats)
    pq.write_table(vendor_table, f'{BATCH_VIEWS_PATH}/vendor_stats_{date_str}.parquet')
    print(f"✓ Created vendor_stats view")
    
    return f"Created 3 batch views for {date_str}"

with DAG(
    'taxi_batch_processing',
    default_args=default_args,
    description='Batch processing pipeline: Avro master data and Parquet batch views',
    schedule_interval='@daily',
    start_date=datetime(2025, 12, 1),
    catchup=False,
    tags=['batch', 'hive', 'parquet', 'avro'],
) as dag:
    
    ingest_task = PythonOperator(
        task_id='ingest_to_avro',
        python_callable=ingest_to_avro,
        provide_context=True,
    )
    
    batch_views_task = PythonOperator(
        task_id='create_batch_views',
        python_callable=create_batch_views,
        provide_context=True,
    )
    
    ingest_task >> batch_views_task
