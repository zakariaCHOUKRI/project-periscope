"""
Batch Processor - Creates Avro master data and Parquet batch views.
Can be run standalone or triggered by Airflow.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import fastavro
from fastavro import writer
import pyarrow as pa
import pyarrow.parquet as pq

# Configuration
DATA_DIR = 'data'
MASTER_DATA_PATH = f'{DATA_DIR}/avro'
BATCH_VIEWS_PATH = f'{DATA_DIR}/parquet'
RAW_DATA_FILE = f'{DATA_DIR}/taxi_subset.csv'

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km using Haversine formula."""
    R = 6371
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def ingest_to_avro():
    """Read raw data and write to Avro format for master data storage."""
    date_str = datetime.now().strftime('%Y%m%d')
    
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

def create_batch_views():
    """Create Parquet batch views with aggregations."""
    date_str = datetime.now().strftime('%Y%m%d')
    
    print(f"Creating batch views for {date_str}")
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Parse datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    
    # Calculate distance
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

def main():
    print("=" * 60)
    print("Batch Processor - Lambda Architecture Batch Layer")
    print("=" * 60)
    
    # Create Avro master data
    print("\n[1/2] Creating Avro master data...")
    ingest_to_avro()
    
    # Create Parquet batch views
    print("\n[2/2] Creating Parquet batch views...")
    create_batch_views()
    
    print("\n" + "=" * 60)
    print("✓ Batch processing complete!")
    print(f"  - Master data: {MASTER_DATA_PATH}/")
    print(f"  - Batch views: {BATCH_VIEWS_PATH}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
