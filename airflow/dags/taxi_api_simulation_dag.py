"""
Airflow DAG: API Simulation
Simulates a real-time API by streaming NYC Taxi data to Kafka.
This DAG reads from the taxi dataset and publishes records to Kafka at intervals.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import json
import time
import pandas as pd
import os

# Configuration
KAFKA_BOOTSTRAP_SERVERS = ['kafka:29092']  # Internal Docker network
KAFKA_TOPIC = 'nyc-taxi-stream'
DATA_FILE = '/opt/airflow/data/taxi_subset.csv'
BATCH_SIZE = 50  # Records per batch
DELAY_BETWEEN_RECORDS = 0.1  # Seconds

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def stream_taxi_data(**context):
    """Stream taxi records to Kafka, simulating a real-time API."""
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
    
    # Wait for Kafka to be ready
    producer = None
    for attempt in range(10):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            print(f"✓ Connected to Kafka on attempt {attempt + 1}")
            break
        except NoBrokersAvailable:
            print(f"Kafka not ready, retrying... ({attempt + 1}/10)")
            time.sleep(5)
    
    if not producer:
        raise Exception("Could not connect to Kafka")
    
    # Read data
    print(f"Reading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    # Get execution date to determine which batch to send
    execution_date = context['execution_date']
    batch_num = execution_date.hour % (len(df) // BATCH_SIZE)
    start_idx = batch_num * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(df))
    
    batch_df = df.iloc[start_idx:end_idx]
    print(f"Streaming batch {batch_num}: records {start_idx} to {end_idx}")
    
    records_sent = 0
    for _, row in batch_df.iterrows():
        record = {
            'id': row['id'],
            'vendor_id': int(row['vendor_id']),
            'pickup_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'original_pickup_datetime': row['pickup_datetime'],
            'passenger_count': int(row['passenger_count']),
            'pickup_longitude': float(row['pickup_longitude']),
            'pickup_latitude': float(row['pickup_latitude']),
            'dropoff_longitude': float(row['dropoff_longitude']),
            'dropoff_latitude': float(row['dropoff_latitude']),
            'store_and_fwd_flag': row['store_and_fwd_flag'],
            'trip_duration': int(row['trip_duration'])
        }
        
        producer.send(KAFKA_TOPIC, value=record)
        records_sent += 1
        time.sleep(DELAY_BETWEEN_RECORDS)
    
    producer.flush()
    producer.close()
    
    print(f"✓ Successfully sent {records_sent} records to Kafka topic '{KAFKA_TOPIC}'")
    return records_sent

with DAG(
    'taxi_api_simulation',
    default_args=default_args,
    description='Simulates a real-time taxi data API streaming to Kafka',
    schedule_interval=timedelta(minutes=5),  # Run every 5 minutes
    start_date=datetime(2025, 12, 1),
    catchup=False,
    tags=['streaming', 'kafka', 'api'],
) as dag:
    
    stream_task = PythonOperator(
        task_id='stream_to_kafka',
        python_callable=stream_taxi_data,
        provide_context=True,
    )
