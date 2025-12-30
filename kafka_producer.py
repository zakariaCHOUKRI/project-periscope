"""
Kafka Producer - Streams NYC Taxi data to Kafka.
Simulates real-time data ingestion.
"""

import json
import time
import pandas as pd
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import sys

KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KAFKA_TOPIC = 'nyc-taxi-stream'
DATA_FILE = 'data/train.csv'

def wait_for_kafka(max_retries=30, retry_interval=2):
    """Wait for Kafka to be available."""
    print("Waiting for Kafka to be available...")
    for i in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            print("✓ Connected to Kafka!")
            return producer
        except NoBrokersAvailable:
            print(f"  Attempt {i+1}/{max_retries}: Kafka not ready, retrying...")
            time.sleep(retry_interval)
    
    print("✗ Could not connect to Kafka after max retries")
    sys.exit(1)

def stream_data(producer, num_records=100, delay=0.5):
    """Stream taxi records to Kafka."""
    print(f"\nLoading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Select only the records we need
    df = df.head(num_records)
    print(f"Streaming {len(df)} records to topic '{KAFKA_TOPIC}'...")
    
    for index, row in df.iterrows():
        # Create record with simulated live timestamp
        record = {
            'id': row['id'],
            'vendor_id': int(row['vendor_id']),
            'pickup_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'passenger_count': int(row['passenger_count']),
            'pickup_longitude': float(row['pickup_longitude']),
            'pickup_latitude': float(row['pickup_latitude']),
            'dropoff_longitude': float(row['dropoff_longitude']),
            'dropoff_latitude': float(row['dropoff_latitude']),
            'store_and_fwd_flag': row['store_and_fwd_flag'],
            'trip_duration': int(row['trip_duration'])
        }
        
        # Send to Kafka
        producer.send(KAFKA_TOPIC, value=record)
        
        if (index + 1) % 10 == 0:
            print(f"  Sent {index + 1}/{len(df)} records...")
        
        time.sleep(delay)  # Throttle to simulate real-time
    
    producer.flush()
    print(f"\n✓ Successfully streamed {len(df)} records to Kafka!")

def main():
    print("=" * 50)
    print("NYC Taxi Data Producer")
    print("=" * 50)
    
    # Connect to Kafka
    producer = wait_for_kafka()
    
    try:
        # Stream data (adjust num_records and delay as needed)
        stream_data(producer, num_records=5000, delay=0.3)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        producer.close()
        print("Producer closed.")

if __name__ == "__main__":
    main()
