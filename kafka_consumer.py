"""
Kafka Consumer - Speed Layer processor with ML inference.
Processes NYC Taxi data from Kafka and makes predictions.
"""

import json
import os
import time
import pickle
import numpy as np
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import pandas as pd

KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KAFKA_TOPIC = 'nyc-taxi-stream'
OUTPUT_DIR = 'data/speed_layer'
MODEL_PATH = 'model/taxi_model.pkl'

def wait_for_kafka(max_retries=30, retry_interval=2):
    """Wait for Kafka to be available."""
    print("Waiting for Kafka to be available...")
    for i in range(max_retries):
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset='latest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=5000
            )
            print("✓ Connected to Kafka!")
            return consumer
        except NoBrokersAvailable:
            print(f"  Attempt {i+1}/{max_retries}: Kafka not ready, retrying...")
            time.sleep(retry_interval)
    
    print("✗ Could not connect to Kafka after max retries")
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km using Haversine formula."""
    R = 6371
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))

def load_model():
    """Load the trained ML model."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            print(f"  ✓ Model loaded (RMSE: {model_data['rmse']:.2f})")
            return model_data
    print("  ⚠ No model found, running without predictions")
    return None

def predict_duration(model_data, record):
    """Make prediction using the ML model."""
    if model_data is None:
        return None, None
    
    try:
        # Parse datetime for features
        pickup_dt = datetime.strptime(record['pickup_datetime'], "%Y-%m-%d %H:%M:%S")
        hour = pickup_dt.hour
        day_of_week = pickup_dt.weekday()
        month = pickup_dt.month
        
        features = np.array([[
            record['passenger_count'],
            record['pickup_longitude'],
            record['pickup_latitude'],
            record['dropoff_longitude'],
            record['dropoff_latitude'],
            hour,
            day_of_week,
            month,
            record['distance_km']
        ]])
        
        prediction = model_data['model'].predict(features)[0]
        error = abs(record['trip_duration'] - prediction)
        return float(prediction), float(error)
    except Exception as e:
        print(f"  Prediction error: {e}")
        return None, None

def main():
    print("=" * 60)
    print("Kafka Consumer - Speed Layer with ML Inference")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("\nLoading ML model...")
    model_data = load_model()
    
    # Connect to Kafka
    consumer = wait_for_kafka()
    if not consumer:
        return
    
    print(f"\nListening on topic '{KAFKA_TOPIC}'...")
    print("Press Ctrl+C to stop\n")
    
    batch = []
    batch_size = 10
    file_counter = 0
    total_processed = 0
    total_prediction_error = 0
    
    try:
        while True:
            # Poll for messages
            for message in consumer:
                record = message.value
                
                # Calculate distance
                record['distance_km'] = haversine_distance(
                    record['pickup_latitude'],
                    record['pickup_longitude'],
                    record['dropoff_latitude'],
                    record['dropoff_longitude']
                )
                
                # Make prediction
                predicted_duration, prediction_error = predict_duration(model_data, record)
                record['predicted_duration'] = predicted_duration
                record['prediction_error'] = prediction_error
                record['processed_time'] = datetime.now().isoformat()
                
                batch.append(record)
                total_processed += 1
                
                if prediction_error:
                    total_prediction_error += prediction_error
                
                # Log with prediction info
                pred_info = f", predicted: {predicted_duration:.0f}s" if predicted_duration else ""
                print(f"  Received: {record['id']} (dist: {record['distance_km']:.2f}km, actual: {record['trip_duration']}s{pred_info})")
                
                # Write batch to file
                if len(batch) >= batch_size:
                    output_file = f"{OUTPUT_DIR}/batch_{file_counter:05d}.json"
                    with open(output_file, 'w') as f:
                        for r in batch:
                            f.write(json.dumps(r) + '\n')
                    
                    avg_error = total_prediction_error / total_processed if total_processed > 0 else 0
                    print(f"  ✓ Wrote batch to {output_file} (avg error: {avg_error:.1f}s)")
                    batch = []
                    file_counter += 1
            
            # Small sleep to prevent busy loop
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping consumer...")
        
        # Write remaining batch
        if batch:
            output_file = f"{OUTPUT_DIR}/batch_{file_counter:05d}.json"
            with open(output_file, 'w') as f:
                for r in batch:
                    f.write(json.dumps(r) + '\n')
            print(f"  ✓ Wrote final batch to {output_file}")
    
    finally:
        consumer.close()
        print(f"\nConsumer closed. Processed {total_processed} records.")
        if total_processed > 0 and model_data:
            print(f"Average prediction error: {total_prediction_error/total_processed:.1f} seconds")

if __name__ == "__main__":
    main()
