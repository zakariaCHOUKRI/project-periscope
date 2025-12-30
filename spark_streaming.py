"""
Spark Streaming - Speed Layer of Lambda Architecture.
Processes NYC Taxi data from Kafka in real-time with:
- Temporary tables for real-time views
- ML model inference for trip duration predictions
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, current_timestamp, udf, struct, lit,
    window, avg, count, sum as spark_sum
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, FloatType, TimestampType
)
from pyspark.ml.feature import VectorAssembler
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Schema matching the Kafka producer records
schema = StructType([
    StructField("id", StringType(), True),
    StructField("vendor_id", IntegerType(), True),
    StructField("pickup_datetime", StringType(), True),
    StructField("original_pickup_datetime", StringType(), True),
    StructField("passenger_count", IntegerType(), True),
    StructField("pickup_longitude", DoubleType(), True),
    StructField("pickup_latitude", DoubleType(), True),
    StructField("dropoff_longitude", DoubleType(), True),
    StructField("dropoff_latitude", DoubleType(), True),
    StructField("store_and_fwd_flag", StringType(), True),
    StructField("trip_duration", IntegerType(), True)
])

# Load ML model globally
MODEL_PATH = "model/taxi_model.pkl"
model_data = None

def load_ml_model():
    """Load the trained ML model."""
    global model_data
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            print(f"✓ Model loaded (RMSE: {model_data['rmse']:.2f})")
            return model_data
    print("⚠ No model found")
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km using Haversine formula."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0
    R = 6371
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))

def predict_duration(passenger_count, pickup_lon, pickup_lat, 
                     dropoff_lon, dropoff_lat, hour, day_of_week, month, distance_km):
    """Predict trip duration using the ML model."""
    global model_data
    if model_data is None:
        return None
    
    try:
        features = np.array([[
            passenger_count, pickup_lon, pickup_lat,
            dropoff_lon, dropoff_lat, hour, day_of_week, month, distance_km
        ]])
        prediction = model_data['model'].predict(features)[0]
        return float(prediction)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def create_spark_session():
    """Create Spark session with Kafka support."""
    return SparkSession.builder \
        .appName("TaxiSpeedLayer") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.0") \
        .config("spark.sql.streaming.checkpointLocation", "data/checkpoint") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

def process_batch(batch_df, batch_id):
    """Process each micro-batch: calculate features and make predictions."""
    global model_data
    
    if batch_df.isEmpty():
        return
    
    print(f"\n--- Processing Batch {batch_id} ---")
    
    # Convert to Pandas for easier processing
    pdf = batch_df.toPandas()
    
    # Parse datetime and extract features
    pdf['pickup_datetime'] = pd.to_datetime(pdf['pickup_datetime'])
    pdf['hour'] = pdf['pickup_datetime'].dt.hour
    pdf['day_of_week'] = pdf['pickup_datetime'].dt.dayofweek
    pdf['month'] = pdf['pickup_datetime'].dt.month
    
    # Calculate distance
    pdf['distance_km'] = pdf.apply(
        lambda row: haversine_distance(
            row['pickup_latitude'], row['pickup_longitude'],
            row['dropoff_latitude'], row['dropoff_longitude']
        ), axis=1
    )
    
    # Make predictions if model is available
    if model_data is not None:
        features = ['passenger_count', 'pickup_longitude', 'pickup_latitude',
                   'dropoff_longitude', 'dropoff_latitude', 'hour', 
                   'day_of_week', 'month', 'distance_km']
        X = pdf[features].fillna(0)
        pdf['predicted_duration'] = model_data['model'].predict(X)
        pdf['prediction_error'] = abs(pdf['trip_duration'] - pdf['predicted_duration'])
    else:
        pdf['predicted_duration'] = None
        pdf['prediction_error'] = None
    
    pdf['processed_time'] = datetime.now().isoformat()
    pdf['batch_id'] = batch_id
    
    # Ensure proper types for Spark DataFrame creation
    pdf['predicted_duration'] = pdf['predicted_duration'].astype(float)
    pdf['prediction_error'] = pdf['prediction_error'].astype(float)
    
    # Write to speed layer output
    output_dir = "data/speed_layer"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/batch_{batch_id:05d}.json"
    pdf.to_json(output_file, orient='records', lines=True, date_format='iso')
    print(f"  ✓ Wrote {len(pdf)} records to {output_file}")
    
    # Print sample predictions
    if model_data is not None:
        sample = pdf[['id', 'distance_km', 'trip_duration', 'predicted_duration', 'prediction_error']].head(3)
        print(f"  Sample predictions:\n{sample.to_string()}")
    
    # Create/update temporary table for real-time views
    spark = SparkSession.builder.getOrCreate()
    
    # Define explicit schema for the DataFrame
    speed_layer_schema = StructType([
        StructField("id", StringType(), True),
        StructField("vendor_id", IntegerType(), True),
        StructField("pickup_datetime", StringType(), True),
        StructField("original_pickup_datetime", StringType(), True),
        StructField("passenger_count", IntegerType(), True),
        StructField("pickup_longitude", DoubleType(), True),
        StructField("pickup_latitude", DoubleType(), True),
        StructField("dropoff_longitude", DoubleType(), True),
        StructField("dropoff_latitude", DoubleType(), True),
        StructField("store_and_fwd_flag", StringType(), True),
        StructField("trip_duration", IntegerType(), True),
        StructField("hour", IntegerType(), True),
        StructField("day_of_week", IntegerType(), True),
        StructField("month", IntegerType(), True),
        StructField("distance_km", DoubleType(), True),
        StructField("predicted_duration", DoubleType(), True),
        StructField("prediction_error", DoubleType(), True),
        StructField("processed_time", StringType(), True),
        StructField("batch_id", IntegerType(), True),
    ])
    
    # Convert pandas timestamp to string for Spark compatibility
    pdf['pickup_datetime'] = pdf['pickup_datetime'].astype(str)
    
    spark_df = spark.createDataFrame(pdf, schema=speed_layer_schema)
    spark_df.createOrReplaceTempView("speed_layer_trips")
    
    # Real-time aggregations (temporary table)
    agg_df = spark.sql("""
        SELECT 
            hour,
            COUNT(*) as trip_count,
            AVG(distance_km) as avg_distance,
            AVG(trip_duration) as avg_actual_duration,
            AVG(predicted_duration) as avg_predicted_duration
        FROM speed_layer_trips
        GROUP BY hour
        ORDER BY hour
    """)
    agg_df.createOrReplaceTempView("speed_layer_hourly_stats")
    print(f"  ✓ Updated temporary tables")

def main():
    print("=" * 60)
    print("Spark Streaming - Speed Layer (Lambda Architecture)")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("data/speed_layer", exist_ok=True)
    os.makedirs("data/checkpoint", exist_ok=True)
    
    # Load ML model
    print("\nLoading ML model...")
    load_ml_model()
    
    # Create Spark session
    print("\nInitializing Spark session...")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # Read stream from Kafka
    print("\nConnecting to Kafka stream...")
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "nyc-taxi-stream") \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()
    
    # Parse JSON from Kafka value
    parsed_df = kafka_df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")
    
    # Start streaming query with foreachBatch for ML predictions
    print("Starting stream processing with ML inference...")
    print("Press Ctrl+C to stop\n")
    
    query = parsed_df.writeStream \
        .outputMode("append") \
        .foreachBatch(process_batch) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\nStopping stream...")
        query.stop()
        print("Stream stopped.")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
