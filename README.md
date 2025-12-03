# Project Periscope - Lambda Architecture Implementation

A complete Lambda Architecture implementation for NYC Taxi data processing and visualization using Apache Kafka, Apache Spark, Apache Hive, Apache Airflow, and Streamlit.

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    DATA SOURCE                          │
                    │              (NYC Taxi CSV Dataset)                     │
                    └─────────────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
    ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
    │    INGESTION LAYER    │  │    INGESTION LAYER    │  │      BATCH LAYER      │
    │   (Airflow DAG/API)   │  │   (Kafka Producer)    │  │   (Batch Processor)   │
    └───────────────────────┘  └───────────────────────┘  └───────────────────────┘
                    │                       │                       │
                    ▼                       ▼                       ▼
    ┌───────────────────────────────────────────────┐  ┌───────────────────────┐
    │              APACHE KAFKA                      │  │    MASTER DATA        │
    │         (Message Streaming)                    │  │   (Avro Format)       │
    │         Topic: nyc-taxi-stream                 │  └───────────────────────┘
    └───────────────────────────────────────────────┘              │
                    │                                              ▼
        ┌───────────┴───────────┐                     ┌───────────────────────┐
        ▼                       ▼                     │    BATCH VIEWS        │
┌───────────────────┐  ┌───────────────────┐          │  (Parquet Format)     │
│   SPEED LAYER     │  │   SPEED LAYER     │          │  - Hourly Stats       │
│ (Kafka Consumer)  │  │(Spark Streaming)  │          │  - Daily Summary      │
│  + ML Inference   │  │ + Temp Tables     │          │  - Vendor Stats       │
└───────────────────┘  └───────────────────┘          └───────────────────────┘
        │                       │                               │
        ▼                       ▼                               │
┌───────────────────────────────────────────────────────────────┴──────────────┐
│                           SERVING LAYER                                       │
│                      (Streamlit Dashboard)                                    │
│   - Real-time Speed Layer Views                                               │
│   - Batch Layer Historical Analytics                                          │
│   - ML Model Predictions & Error Analysis                                     │
│   - Combined Lambda View                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Machine Learning (Phase 1)
- **File**: `train_model.py`
- **Model**: RandomForest for trip duration prediction
- **Output**: `model/taxi_model.pkl`

### 2. Streaming Ingestion (Phase 2)
- **Kafka Producer**: `kafka_producer.py` - Streams taxi data to Kafka
- **Airflow DAG**: `airflow/dags/taxi_api_simulation_dag.py` - Simulates API

### 3. Speed Layer (Phase 3)
- **Kafka Consumer**: `kafka_consumer.py` - Simple consumer with ML inference
- **Spark Streaming**: `spark_streaming.py` - Full Spark with temp tables
- **Output**: `data/speed_layer/*.json`

### 4. Batch Layer (Phase 3)
- **Batch Processor**: `batch_processor.py`
- **Airflow DAG**: `airflow/dags/taxi_batch_processing_dag.py`
- **Master Data**: `data/avro/` (Avro format)
- **Batch Views**: `data/parquet/` (Parquet format)
  - `hourly_stats_*.parquet`
  - `daily_summary_*.parquet`
  - `vendor_stats_*.parquet`

### 5. Serving Layer (Phase 4)
- **Dashboard**: `app.py` (Streamlit)
- **Features**:
  - Speed Layer real-time view with ML predictions
  - Batch Layer historical analytics
  - Combined Lambda Architecture view

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- At least 8GB RAM available

## Quick Start

### Step 1: Clone and Setup Environment

```bash
cd ~/project-periscope
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn numpy kafka-python streamlit pyspark fastavro pyarrow
```

### Step 2: Train the ML Model

```bash
python train_model.py
```

### Step 3: Create Batch Layer Data

```bash
python batch_processor.py
```

### Step 4: Start All Infrastructure

```bash
docker-compose up -d
```

Wait about 60 seconds for all services to initialize.

### Step 5: Verify Services are Running

```bash
docker-compose ps
```

All services should show as "Up" or "healthy".

## Running the Pipeline

### Option A: Simple Pipeline (Recommended for Testing)

**Terminal 1 - Start the Consumer:**
```bash
source .venv/bin/activate
python kafka_consumer.py
```

**Terminal 2 - Start the Producer:**
```bash
source .venv/bin/activate
python kafka_producer.py
```

**Terminal 3 - Start the Dashboard:**
```bash
source .venv/bin/activate
streamlit run app.py
```

### Option B: Full Spark Streaming Pipeline

**Terminal 1 - Start Spark Streaming:**
```bash
source .venv/bin/activate
python spark_streaming.py
```

**Terminal 2 - Start the Producer:**
```bash
source .venv/bin/activate
python kafka_producer.py
```

**Terminal 3 - Start the Dashboard:**
```bash
source .venv/bin/activate
streamlit run app.py
```

## Accessing the Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **Streamlit Dashboard** | http://localhost:8501 | - |
| **Airflow UI** | http://localhost:8081 | admin / admin |
| **Spark Master UI** | http://localhost:8085 | - |
| **Hive Server** | localhost:10000 | - |
| **Kafka** | localhost:9092 | - |

## Using the Dashboard

1. **Speed Layer View**: Shows real-time streaming data with ML predictions
2. **Batch Layer View**: Shows historical aggregations from Parquet files
3. **Combined View**: Compares Speed Layer vs Batch Layer metrics

Toggle between views using the sidebar radio buttons.

## Shutting Down

### Stop the Dashboard and Python Processes
Press `Ctrl+C` in each terminal running Python scripts.

### Stop Docker Infrastructure
```bash
docker-compose down
```

### Stop and Remove All Data (Clean Shutdown)
```bash
docker-compose down -v
rm -rf data/speed_layer/* data/avro/* data/parquet/*
```

## File Structure

```
project-periscope/
├── train_model.py          # ML model training
├── kafka_producer.py       # Stream data to Kafka
├── kafka_consumer.py       # Speed layer with ML inference
├── spark_streaming.py      # Spark streaming processor
├── batch_processor.py      # Batch layer processor
├── app.py                  # Streamlit dashboard
├── docker-compose.yaml     # Infrastructure
├── model/
│   └── taxi_model.pkl      # Trained ML model
├── data/
│   ├── train.csv           # Full training data
│   ├── test.csv            # Test data
│   ├── taxi_subset.csv     # Sample data (10k records)
│   ├── avro/               # Master data (Avro)
│   ├── parquet/            # Batch views (Parquet)
│   └── speed_layer/        # Real-time data (JSON)
└── airflow/
    └── dags/
        ├── taxi_api_simulation_dag.py
        └── taxi_batch_processing_dag.py
```

## Technologies Used

| Component | Technology |
|-----------|------------|
| Streaming | Apache Kafka 7.4.0 |
| Speed Layer | Apache Spark 3.5.0 / Python Consumer |
| Batch Layer | Apache Hive 3.1.3 + Avro/Parquet |
| Orchestration | Apache Airflow 2.7.3 |
| ML | Scikit-learn (RandomForest) |
| Visualization | Streamlit |
| Database | PostgreSQL 13 (Hive Metastore) |
| Infrastructure | Docker Compose |

## Troubleshooting

### Kafka Connection Issues
```bash
# Check Kafka logs
docker-compose logs kafka

# Restart Kafka
docker-compose restart kafka
```

### No Data in Dashboard
1. Ensure consumer is running before producer
2. Check the `data/speed_layer/` directory for JSON files
3. Verify Kafka is healthy: `docker-compose ps`

### Airflow DAG Not Visible
```bash
# Check Airflow logs
docker-compose logs airflow

# Restart Airflow
docker-compose restart airflow
```

### Out of Memory
- Reduce Spark worker memory in `docker-compose.yaml`
- Use the simple Kafka consumer instead of Spark streaming
- Stop unnecessary services: `docker-compose stop spark-master spark-worker`
