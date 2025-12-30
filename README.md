# Project Periscope - Lambda Architecture Implementation

A complete Lambda Architecture implementation for NYC Taxi data processing and visualization using Apache Kafka, Apache Spark, Apache Hive, Apache Airflow, Apache Submarine (ML), and Streamlit.

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

                    ┌─────────────────────────────────────────────────────────┐
                    │              APACHE SUBMARINE (ML Platform)              │
                    │   - Experiment Tracking & Model Registry                 │
                    │   - Distributed Training (PyTorch/TensorFlow)            │
                    │   - Model Serving & Deployment                           │
                    └─────────────────────────────────────────────────────────┘
```

## Components

### 1. Machine Learning with Apache Submarine (Phase 1)
- **File**: `train_model.py`
- **Training Data**: Full dataset (~1.4 million records from `train.csv`)
- **Data Cleaning**: Removes outliers (trips < 1 min or > 2 hours)
- **ML Platform**: Apache Submarine for experiment tracking and model management
- **Framework**: PyTorch Deep Neural Network for distributed training
- **Features**:
  - Submarine experiment tracking and logging
  - Model registry integration
  - Distributed training support (GPU/CPU)
  - Baseline comparison with GradientBoosting
- **Output**: 
  - `model/taxi_model.pkl` - Trained model artifact
  - `model/experiment_metadata.json` - Submarine experiment metadata

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
pip install pandas scikit-learn numpy kafka-python streamlit pyspark fastavro pyarrow torch apache-submarine
```

### Step 2: Train the ML Model

```bash
python train_model.py
```

This uses Apache Submarine for experiment tracking and trains:
- PyTorch Deep Neural Network (Submarine-compatible for distributed training)
- GradientBoosting baseline for comparison
- Logs metrics to Submarine experiment tracking
- Registers model to Submarine Model Registry

### Step 3: Create Batch Layer Data

```bash
python batch_processor.py
```

### Step 4: Start All Infrastructure

First, create a `.env` file to set the Airflow user ID (required for proper permissions):

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
```

Then start all services:

```bash
docker-compose up -d
```

Wait about 60-90 seconds for all services to initialize (Airflow needs extra time for database migration and pip installs).

### Step 5: Verify Services are Running

```bash
docker-compose ps
```

All services should show as "Up" or "healthy".

### Step 6: Run the Streaming Pipeline

Open 3 terminal windows and run:

**Terminal 1 - Speed Layer (Spark Streaming with ML):**
```bash
cd ~/project-periscope
source .venv/bin/activate
python spark_streaming.py
```

**Terminal 2 - Data Producer (Streams to Kafka):**
```bash
cd ~/project-periscope
source .venv/bin/activate
python kafka_producer.py
```

**Terminal 3 - Dashboard (Serving Layer):**
```bash
cd ~/project-periscope
source .venv/bin/activate
streamlit run app.py
```

### Step 7: Trigger the Lambda Architecture Pipeline (Optional)

The main Airflow DAG runs daily, but you can trigger it manually:

1. Open Airflow UI: http://localhost:8081 (admin / admin)
2. Find `lambda_architecture_pipeline` DAG
3. Click the play button to trigger manually

This will run the full pipeline: data validation → batch processing → ML training → reconciliation.

## Accessing the Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **Streamlit Dashboard** | http://localhost:8501 | - |
| **Airflow UI** | http://localhost:8081 | admin / admin |
| **Spark Master UI** | http://localhost:8085 | - |
| **Submarine UI** | http://localhost:8080 | - |
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

## Airflow DAGs - Pipeline Orchestration

The project includes **4 comprehensive Airflow DAGs** that orchestrate the entire Lambda Architecture:

### 1. Lambda Architecture Pipeline (`lambda_architecture_pipeline_dag.py`) ⭐
**The main production DAG with 20+ tasks organized in 5 TaskGroups:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAMBDA ARCHITECTURE PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────┐     │
│  │  DATA VALIDATION │    │              BATCH LAYER                    │     │
│  │  ───────────────│    │  ─────────────────────────────────────────  │     │
│  │  • validate_raw │───▶│  • clean_and_transform                      │     │
│  │  • quality_gate │    │  • create_master_data (Avro)                │     │
│  │  • data_profiling│    │  ├─▶ hourly_stats_view ─────┐              │     │
│  │  • detect_anomalies│   │  ├─▶ daily_summary_view ───┤              │     │
│  └─────────────────┘    │  ├─▶ vendor_stats_view ─────┼─▶ batch_done │     │
│                         │  ├─▶ geographic_stats_view ─┤              │     │
│                         │  └─▶ temporal_patterns_view ┘              │     │
│                         └─────────────────────────────────────────────┘     │
│                                        │                                     │
│           ┌────────────────────────────┴────────────────────────────┐       │
│           ▼                                                          ▼       │
│  ┌─────────────────┐                                      ┌─────────────────┐│
│  │   ML PIPELINE   │                                      │   SPEED LAYER   ││
│  │  ───────────────│                                      │  ───────────────││
│  │  • check_model  │                                      │  • initialize   ││
│  │  ├─▶ train_model│                                      │  • verify_kafka ││
│  │  └─▶ evaluate   │                                      └─────────────────┘│
│  │  • performance_ │                                              │          │
│  │    gate         │                                              │          │
│  └─────────────────┘                                              │          │
│           │                                                        │          │
│           └────────────────────────┬───────────────────────────────┘          │
│                                    ▼                                          │
│                         ┌─────────────────┐                                   │
│                         │  SERVING LAYER  │                                   │
│                         │  ───────────────│                                   │
│                         │  • reconcile_   │                                   │
│                         │    layers       │                                   │
│                         │  • generate_    │                                   │
│                         │    report       │                                   │
│                         │  • send_        │                                   │
│                         │    notification │                                   │
│                         └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Features:**
- **TaskGroups** for logical organization
- **BranchPythonOperator** for conditional model training vs evaluation
- **ShortCircuitOperator** for quality gates (fails pipeline on bad data)
- **Parallel execution** of 5 batch views simultaneously
- **XCom** for passing data between tasks
- **Comprehensive reporting** and notifications

### 2. Streaming Pipeline Monitor (`streaming_pipeline_monitor_dag.py`) ⭐
**Real-time monitoring DAG running every 5 minutes:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STREAMING PIPELINE MONITOR                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐         ┌─────────────────┐                            │
│  │ KAFKA MONITORING│         │ SPEED LAYER MON │                            │
│  │  • broker_health│         │  • freshness    │                            │
│  │  • topic_health │         │  • throughput   │                            │
│  │  • consumer_lag │         │  • quality      │                            │
│  └────────┬────────┘         └────────┬────────┘                            │
│           │                           │                                      │
│           └───────────┬───────────────┘                                      │
│                       ▼                                                      │
│           ┌─────────────────┐                                               │
│           │ PIPELINE CONTROL│                                               │
│           │  • check_producer│                                              │
│           │  • dashboard_data│                                              │
│           └────────┬────────┘                                               │
│                    ▼                                                         │
│           ┌─────────────────┐                                               │
│           │ CIRCUIT BREAKER │                                               │
│           │  • evaluate     │                                               │
│           └────────┬────────┘                                               │
│                    │                                                         │
│        ┌───────────┴───────────┐                                            │
│        ▼                       ▼                                            │
│  ┌───────────┐          ┌───────────┐                                       │
│  │SEND ALERT │          │LOG HEALTHY│                                       │
│  └───────────┘          └───────────┘                                       │
│                    │                                                         │
│                    ▼                                                         │
│           ┌─────────────────┐                                               │
│           │    CLEANUP      │                                               │
│           └─────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Features:**
- **Circuit breaker pattern** for automatic alerting
- **Kafka health monitoring** (brokers, topics, consumer lag)
- **Speed layer metrics** (freshness, throughput, quality)
- **Dashboard data generation** for operational visibility
- **Auto-cleanup** of old metrics files

### 3. Taxi API Simulation (`taxi_api_simulation_dag.py`)
Simulates real-time API by streaming data to Kafka every 5 minutes.

### 4. Taxi Batch Processing (`taxi_batch_processing_dag.py`)
Basic batch processing: Avro ingestion → Parquet views.

## File Structure

```
project-periscope/
├── train_model.py          # ML model training with Apache Submarine
├── kafka_producer.py       # Stream data to Kafka
├── kafka_consumer.py       # Speed layer with ML inference
├── spark_streaming.py      # Spark streaming processor
├── batch_processor.py      # Batch layer processor
├── app.py                  # Streamlit dashboard
├── docker-compose.yaml     # Infrastructure
├── model/
│   ├── taxi_model.pkl      # Trained ML model
│   └── experiment_metadata.json  # Submarine experiment metadata
├── data/
│   ├── train.csv           # Full training data (~1.4M records)
│   ├── test.csv            # Test data
│   ├── avro/               # Master data (Avro)
│   ├── parquet/            # Batch views (Parquet)
│   ├── speed_layer/        # Real-time data (JSON)
│   ├── metrics/            # Streaming metrics (from monitor DAG)
│   └── reports/            # Pipeline reports (from lambda DAG)
└── airflow/
    └── dags/
        ├── lambda_architecture_pipeline_dag.py  # ⭐ Main pipeline (20+ tasks)
        ├── streaming_pipeline_monitor_dag.py    # ⭐ Real-time monitoring
        ├── taxi_api_simulation_dag.py           # API simulation
        └── taxi_batch_processing_dag.py         # Basic batch processing
```

## Technologies Used

| Component | Technology |
|-----------|------------|
| Streaming | Apache Kafka 7.4.0 |
| Speed Layer | Apache Spark 3.5.0 / Python Consumer |
| Batch Layer | Apache Hive 3.1.3 + Avro/Parquet |
| Orchestration | Apache Airflow 2.7.3 |
| ML Platform | Apache Submarine (Experiment Tracking, Model Registry) |
| ML Framework | PyTorch (Deep Neural Network) |
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
