**The Strategy:** We must prioritize **RAM efficiency**.
1.  **Do not run everything at once.** Run Phase 1 (Submarine/K8s), save the model, then **shut it down** before starting Phase 2/3 (Kafka/Spark).
2.  **Hybrid Approach:** Run the heavy infrastructure (Kafka, Spark) in Docker, but run the orchestration (Airflow) and Dashboard (Streamlit) natively in your WSL Python environment to save the overhead of extra containers.

Here is your specific implementation guide for `project-periscope`.

### Phase 1: Machine Learning (The Brain)

**Constraint:** Apache Submarine requires Kubernetes.
**Action:** We spin up K8s, do the job, and **destroy it immediately** to free up RAM for the next phases.

1.  **Setup Project Directory:**
    ```bash
    mkdir -p ~/project-periscope/model
    cd ~/project-periscope
    ```

2.  **Start Minikube (The minimal K8s):**
    ```bash
    # Start small
    minikube start --cpus 4 --memory 6000
    ```

3.  **Install Submarine:**
    ```bash
    helm repo add submarine https://apache.github.io/submarine/charts
    helm install submarine submarine/submarine
    kubectl port-forward --namespace default svc/submarine-traefik 8080:80
    ```

4.  **The Task:**
    *   Go to `http://localhost:8080`.
    *   Create a Notebook. Upload your NYC Taxi subset.
    *   Train your `LinearRegression` or `RandomForest`.
    *   **Export:** Save the trained model as `taxi_model.pkl`.
    *   **Download:** In the Notebook UI, download `taxi_model.pkl` to your computer.

5.  **Cleanup (CRITICAL):**
    *   Move `taxi_model.pkl` into `~/project-periscope/model/`.
    *   **Kill Kubernetes** to free up your RAM:
    ```bash
    minikube stop
    minikube delete
    ```

---

### Phase 2 & 3: The Infrastructure (Docker Compose)

Now that K8s is gone, we have RAM for Kafka and Spark.

1.  **Create `docker-compose.yaml`** inside `~/project-periscope/`:

```yaml
version: '3'
services:
  # --- The Messenger ---
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      # Allow connection from inside Docker (spark) AND outside (your WSL python scripts)
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  # --- The Processor ---
  spark-master:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=master
    ports:
      - "8080:8080" # Spark UI
      - "7077:7077"
    volumes:
      - ./model:/opt/bitnami/spark/model  # Mount your model here
      - ./data:/opt/bitnami/spark/data    # Mount data storage

  spark-worker:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_MEMORY=2G  # Limit RAM usage
    depends_on:
      - spark-master
    volumes:
      - ./model:/opt/bitnami/spark/model
      - ./data:/opt/bitnami/spark/data
```

2.  **Start the engines:**
    ```bash
    docker-compose up -d
    ```

---

### Phase 2 Implementation: Ingestion (Airflow)

**RAM Saving Trick:** Don't run Airflow in Docker. Run it locally in a Python Virtual Environment (`venv`) inside WSL.

1.  **Setup Python Environment:**
    ```bash
    cd ~/project-periscope
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    pip install apache-airflow kafka-python pandas pyspark streamlit watchdog
    ```

2.  **Initialize Airflow:**
    ```bash
    export AIRFLOW_HOME=~/project-periscope/airflow
    airflow db init
    airflow users create --username admin --password admin --firstname Peter --lastname Parker --role Admin --email spidey@avengers.com
    ```

3.  **Start Airflow (Background):**
    ```bash
    airflow webserver -p 8081 -D
    airflow scheduler -D
    ```
    *Access Airflow UI at `http://localhost:8081`.*

4.  **Create the Producer DAG:**
    Create file: `~/project-periscope/airflow/dags/taxi_stream_dag.py`
    ```python
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime
    from kafka import KafkaProducer
    import json
    import pandas as pd
    import time

    def stream_data():
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'], # Connects to Docker Kafka
                                 value_serializer=lambda x: json.dumps(x).encode('utf-8'))
        # Read a chunk of your CSV
        df = pd.read_csv('/home/your_user/project-periscope/data/taxi_subset.csv').head(1000)
        
        for index, row in df.iterrows():
            record = row.to_dict()
            record['pickup_datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Simulate Live
            producer.send('nyc-taxi-stream', value=record)
            time.sleep(0.5) # Throttle speed

    with DAG('nyc_taxi_producer', start_date=datetime(2023, 1, 1), schedule_interval='@once') as dag:
        task1 = PythonOperator(task_id='stream_to_kafka', python_callable=stream_data)
    ```

---

### Phase 3 Implementation: The Lambda Code (Spark)

We submit jobs from WSL to the Docker Spark Cluster.

**Path A: Speed Layer (Streaming)**
Create: `~/project-periscope/spark_streaming.py`

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import StructType, StringType, DoubleType
import os

# Connect to the Spark Master container
spark = SparkSession.builder \
    .appName("TaxiSpeedLayer") \
    .master("spark://localhost:7077") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
    .getOrCreate()

# Define Schema based on your CSV
schema = StructType().add("pickup_datetime", StringType()).add("passenger_count", DoubleType()).add("trip_distance", DoubleType())

# Read Stream from Docker Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \ # Note: Internal Docker DNS
    .option("subscribe", "nyc-taxi-stream") \
    .load()

# Deserialize JSON
json_df = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

# TODO: Load your .pkl model here using a UDF (User Defined Function)
# For now, let's just calculate a simple metric to test flow
processed_df = json_df.withColumn("processed_time", current_timestamp())

# Write Stream to Memory (for dashboard querying) or File
query = processed_df.writeStream \
    .outputMode("append") \
    .format("json") \
    .option("path", "/opt/bitnami/spark/data/live_output") \
    .option("checkpointLocation", "/opt/bitnami/spark/data/checkpoint") \
    .start()

query.awaitTermination()
```

**Run it:**
```bash
# You need the Kafka JARs locally to submit
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 spark_streaming.py
```

---

### Phase 4: Visualization (Streamlit)

Create `~/project-periscope/app.py`. Since we mapped volumes, Streamlit (running in WSL) can see the files Spark (running in Docker) wrote to the `data/` folder.

```python
import streamlit as st
import pandas as pd
import glob
import time

st.title("Project Periscope: NYC Taxi Monitor")

placeholder = st.empty()

while True:
    # Read the latest JSON files generated by Spark Streaming
    # Note: Spark writes many small JSON files
    all_files = glob.glob("data/live_output/*.json")
    if all_files:
        latest_file = max(all_files, key=os.path.getctime)
        df = pd.read_json(latest_file, lines=True)
        
        with placeholder.container():
            kpi1, kpi2 = st.columns(2)
            kpi1.metric(label="Latest Trip Distance", value=f"{df.iloc[-1]['trip_distance']} miles")
            kpi2.metric(label="Passenger Count", value=df.iloc[-1]['passenger_count'])
            
            st.subheader("Live Data Stream")
            st.dataframe(df.tail(10))

    time.sleep(2)
```

Run it:
```bash
streamlit run app.py
```

### Summary of Resource Usage
1.  **Minikube:** Runs alone (consumes 6GB). Stopped after model training.
2.  **Docker Compose:** Runs Kafka + Spark (consumes ~4GB).
3.  **WSL Local:** Runs Airflow + Streamlit + VS Code (consumes ~2-3GB).
4.  **Windows OS:** Has ~4-6GB breathing room.

This setup prevents your ThinkPad from freezing while allowing you to build the full pipeline.