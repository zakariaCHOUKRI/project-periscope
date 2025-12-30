"""
Streamlit Dashboard - Serving Layer of Lambda Architecture.
Visualizes both Speed Layer (real-time) and Batch Layer results with ML predictions.
"""

import streamlit as st
import pandas as pd
import glob
import os
import time
import json
import pyarrow.parquet as pq

st.set_page_config(
    page_title="Project Periscope - Lambda Architecture",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 Project Periscope: NYC Taxi Analytics")
st.markdown("**Lambda Architecture Dashboard** - Real-time Speed Layer + Batch Layer Views")

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 10, 3)

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    ["Speed Layer (Real-time)", "Batch Layer (Historical)", "Combined View"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 System Status")

def check_service_status():
    """Check status of various services."""
    status = {}
    
    # Check Kafka
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 9092))
        status['kafka'] = result == 0
        sock.close()
    except:
        status['kafka'] = False
    
    # Check Speed Layer data
    speed_files = glob.glob("data/speed_layer/*.json")
    status['speed_layer'] = len(speed_files) > 0
    
    # Check Batch Layer data
    batch_files = glob.glob("data/parquet/*.parquet")
    status['batch_layer'] = len(batch_files) > 0
    
    # Check ML Model
    status['ml_model'] = os.path.exists("model/taxi_model.pkl")
    
    return status

status = check_service_status()
st.sidebar.markdown(f"**Kafka:** {'🟢' if status['kafka'] else '🔴'}")
st.sidebar.markdown(f"**Speed Layer:** {'🟢' if status['speed_layer'] else '🟡'}")
st.sidebar.markdown(f"**Batch Layer:** {'🟢' if status['batch_layer'] else '🟡'}")
st.sidebar.markdown(f"**ML Model:** {'🟢' if status['ml_model'] else '🔴'}")

def load_speed_layer_data(max_files=50):
    """Load latest data from Speed Layer."""
    output_path = "data/speed_layer"
    
    if not os.path.exists(output_path):
        return None
    
    all_files = glob.glob(f"{output_path}/*.json")
    if not all_files:
        return None
    
    all_files.sort(key=os.path.getmtime, reverse=True)
    recent_files = all_files[:max_files]
    
    dfs = []
    for file in recent_files:
        try:
            df = pd.read_json(file, lines=True)
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def load_batch_layer_data():
    """Load batch views from Parquet files."""
    parquet_path = "data/parquet"
    batch_data = {}
    
    if not os.path.exists(parquet_path):
        return batch_data
    
    # Load hourly stats
    hourly_files = glob.glob(f"{parquet_path}/hourly_stats_*.parquet")
    if hourly_files:
        latest = max(hourly_files, key=os.path.getmtime)
        batch_data['hourly_stats'] = pd.read_parquet(latest)
    
    # Load daily summary
    daily_files = glob.glob(f"{parquet_path}/daily_summary_*.parquet")
    if daily_files:
        latest = max(daily_files, key=os.path.getmtime)
        batch_data['daily_summary'] = pd.read_parquet(latest)
    
    # Load vendor stats
    vendor_files = glob.glob(f"{parquet_path}/vendor_stats_*.parquet")
    if vendor_files:
        latest = max(vendor_files, key=os.path.getmtime)
        batch_data['vendor_stats'] = pd.read_parquet(latest)
    
    return batch_data

def display_speed_layer():
    """Display Speed Layer (real-time) data."""
    st.header("⚡ Speed Layer - Real-time Processing")
    
    df = load_speed_layer_data()
    
    if df is not None and not df.empty:
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trips", len(df))
        with col2:
            avg_distance = df['distance_km'].mean() if 'distance_km' in df.columns else 0
            st.metric("Avg Distance", f"{avg_distance:.2f} km")
        with col3:
            avg_duration = df['trip_duration'].mean() / 60 if 'trip_duration' in df.columns else 0
            st.metric("Avg Duration", f"{avg_duration:.1f} min")
        with col4:
            if 'predicted_duration' in df.columns and df['predicted_duration'].notna().any():
                mae = df['prediction_error'].mean()
                st.metric("Prediction MAE", f"{mae:.1f} sec")
            else:
                st.metric("Predictions", "N/A")
        
        st.markdown("---")
        
        # ML Predictions Analysis
        if 'predicted_duration' in df.columns and df['predicted_duration'].notna().any():
            st.subheader("🤖 ML Model Predictions")
            
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                # Actual vs Predicted scatter
                st.markdown("**Actual vs Predicted Duration**")
                chart_df = df[['trip_duration', 'predicted_duration']].dropna().head(100)
                st.scatter_chart(chart_df, x='trip_duration', y='predicted_duration')
            
            with pred_col2:
                # Model accuracy metrics
                st.markdown("**Model Accuracy Metrics**")
                if 'prediction_error' in df.columns:
                    error_df = df['prediction_error'].dropna()
                    within_1min = (error_df.abs() <= 60).sum() / len(error_df) * 100
                    within_5min = (error_df.abs() <= 300).sum() / len(error_df) * 100
                    accuracy_data = pd.DataFrame({
                        'Threshold': ['Within 1 min', 'Within 5 min'],
                        'Accuracy (%)': [within_1min, within_5min]
                    })
                    st.bar_chart(accuracy_data.set_index('Threshold'))
        
        st.markdown("---")
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("⏰ Trips by Hour")
            if 'hour' in df.columns:
                hourly_counts = df['hour'].value_counts().sort_index()
                st.bar_chart(hourly_counts)
        
        with chart_col2:
            st.subheader("🚗 Vendor Performance")
            if 'vendor_id' in df.columns and 'trip_duration' in df.columns:
                vendor_stats = df.groupby('vendor_id').agg({
                    'trip_duration': 'mean',
                    'distance_km': 'mean'
                }).round(2)
                vendor_stats.columns = ['Avg Duration (s)', 'Avg Distance (km)']
                vendor_stats.index = vendor_stats.index.map({1: 'Vendor 1', 2: 'Vendor 2'})
                st.dataframe(vendor_stats)
                st.bar_chart(df['vendor_id'].value_counts().sort_index())
        
        # Map
        if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude']):
            st.subheader("🗺️ Pickup Locations")
            map_data = df[['pickup_latitude', 'pickup_longitude']].rename(
                columns={'pickup_latitude': 'lat', 'pickup_longitude': 'lon'}
            )
            map_data = map_data[
                (map_data['lat'] > 40.5) & (map_data['lat'] < 41) &
                (map_data['lon'] > -74.3) & (map_data['lon'] < -73.7)
            ]
            if not map_data.empty:
                st.map(map_data)
        
        # Latest records
        st.subheader("📋 Latest Trips (with Predictions)")
        display_cols = ['id', 'pickup_datetime', 'passenger_count', 'distance_km', 
                       'trip_duration', 'predicted_duration', 'prediction_error']
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols].tail(10), width="stretch")
        
    else:
        st.info("⏳ Waiting for Speed Layer data...")
        st.markdown("""
        Start the streaming pipeline:
        1. Run `python kafka_consumer.py` or `python spark_streaming.py`
        2. Run `python kafka_producer.py` in another terminal
        """)

def display_batch_layer():
    """Display Batch Layer (historical) data."""
    st.header("📦 Batch Layer - Historical Analysis")
    
    batch_data = load_batch_layer_data()
    
    if batch_data:
        # Daily Summary
        if 'daily_summary' in batch_data:
            st.subheader("📅 Daily Summary")
            daily = batch_data['daily_summary']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trips", f"{int(daily['total_trips'].iloc[0]):,}")
            with col2:
                st.metric("Avg Duration", f"{daily['avg_duration'].iloc[0]/60:.1f} min")
            with col3:
                st.metric("Avg Distance", f"{daily['avg_distance'].iloc[0]:.2f} km")
            with col4:
                st.metric("Avg Passengers", f"{daily['avg_passengers'].iloc[0]:.1f}")
        
        st.markdown("---")
        
        # Hourly Stats
        if 'hourly_stats' in batch_data:
            st.subheader("⏰ Hourly Statistics (Batch View)")
            hourly = batch_data['hourly_stats']
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("**Trip Count by Hour**")
                st.bar_chart(hourly.set_index('hour')['trip_count'])
            
            with chart_col2:
                st.markdown("**Average Duration by Hour**")
                st.line_chart(hourly.set_index('hour')['avg_duration'])
            
            st.dataframe(hourly, width="stretch")
        
        # Vendor Stats
        if 'vendor_stats' in batch_data:
            st.subheader("🏢 Vendor Statistics")
            vendor = batch_data['vendor_stats']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Trips per Vendor**")
                st.bar_chart(vendor.set_index('vendor_id')['trip_count'])
            with col2:
                st.dataframe(vendor, width="stretch")
    
    else:
        st.info("⏳ Waiting for Batch Layer data...")
        st.markdown("""
        Run the batch processing pipeline:
        1. Trigger the `taxi_batch_processing` DAG in Airflow
        2. Or run: `python batch_processor.py`
        """)

def display_combined_view():
    """Display combined Speed + Batch Layer view."""
    st.header("🔄 Combined Lambda Architecture View")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Speed Layer (Real-time)")
        speed_df = load_speed_layer_data()
        if speed_df is not None:
            st.metric("Records", len(speed_df))
            if 'distance_km' in speed_df.columns:
                st.metric("Avg Distance (RT)", f"{speed_df['distance_km'].mean():.2f} km")
            if 'predicted_duration' in speed_df.columns:
                st.metric("Avg Predicted Duration", f"{speed_df['predicted_duration'].mean():.0f} sec")
        else:
            st.warning("No speed layer data")
    
    with col2:
        st.subheader("📦 Batch Layer (Historical)")
        batch_data = load_batch_layer_data()
        if 'daily_summary' in batch_data:
            daily = batch_data['daily_summary']
            st.metric("Total Records", f"{int(daily['total_trips'].iloc[0]):,}")
            st.metric("Avg Distance (Batch)", f"{daily['avg_distance'].iloc[0]:.2f} km")
            st.metric("Avg Duration (Batch)", f"{daily['avg_duration'].iloc[0]:.0f} sec")
        else:
            st.warning("No batch layer data")
    
    st.markdown("---")
    
    # Comparison if both available
    if speed_df is not None and 'daily_summary' in batch_data:
        st.subheader("📊 Speed vs Batch Comparison")
        
        comparison_df = pd.DataFrame({
            'Metric': ['Avg Distance (km)', 'Avg Duration (sec)'],
            'Speed Layer': [
                speed_df['distance_km'].mean() if 'distance_km' in speed_df.columns else 0,
                speed_df['trip_duration'].mean() if 'trip_duration' in speed_df.columns else 0
            ],
            'Batch Layer': [
                batch_data['daily_summary']['avg_distance'].iloc[0],
                batch_data['daily_summary']['avg_duration'].iloc[0]
            ]
        })
        
        st.dataframe(comparison_df, width="stretch")

# Main content based on selection
placeholder = st.empty()

with placeholder.container():
    if data_source == "Speed Layer (Real-time)":
        display_speed_layer()
    elif data_source == "Batch Layer (Historical)":
        display_batch_layer()
    else:
        display_combined_view()

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
