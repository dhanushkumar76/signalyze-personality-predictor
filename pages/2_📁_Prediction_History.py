import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Configuration
LOG_CSV = "logs/prediction_log.csv"
# FIX: Updated TRAIT_NAMES to match the new 5-trait model
TRAIT_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
# FIX: Updated NUM_TRAITS
NUM_TRAITS = 5
CLASS_MAP = {0: "Disagree", 1: "Neutral", 2: "Agree"}
CLASS_COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}

st.set_page_config(page_title="📊 Prediction Analytics", layout="wide")
st.title("📊 Prediction History & Analytics")
st.markdown("Comprehensive analysis of all signature personality predictions")

# Check if log file exists
if not os.path.exists(LOG_CSV):
    st.warning("📝 No prediction history found yet. Run a prediction from the Home page first.")
    st.info("💡 **Tip**: Upload a signature on the main page to start building your prediction history!")
    st.stop()

@st.cache_data
def load_prediction_data():
    """Load and preprocess prediction data"""
    try:
        df = pd.read_csv(LOG_CSV)
        if df.empty:
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(by='timestamp', ascending=False)
        
        # FIX: The new log format stores predictions directly, without parentheses
        prediction_cols = []
        for trait in TRAIT_NAMES:
            if trait in df.columns:
                df[f'{trait}_prediction'] = df[trait].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else None)
                df[f'{trait}_confidence'] = df[trait].str.extract(r'\((\d+\.\d+)\)')[0].astype(float)
            prediction_cols.extend([f'{trait}_prediction', f'{trait}_confidence'])
        
        return df
    except Exception as e:
        st.error(f"Error loading prediction data: {e}")
        return None

# Load data
try:
    df = load_prediction_data()
except Exception as e:
    st.error(f"Error loading prediction data: {e}")
    st.stop()

if df is None or 'timestamp' not in df.columns:
    st.info("ℹ️ No predictions logged yet or timestamp missing.")
    st.stop()

# Sidebar controls
st.sidebar.header("🎛️ Analytics Controls")

# Date range filter
date_range = st.sidebar.date_input(
    "📅 Select Date Range",
    value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
    min_value=df['timestamp'].min().date(),
    max_value=df['timestamp'].max().date()
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]
else:
    filtered_df = df

analysis_type = st.sidebar.selectbox(
    "📊 Analysis Type",
    ["Overview", "Trait Analysis", "Temporal Trends", "Detailed Records"]
)

# Main content based on selection
if analysis_type == "Overview":
    st.header("📋 Overview Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Predictions", len(filtered_df))
    
    with col2:
        if not filtered_df.empty:
            avg_confidence = np.mean([filtered_df[f'{trait}_confidence'].mean() 
                                    for trait in TRAIT_NAMES 
                                    if f'{trait}_confidence' in filtered_df.columns])
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    if not filtered_df.empty:
        st.subheader("🥧 Prediction Distribution")
        
        all_predictions = []
        for trait in TRAIT_NAMES:
            if f'{trait}_prediction' in filtered_df.columns:
                all_predictions.extend(filtered_df[f'{trait}_prediction'].dropna().tolist())
        
        if all_predictions:
            pred_counts = pd.Series(all_predictions).value_counts()
            fig = px.pie(values=pred_counts.values, names=pred_counts.index, 
                         title="Distribution of All Personality Predictions")
            st.plotly_chart(fig, use_container_width=True)
    
    METRICS_CSV = "model/metrics_report.csv"
    if os.path.exists(METRICS_CSV):
        st.subheader("📊 Trait-wise Evaluation Metrics")
        metrics_df = pd.read_csv(METRICS_CSV)
        st.dataframe(metrics_df, use_container_width=True)
        
        fig = go.Figure()
        for metric in ['F1 Score', 'Precision', 'Recall', 'Accuracy']:
            fig.add_trace(go.Bar(x=metrics_df['Trait'], y=metrics_df[metric], name=metric))
        fig.update_layout(barmode='group', title="Trait-wise Evaluation Metrics", xaxis_title="Trait", yaxis_title="Score", legend_title="Metric", xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Trait Analysis":
    st.header("🧠 Individual Trait Analysis")
    
    selected_trait = st.selectbox("Select Trait for Analysis", TRAIT_NAMES)
    
    if f'{selected_trait}_prediction' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            pred_col = f'{selected_trait}_prediction'
            pred_counts = filtered_df[pred_col].value_counts()
            
            fig = px.bar(x=pred_counts.index, y=pred_counts.values,
                         title=f'{selected_trait} Prediction Distribution',
                         labels={'x': 'Prediction', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            conf_col = f'{selected_trait}_confidence'
            fig = px.histogram(filtered_df, x=conf_col, nbins=20,
                             title=f'{selected_trait} Confidence Distribution',
                             labels={conf_col: 'Confidence Score'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"📊 {selected_trait} Statistics")
        
        conf_stats = filtered_df[conf_col].describe()
        most_common = filtered_df[pred_col].mode().iloc[0] if not filtered_df[pred_col].empty else "N/A"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Most Common Prediction", most_common)
        with col2:
            st.metric("Average Confidence", f"{conf_stats['mean']:.3f}")
        with col3:
            st.metric("Confidence Std Dev", f"{conf_stats['std']:.3f}")

elif analysis_type == "Temporal Trends":
    st.header("📈 Temporal Analysis")
    
    if not filtered_df.empty:
        st.subheader("📅 Predictions Over Time")
        
        daily_counts = filtered_df.groupby(filtered_df['timestamp'].dt.date).size()
        
        fig = px.line(x=daily_counts.index, y=daily_counts.values,
                      title="Daily Prediction Volume",
                      labels={'x': 'Date', 'y': 'Number of Predictions'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🕐 Usage Patterns by Hour")
        hourly_counts = filtered_df.groupby(filtered_df['timestamp'].dt.hour).size()
        
        fig = px.bar(x=hourly_counts.index, y=hourly_counts.values,
                     title="Prediction Activity by Hour of Day",
                     labels={'x': 'Hour', 'y': 'Predictions'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Confidence Trends")
        
        confidence_cols = [f'{trait}_confidence' for trait in TRAIT_NAMES if f'{trait}_confidence' in filtered_df.columns]
        if confidence_cols:
            daily_conf = filtered_df.groupby(filtered_df['timestamp'].dt.date)[confidence_cols].mean()
            daily_conf['avg_confidence'] = daily_conf.mean(axis=1)
            
            fig = px.line(x=daily_conf.index, y=daily_conf['avg_confidence'],
                          title="Average Daily Confidence Scores",
                          labels={'x': 'Date', 'y': 'Average Confidence'})
            st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Detailed Records":
    st.header("📋 Detailed Prediction Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_filename = st.text_input("🔍 Search by filename", "")
    
    with col2:
        trait_filter = st.selectbox("Filter by trait prediction", 
                                    ["All"] + [f"{trait} - {CLASS_MAP[0]}" for trait in TRAIT_NAMES] + 
                                    [f"{trait} - {CLASS_MAP[1]}" for trait in TRAIT_NAMES] +
                                    [f"{trait} - {CLASS_MAP[2]}" for trait in TRAIT_NAMES])
    
    display_df = filtered_df.copy()
    
    if search_filename:
        display_df = display_df[display_df['filename'].str.contains(search_filename, case=False, na=False)]
    
    display_cols = ['timestamp', 'filename']
    for trait in TRAIT_NAMES:
        if trait in filtered_df.columns:
            display_cols.append(trait)
    
    if not display_df.empty:
        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            height=400
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Filtered Data (CSV)",
                csv,
                f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        
        with col2:
            json_data = display_df.to_json(orient='records', date_format='iso')
            st.download_button(
                "📥 Download as JSON",
                json_data,
                f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    else:
        st.info("No records match the current filters.")

st.markdown("---")
if not filtered_df.empty:
    quality_metrics = []
    
    prediction_cols = [f'{trait}_prediction' for trait in TRAIT_NAMES]
    available_predictions = sum(1 for col in prediction_cols if col in filtered_df.columns)
    
    quality_metrics.append(f"📊 {available_predictions}/{len(TRAIT_NAMES)} personality traits tracked")
    
    st.info(" | ".join(quality_metrics))