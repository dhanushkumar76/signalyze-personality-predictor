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
LOG_CSV = "logs/prediction_log.csv"  # Fixed path to match logger
TRAIT_NAMES = [
    "Confidence", "Emotional Stability", "Sociability", "Responsiveness",
    "Concentration", "Introversion", "Creativity", "Decision-Making"
]

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
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp', ascending=False)
        
        # Extract prediction results from trait columns
        prediction_cols = []
        for trait in TRAIT_NAMES:
            if trait in df.columns:
                # Extract just the prediction part (before the parentheses)
                df[f'{trait}_prediction'] = df[trait].str.extract(r'(\w+)')[0]
                df[f'{trait}_confidence'] = df[trait].str.extract(r'\((\d+\.\d+)\)')[0].astype(float)
                prediction_cols.extend([f'{trait}_prediction', f'{trait}_confidence'])
        
        return df
    except Exception as e:
        st.error(f"Error loading prediction data: {e}")
        return None

# Load data
df = load_prediction_data()

if df is None:
    st.info("ℹ️ No predictions logged yet.")
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

# Filter data by date range
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]
else:
    filtered_df = df

# Analytics type selection
analysis_type = st.sidebar.selectbox(
    "📊 Analysis Type",
    ["Overview", "Trait Analysis", "Temporal Trends", "Visual Traits Analysis", "Detailed Records"]
)

# Main content based on selection
if analysis_type == "Overview":
    st.header("📋 Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col3:
        if not filtered_df.empty:
            time_span = (filtered_df['timestamp'].max() - filtered_df['timestamp'].min()).days
            st.metric("Days Tracked", time_span)
        else:
            st.metric("Days Tracked", "0")
    
    with col4:
        if not filtered_df.empty:
            unique_files = filtered_df['filename'].nunique()
            st.metric("Unique Signatures", unique_files)
        else:
            st.metric("Unique Signatures", "0")
    
    # Prediction distribution pie chart
    if not filtered_df.empty:
        st.subheader("🥧 Prediction Distribution")
        
        # Aggregate all predictions
        all_predictions = []
        for trait in TRAIT_NAMES:
            if f'{trait}_prediction' in filtered_df.columns:
                all_predictions.extend(filtered_df[f'{trait}_prediction'].dropna().tolist())
        
        if all_predictions:
            pred_counts = pd.Series(all_predictions).value_counts()
            fig = px.pie(values=pred_counts.values, names=pred_counts.index, 
                        title="Distribution of All Personality Predictions")
            st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Trait Analysis":
    st.header("🧠 Individual Trait Analysis")
    
    # Trait selection
    selected_trait = st.selectbox("Select Trait for Analysis", TRAIT_NAMES)
    
    if f'{selected_trait}_prediction' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution for selected trait
            pred_col = f'{selected_trait}_prediction'
            pred_counts = filtered_df[pred_col].value_counts()
            
            fig = px.bar(x=pred_counts.index, y=pred_counts.values,
                        title=f'{selected_trait} Prediction Distribution',
                        labels={'x': 'Prediction', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            conf_col = f'{selected_trait}_confidence'
            fig = px.histogram(filtered_df, x=conf_col, nbins=20,
                             title=f'{selected_trait} Confidence Distribution',
                             labels={conf_col: 'Confidence Score'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
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
        # Predictions over time
        st.subheader("📅 Predictions Over Time")
        
        # Daily prediction counts
        daily_counts = filtered_df.groupby(filtered_df['timestamp'].dt.date).size()
        
        fig = px.line(x=daily_counts.index, y=daily_counts.values,
                     title="Daily Prediction Volume",
                     labels={'x': 'Date', 'y': 'Number of Predictions'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly pattern
        st.subheader("🕐 Usage Patterns by Hour")
        hourly_counts = filtered_df.groupby(filtered_df['timestamp'].dt.hour).size()
        
        fig = px.bar(x=hourly_counts.index, y=hourly_counts.values,
                    title="Prediction Activity by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Predictions'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence trends
        st.subheader("📊 Confidence Trends")
        
        # Calculate daily average confidence
        confidence_cols = [f'{trait}_confidence' for trait in TRAIT_NAMES if f'{trait}_confidence' in filtered_df.columns]
        if confidence_cols:
            daily_conf = filtered_df.groupby(filtered_df['timestamp'].dt.date)[confidence_cols].mean()
            daily_conf['avg_confidence'] = daily_conf.mean(axis=1)
            
            fig = px.line(x=daily_conf.index, y=daily_conf['avg_confidence'],
                         title="Average Daily Confidence Scores",
                         labels={'x': 'Date', 'y': 'Average Confidence'})
            st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Visual Traits Analysis":
    st.header("🎨 Visual Traits Analysis")
    
    # Visual traits distribution
    visual_traits = ['ink_density', 'aspect_ratio', 'slant_angle']
    
    for trait in visual_traits:
        if trait in filtered_df.columns:
            st.subheader(f"📊 {trait.replace('_', ' ').title()} Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution histogram
                fig = px.histogram(filtered_df, x=trait, nbins=20,
                                 title=f"{trait.replace('_', ' ').title()} Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(filtered_df, y=trait,
                           title=f"{trait.replace('_', ' ').title()} Box Plot")
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            stats = filtered_df[trait].describe()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{stats['mean']:.4f}")
            with col2:
                st.metric("Median", f"{stats['50%']:.4f}")
            with col3:
                st.metric("Std Dev", f"{stats['std']:.4f}")
            with col4:
                st.metric("Range", f"{stats['max'] - stats['min']:.4f}")
    
    # Correlation analysis
    st.subheader("🔗 Visual Traits Correlation")
    
    available_traits = [trait for trait in visual_traits if trait in filtered_df.columns]
    if len(available_traits) > 1:
        corr_matrix = filtered_df[available_traits].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Visual Traits Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Detailed Records":
    st.header("� Detailed Prediction Records")
    
    # Search and filter options
    col1, col2 = st.columns(2)
    
    with col1:
        search_filename = st.text_input("🔍 Search by filename", "")
    
    with col2:
        trait_filter = st.selectbox("Filter by trait prediction", 
                                   ["All"] + [f"{trait} - Agree" for trait in TRAIT_NAMES] + 
                                   [f"{trait} - Neutral" for trait in TRAIT_NAMES] +
                                   [f"{trait} - Disagree" for trait in TRAIT_NAMES])
    
    # Apply filters
    display_df = filtered_df.copy()
    
    if search_filename:
        display_df = display_df[display_df['filename'].str.contains(search_filename, case=False, na=False)]
    
    # Format for display
    display_cols = ['timestamp', 'filename', 'ink_density', 'aspect_ratio', 'slant_angle']
    
    # Add formatted trait predictions
    for trait in TRAIT_NAMES:
        if trait in filtered_df.columns:
            display_cols.append(trait)
    
    # Show data
    if not display_df.empty:
        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            height=400
        )
        
        # Download options
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

# Footer with summary info
st.markdown("---")
st.markdown(f"**📊 Dataset Summary**: {len(filtered_df)} predictions from {filtered_df['timestamp'].min().strftime('%Y-%m-%d')} to {filtered_df['timestamp'].max().strftime('%Y-%m-%d')}")

# Data quality indicators
if not filtered_df.empty:
    quality_metrics = []
    
    # Check for missing visual traits
    visual_traits = ['ink_density', 'aspect_ratio', 'slant_angle']
    missing_traits = sum(1 for trait in visual_traits if trait not in filtered_df.columns or filtered_df[trait].isnull().any())
    
    if missing_traits == 0:
        quality_metrics.append("✅ Complete visual traits data")
    else:
        quality_metrics.append(f"⚠️ {missing_traits} visual traits have missing data")
    
    # Check prediction completeness
    prediction_cols = [f'{trait}_prediction' for trait in TRAIT_NAMES]
    available_predictions = sum(1 for col in prediction_cols if col in filtered_df.columns)
    
    quality_metrics.append(f"📊 {available_predictions}/{len(TRAIT_NAMES)} personality traits tracked")
    
    st.info(" | ".join(quality_metrics))
