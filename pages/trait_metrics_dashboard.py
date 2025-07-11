import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Trait Metrics Dashboard", layout="centered")
st.title("📊 Trait-wise Model Evaluation Metrics")

METRICS_CSV = "model/metrics_report.csv"

if not os.path.exists(METRICS_CSV):
    st.warning("No metrics report found. Please train the model first.")
    st.stop()

metrics_df = pd.read_csv(METRICS_CSV)

st.dataframe(metrics_df, use_container_width=True)

# Bar plot for all metrics
fig = go.Figure()
bar_width = 0.2
x = metrics_df['Trait']

for metric in ['F1 Score', 'Precision', 'Recall', 'Accuracy']:
    fig.add_trace(go.Bar(
        x=x,
        y=metrics_df[metric],
        name=metric
    ))

fig.update_layout(
    barmode='group',
    title="Trait-wise Evaluation Metrics",
    xaxis_title="Trait",
    yaxis_title="Score",
    legend_title="Metric",
    xaxis_tickangle=-45,
    height=500
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("This dashboard shows F1, Precision, Recall, and Accuracy for each trait as evaluated on the validation set after training.")

# No explicit TRAIT_NAMES or NUM_TRAITS, but ensure dashboard expects only the four traits in metrics_report.csv
# If any hardcoded references to 8 traits exist, update to 4.
