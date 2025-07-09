import streamlit as st
import pandas as pd
import os

LOG_CSV = "logs/prediction_history.csv"

st.set_page_config(page_title="Prediction History", layout="centered")
st.title("Prediction History")

if not os.path.exists(LOG_CSV):
    st.warning("No prediction history found yet. Run a prediction from the Home page first.")
    st.stop()

try:
    df = pd.read_csv(LOG_CSV)
    if df.empty:
        st.info("ℹNo predictions logged yet.")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp', ascending=False)

        st.dataframe(df, use_container_width=True, height=500)

        with st.expander("📥 Download Log as CSV"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "prediction_history.csv", "text/csv")

except Exception as e:
    st.error(f"Error reading log: {e}")
