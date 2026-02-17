import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
import os
import joblib

# Sidebar for configuration
st.sidebar.title("Fraud Guard Settings")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 5)

# Page Layout
st.set_page_config(page_title="Fraud Guard Dashboard", layout="wide")
st.title("🛡️ Fraud Guard: Real-Time Monitoring & Detection")

# Initialize session state for transaction logs
if 'tx_logs' not in st.session_state:
    st.session_state.tx_logs = pd.DataFrame(columns=[
        'Timestamp', 'User ID', 'Probability', 'Prediction', 'Risk Level'
    ])

# Top Row: Key Metrics
m1, m2, m3, m4 = st.columns(4)

total_tx = len(st.session_state.tx_logs)
fraud_count = len(st.session_state.tx_logs[st.session_state.tx_logs['Prediction'] == 1])
avg_prob = st.session_state.tx_logs['Probability'].mean() if total_tx > 0 else 0

m1.metric("Total Transactions", total_tx)
m2.metric("Fraud Cases Detected", fraud_count, delta=f"{fraud_count/(total_tx if total_tx > 0 else 1)*100:.1f}%", delta_color="inverse")
m3.metric("Avg Fraud Prob", f"{avg_prob:.2f}")
m4.metric("Loss Prevented (Est.)", f"${fraud_count * 50:,.0f}")

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Transaction Feed")
    st.dataframe(st.session_state.tx_logs.tail(10), use_container_width=True)
    
    # Simple Chart: Fraud Prob Over time
    if total_tx > 0:
        fig = px.line(st.session_state.tx_logs, x='Timestamp', y='Probability', title="Transaction Risk Trend")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Model Insights")
    if os.path.exists("shap_summary.png"):
        st.image("shap_summary.png", caption="Global Feature Importance (SHAP)")
    else:
        st.info("SHAP summary plot not found. Run training to generate.")

    st.subheader("Monitoring & Drift")
    if st.button("Run Drift Analysis"):
        with st.spinner("Analyzing data drift..."):
            try:
                from src.models.drift import check_drift
                check_drift()
                st.success("Drift analysis complete!")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if os.path.exists("docs/drift_report.html"):
        st.info("Drift report is ready.")
        # Streamlit doesn't serve local HTML files easily for security reasons
        # But we can provide a button or a message
        st.markdown("**Report generated!** Check `docs/drift_report.html` in the project root.")
    else:
        st.info("No drift report generated yet.")

# Simulation Trigger (Optional)
if st.button("Simulate Next Transaction"):
    try:
        # For demo purposes, we can manually trigger the simulator if imported
        # Or just show how to fetch data
        from src.utils.simulator import TransactionSimulator
        sim = TransactionSimulator(api_url=f"{api_url}/predict")
        res = sim.send_transaction()
        
        if res:
            new_row = {
                'Timestamp': datetime.now().strftime("%H:%M:%S"),
                'User ID': "Simulated",
                'Probability': res['fraud_probability'],
                'Prediction': res['prediction'],
                'Risk Level': res['risk_level']
            }
            st.session_state.tx_logs = pd.concat([st.session_state.tx_logs, pd.DataFrame([new_row])], ignore_index=True)
            st.rerun()
    except Exception as e:
        st.error(f"Error simulating: {e}")

# Footer
st.markdown("---")
st.caption("10 Academy Week 12 Challenge | Senior Fraud Detection System")
