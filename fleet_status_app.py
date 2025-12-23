import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Set page config
st.set_page_config(page_title="Fleet Maintenance Status", layout="wide", page_icon="🚢")

# Title
st.title("🚢 Predictive Maintenance - Fleet-wide Status Dashboard")

# Load data
@st.cache_data
def load_data():
    """Load the training data"""
    try:
        # Load train data to have actual TTF for verification if possible
        df = pd.read_csv('data/train.csv')
        return df
    except FileNotFoundError:
        st.error("Error: data/train.csv not found. Please ensure the data file exists.")
        return None

# Load model
@st.cache_resource
def load_model():
    """Load the regression model"""
    model_path = 'Models-regression/best_rf_regressor.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Load the data and model
df = load_data()
model = load_model()

if df is not None:
    # Sidebar Configuration
    st.sidebar.header("Fleet Analysis Settings")
    
    features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 
                's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 
                's17', 's18', 's19', 's20', 's21']
    
    # Input Cycle Number
    max_cycle_in_data = df['cycle'].max()
    target_cycle = st.sidebar.number_input(
        "Select Snapshot Cycle:",
        min_value=1,
        max_value=int(max_cycle_in_data),
        value=50,
        step=1
    )
    
    st.sidebar.markdown("---")
    st.sidebar.write("This dashboard provides a snapshot of all active engines at the selected cycle.")

    # Filter fleet at target cycle
    fleet_snapshot = df[df['cycle'] == target_cycle].copy()
    
    if fleet_snapshot.empty:
        st.warning(f"No engines reached cycle {target_cycle}. Try a lower cycle number.")
    else:
        # Batch Prediction
        if model is not None:
            X_snapshot = fleet_snapshot[features]
            fleet_snapshot['predicted_rul'] = model.predict(X_snapshot)
        else:
            st.error("Model not found. Please run `source_best_regression.py` first.")
            fleet_snapshot['predicted_rul'] = np.nan

        # Summary Metrics
        st.header(f"📊 Fleet Health Snapshot at Cycle {target_cycle}")
        
        num_engines = len(fleet_snapshot)
        avg_rul = fleet_snapshot['predicted_rul'].mean()
        
        # Engines failing in next 30 cycles
        failing_engines = fleet_snapshot[fleet_snapshot['predicted_rul'] < 30].copy()
        critical_count = len(failing_engines)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Engines Active", num_engines)
        with col2:
            st.metric("Avg. Predicted RUL", f"{avg_rul:.1f} cycles")
        with col3:
            st.metric("Critical Sensors (RUL < 30)", critical_count, delta=critical_count, delta_color="inverse")

        # Failure Forecast Section
        st.divider()
        st.subheader("🚨 Maintenance Forecast (Next 30 Cycles)")
        
        if critical_count > 0:
            st.warning(f"The following {critical_count} engines are predicted to fail within 30 cycles:")
            
            # Show table of critical engines
            display_cols = ['id', 'cycle', 'predicted_rul']
            if 'ttf' in fleet_snapshot.columns:
                display_cols.append('ttf')
            
            st.dataframe(
                failing_engines[display_cols].sort_values('predicted_rul'),
                use_container_width=True
            )
        else:
            st.success("All active engines are predicted to be healthy for the next 30 cycles.")

        # Visualizations
        st.divider()
        st.subheader("📈 Fleet Health Distribution")
        
        # Histogram of RUL
        fig_dist = px.histogram(
            fleet_snapshot, 
            x="predicted_rul", 
            nbins=30,
            title="Distribution of Predicted Remaining Useful Life",
            labels={'predicted_rul': 'Predicted RUL (Cycles)'},
            color_discrete_sequence=['teal']
        )
        fig_dist.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Critical Limit (30)")
        st.plotly_chart(fig_dist, use_container_width=True)

        # Scatter Plot: Predicted vs Actual (if available)
        if 'ttf' in fleet_snapshot.columns:
            fig_scatter = px.scatter(
                fleet_snapshot,
                x='ttf',
                y='predicted_rul',
                hover_data=['id'],
                title="Actual vs Predicted RUL (Fleet Snapshot)",
                labels={'ttf': 'Actual RUL', 'predicted_rul': 'Predicted RUL'},
                opacity=0.6
            )
            fig_scatter.add_shape(
                type="line", line=dict(dash="dash"),
                x0=fleet_snapshot['ttf'].min(), y0=fleet_snapshot['ttf'].min(),
                x1=fleet_snapshot['ttf'].max(), y1=fleet_snapshot['ttf'].max()
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Detailed Fleet Table
        st.divider()
        st.subheader("📋 Detailed Fleet Data")
        st.dataframe(fleet_snapshot.sort_values('predicted_rul'))

else:
    st.error("Unable to load data.")

st.markdown("---")
st.markdown("**Fleet-wide Maintenance Status** | Powered by Random Forest Regressor")
