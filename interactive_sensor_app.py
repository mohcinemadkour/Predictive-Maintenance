import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Set page config
st.set_page_config(page_title="Interactive Sensor Analytics", layout="wide", page_icon="🧪")

# Title
st.title("🧪 Predictive Maintenance - Interactive Sensor Analytics")

# Load data
@st.cache_data
def load_data():
    """Load the training data"""
    try:
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
    st.sidebar.header("Configuration")
    
    features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 
                's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 
                's17', 's18', 's19', 's20', 's21']
    
    if 'id' in df.columns and 'cycle' in df.columns:
        # Engine Selection
        all_unique_engines = sorted(df['id'].unique())
        selected_engine = st.sidebar.selectbox(
            "Select Engine for Detailed Analytics:",
            options=all_unique_engines,
            index=0
        )
        
        # Get data for selected engine
        engine_df = df[df['id'] == selected_engine].sort_values('cycle')
        last_cycle = engine_df['cycle'].max()
        
        # Cycle selection slider
        selected_cycle = st.sidebar.slider(
            f"Select Cycle for Engine {selected_engine}:",
            min_value=int(engine_df['cycle'].min()),
            max_value=int(last_cycle),
            value=int(last_cycle)
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("Use the slider above to retrieve all sensor points for a specific cycle.")

        # Main Display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"📍 Cycle {selected_cycle} Data")
            
            # Feature extraction for prediction
            cycle_data_full = engine_df[engine_df['cycle'] == selected_cycle]
            cycle_data_features = cycle_data_full[features]
            
            # Display sensor table
            st.table(cycle_data_features.T.rename(columns={cycle_data_features.index[0]: 'Value'}))
            
            # RUL Prediction
            st.markdown("---")
            st.subheader("🔮 ML Prediction")
            if model is not None:
                prediction = model.predict(cycle_data_features)[0]
                
                # Compare with Actual TTF if available
                if 'ttf' in cycle_data_full.columns:
                    actual_ttf = cycle_data_full['ttf'].values[0]
                    st.metric("Predicted RUL", f"{prediction:.1f} cycles", delta=f"{prediction - actual_ttf:.1f} vs Actual")
                    st.write(f"**Actual RUL:** {actual_ttf} cycles")
                else:
                    st.metric("Predicted RUL", f"{prediction:.1f} cycles")
                
                # Warning based on health
                if prediction < 30:
                    st.error("⚠️ CRITICAL: Maintenance required immediately!")
                elif prediction < 75:
                    st.warning("⚡ CAUTION: Schedule maintenance soon.")
                else:
                    st.success("✅ HEALTHY: Engine operating within normal parameters.")
            else:
                st.info("Regression model not found in `Models-regression/`. Run `source_best_regression.py` first.")
            
        with col2:
            st.subheader(f"📈 Sensor Trends - Engine {selected_engine}")
            selected_sensors = st.multiselect(
                "Select sensors to visualize:",
                options=features,
                default=['s2', 's3', 's4', 's7', 's11', 's12']
            )
            
            if selected_sensors:
                for sensor in selected_sensors:
                    fig = go.Figure()
                    
                    # Main line plot
                    fig.add_trace(go.Scatter(
                        x=engine_df['cycle'],
                        y=engine_df[sensor],
                        mode='lines',
                        name=f'{sensor} Trend',
                        line=dict(color='royalblue', width=2)
                    ))
                    
                    # Last cycle marker
                    last_val = engine_df[engine_df['cycle'] == last_cycle][sensor].values[0]
                    fig.add_trace(go.Scatter(
                        x=[last_cycle],
                        y=[last_val],
                        mode='markers',
                        name='Last Cycle',
                        marker=dict(color='red', size=12, symbol='star'),
                        hovertemplate='Cycle: %{x}<br>Value: %{y}<br>Status: End of Life'
                    ))
                    
                    # Selected cycle marker
                    if selected_cycle != last_cycle:
                        sel_val = engine_df[engine_df['cycle'] == selected_cycle][sensor].values[0]
                        fig.add_trace(go.Scatter(
                            x=[selected_cycle],
                            y=[sel_val],
                            mode='markers',
                            name='Selected Point',
                            marker=dict(color='orange', size=10, symbol='circle'),
                            hovertemplate='Cycle: %{x}<br>Value: %{y}'
                        ))
                    
                    fig.update_layout(
                        title=f"Sensor {sensor} Over Time",
                        xaxis_title="Cycle",
                        yaxis_title="Value",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select one or more sensors from the dropdown to see trends.")

    else:
        st.error("The dataset must contain 'id' and 'cycle' columns for analytics.")

st.markdown("---")
st.markdown("**Predictive Maintenance - Interactive Sensor Analytics** | Created with Streamlit and Plotly")
