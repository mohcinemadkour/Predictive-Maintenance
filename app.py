import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Set page config
st.set_page_config(page_title="Predictive Maintenance EDA", layout="wide")

# Title
st.title("🔧 Predictive Maintenance - Exploratory Data Analysis")

# Sidebar
st.sidebar.header("Configuration")

# Visualization options
st.sidebar.subheader("Visualization Options")
show_time_series = st.sidebar.checkbox("Show Time Series Plots", value=False)
if show_time_series:
    ts_features_count = st.sidebar.slider("Number of features to plot", min_value=1, max_value=10, value=5)

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

# Load the data
df_tr_lbl = load_data()

if df_tr_lbl is not None:
    # Dataset Overview
    st.header("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df_tr_lbl))
    with col2:
        st.metric("Features", len(df_tr_lbl.columns))
    with col3:
        st.metric("Missing Values", df_tr_lbl.isnull().sum().sum())
    
    # Show first few rows
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data Preview")
        st.dataframe(df_tr_lbl.head(10))
    
    # Feature list
    features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 
                's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 
                's17', 's18', 's19', 's20', 's21']
    
    # Time Series Sidebar Controls
    if show_time_series and 'id' in df_tr_lbl.columns:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Time Series Settings")
        
        unique_engines = df_tr_lbl['id'].unique()
        selected_engines = st.sidebar.multiselect(
            "Select engines to visualize:",
            options=sorted(unique_engines[:20]),  # Limit to first 20 for performance
            default=sorted(unique_engines[:min(3, len(unique_engines))])
        )

        
        ts_features = st.sidebar.multiselect(
            "Select features for time series:",
            features,
            default=features[:min(ts_features_count, len(features))]
        )
    else:
        selected_engines = []
        ts_features = []
    
    # Column Explorer Sidebar Controls
    if 'id' in df_tr_lbl.columns:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Column Explorer Settings")
        
        explore_feature = st.sidebar.selectbox(
            "Select feature to explore:",
            features,
            index=0
        )
        
        num_engines = st.sidebar.number_input(
            "Number of engines to display:",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )
    else:
        explore_feature = None
        num_engines = 10
    
    # Feature Statistics
    st.header("📈 Feature Statistics")
    
    # Standard Deviation Analysis
    st.subheader("Feature Standard Deviation")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    df_tr_lbl[features].std().plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_title("Features Standard Deviation")
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Standard Deviation")
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    
    # Log Standard Deviation
    st.subheader("Feature Standard Deviation (Log Scale)")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    df_tr_lbl[features].std().plot(kind='bar', ax=ax2, logy=True, color='coral')
    ax2.set_title("Features Standard Deviation (log)")
    ax2.set_xlabel("Features")
    ax2.set_ylabel("Standard Deviation (log)")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    
    # Top Variance Features
    st.header("🎯 Feature Analysis")
    
    st.subheader("Features Sorted by Variance")
    features_top_var = df_tr_lbl[features].std().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 10 Features by Variance:**")
        st.dataframe(features_top_var.head(10))
    
    with col2:
        st.write("**Bottom 10 Features by Variance:**")
        st.dataframe(features_top_var.tail(10))
    
    # Correlation with TTF
    if 'ttf' in df_tr_lbl.columns:
        st.subheader("Correlation with Time to Failure (TTF)")
        correlation_with_ttf = df_tr_lbl[features].corrwith(df_tr_lbl.ttf).sort_values(ascending=False)
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        correlation_with_ttf.plot(kind='bar', ax=ax3, color='green')
        ax3.set_title("Feature Correlation with TTF")
        ax3.set_xlabel("Features")
        ax3.set_ylabel("Correlation")
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        st.pyplot(fig3)
        
    
    # Feature Selection Tool
    st.header("🔍 Interactive Feature Explorer")
    
    selected_features = st.multiselect(
        "Select features to visualize:",
        features,
        default=features[:5]
    )
    
    if selected_features:
        # Distribution plots
        st.subheader("Feature Distributions")
        n_cols = 3
        n_rows = (len(selected_features) + n_cols - 1) // n_cols
        
        fig4, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, feature in enumerate(selected_features):
            axes[idx].hist(df_tr_lbl[feature], bins=50, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'Distribution of {feature}')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
        
        # Hide unused subplots
        for idx in range(len(selected_features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig4)
        
        # Statistics table
        st.subheader("Descriptive Statistics")
        st.dataframe(df_tr_lbl[selected_features].describe())
    
    # Correlation Matrix
    st.header("🔗 Feature Correlation Matrix")
    
    corr_features = st.multiselect(
        "Select features for correlation matrix:",
        features,
        default=['s12', 's7', 's21', 's20', 's6', 's14', 's9', 's13', 's8', 's3']
    )
    
    if len(corr_features) > 1:
        fig5, ax5 = plt.subplots(figsize=(12, 10))
        correlation_matrix = df_tr_lbl[corr_features].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax5, 
                    square=True, linewidths=0.5)
        ax5.set_title("Feature Correlation Heatmap")
        st.pyplot(fig5)
    
    # Time Series Plots
    if show_time_series:
        st.header("📉 Time Series Analysis")
        
        # Check if required columns exist
        if 'id' in df_tr_lbl.columns and 'cycle' in df_tr_lbl.columns:
            st.subheader("Sensor Readings Over Time")
            
            if selected_engines and ts_features:
                # Create time series plots
                n_features = len(ts_features)
                fig_ts, axes_ts = plt.subplots(n_features, 1, figsize=(15, n_features * 3))
                
                # Handle single subplot case
                if n_features == 1:
                    axes_ts = [axes_ts]
                
                for idx, feature in enumerate(ts_features):
                    for engine_id in selected_engines:
                        engine_data = df_tr_lbl[df_tr_lbl['id'] == engine_id].sort_values('cycle')
                        axes_ts[idx].plot(engine_data['cycle'], engine_data[feature], 
                                        label=f'Engine {engine_id}', alpha=0.7, linewidth=2)
                    
                    axes_ts[idx].set_xlabel('Cycle')
                    axes_ts[idx].set_ylabel(feature)
                    axes_ts[idx].set_title(f'{feature} Over Time')
                    axes_ts[idx].legend(loc='best')
                    axes_ts[idx].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_ts)
                
                # Show statistics for selected engines
                st.subheader("Selected Engines Statistics")
                selected_data = df_tr_lbl[df_tr_lbl['id'].isin(selected_engines)]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Cycles", len(selected_data))
                with col2:
                    st.metric("Avg Cycles per Engine", 
                             int(selected_data.groupby('id')['cycle'].max().mean()))
                with col3:
                    if 'ttf' in selected_data.columns:
                        st.metric("Avg TTF", f"{selected_data['ttf'].mean():.2f}")
            else:
                st.info("Please select at least one engine and one feature to visualize time series.")
        else:
            st.warning("Time series plotting requires 'id' and 'cycle' columns in the dataset.")
    
    # Column Explorer
    st.header("🔬 Column Explorer")
    
    # Check if required columns exist
    if 'id' in df_tr_lbl.columns and 'cycle' in df_tr_lbl.columns:
        if explore_feature:
            # Get unique engines
            unique_engines = sorted(df_tr_lbl['id'].unique())
            selected_explore_engines = unique_engines[:num_engines]
            
            # Create subplots for each engine
            n_cols = 2
            n_rows = (len(selected_explore_engines) + n_cols - 1) // n_cols
            
            fig_explore, axes_explore = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
            axes_explore = axes_explore.flatten() if n_rows > 1 else axes_explore
            
            for idx, engine_id in enumerate(selected_explore_engines):
                engine_data = df_tr_lbl[df_tr_lbl['id'] == engine_id].sort_values('cycle')
                
                axes_explore[idx].plot(engine_data['cycle'], engine_data[explore_feature], 
                                      color='steelblue', linewidth=2, alpha=0.8)
                axes_explore[idx].set_title(f'Engine {engine_id}', fontsize=10, fontweight='bold')
                axes_explore[idx].set_xlabel('Cycle', fontsize=9)
                axes_explore[idx].set_ylabel(explore_feature, fontsize=9)
                axes_explore[idx].grid(True, alpha=0.3)
                
                # Add ttf information if available
                if 'ttf' in engine_data.columns:
                    max_ttf = engine_data['ttf'].max()
                    axes_explore[idx].text(0.02, 0.98, f'Max TTF: {max_ttf:.0f}', 
                                          transform=axes_explore[idx].transAxes,
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                          fontsize=8)
            
            # Hide unused subplots
            for idx, engine_id in enumerate(selected_explore_engines):
                if idx >= len(axes_explore): break
            # This was a bit messy in previous version, let's just use the clean way
            for i in range(len(selected_explore_engines), len(axes_explore)):
                axes_explore[i].axis('off')
            
            plt.suptitle(f'Feature: {explore_feature} - Evolution Across {len(selected_explore_engines)} Engines', 
                       fontsize=14, fontweight='bold', y=1.00)
            plt.tight_layout()
            st.pyplot(fig_explore)
            
            # Statistics for the explored feature
            st.subheader(f"Statistics for {explore_feature}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df_tr_lbl[explore_feature].mean():.2f}")
            with col2:
                st.metric("Std Dev", f"{df_tr_lbl[explore_feature].std():.2f}")
            with col3:
                st.metric("Min", f"{df_tr_lbl[explore_feature].min():.2f}")
            with col4:
                st.metric("Max", f"{df_tr_lbl[explore_feature].max():.2f}")
    else:
        st.warning("Column explorer requires 'id' and 'cycle' columns in the dataset.")
    
    # Download processed data
    st.header("💾 Export Data")
    
    csv = df_tr_lbl.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="predictive_maintenance_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("**Predictive Maintenance EDA Dashboard** | Built with Streamlit")
