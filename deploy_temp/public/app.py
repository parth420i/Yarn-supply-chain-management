import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys

# --- Environment Hack for stlite/Pyodide ---
# Plotly (via narwhals) checks for pyarrow.ChunkedArray which may be missing in some Pyodide versions.
# We bypass this by converting data to simple lists before passing to Plotly.
def safe_px_bar(*args, **kwargs):
    # Convert DataFrames to lists if x or y are column names
    if 'data_frame' in kwargs and isinstance(kwargs['data_frame'], pd.DataFrame):
        df = kwargs['data_frame']
        if 'x' in kwargs and isinstance(kwargs['x'], str) and kwargs['x'] in df.columns:
            kwargs['x'] = df[kwargs['x']].tolist()
        if 'y' in kwargs and isinstance(kwargs['y'], str) and kwargs['y'] in df.columns:
            kwargs['y'] = df[kwargs['y']].tolist()
        if 'color' in kwargs and isinstance(kwargs['color'], str) and kwargs['color'] in df.columns:
            kwargs['color'] = df[kwargs['color']].tolist()
        # Remove data_frame to force plotly to use the lists directly
        del kwargs['data_frame']
    elif len(args) > 0 and isinstance(args[0], pd.DataFrame):
        df = args[0]
        new_kwargs = kwargs.copy()
        if 'x' in kwargs and isinstance(kwargs['x'], str) and kwargs['x'] in df.columns:
            new_kwargs['x'] = df[kwargs['x']].tolist()
        if 'y' in kwargs and isinstance(kwargs['y'], str) and kwargs['y'] in df.columns:
            new_kwargs['y'] = df[kwargs['y']].tolist()
        if 'color' in kwargs and isinstance(kwargs['color'], str) and kwargs['color'] in df.columns:
            new_kwargs['color'] = df[kwargs['color']].tolist()
        return px.bar(*args[1:], **new_kwargs)
    return px.bar(*args, **kwargs)

try:
    import pyarrow as pa
    if not hasattr(pa, 'ChunkedArray'):
        pa.ChunkedArray = type('ChunkedArray', (), {})
    if not hasattr(pa, 'Table'):
        pa.Table = type('Table', (), {})
except ImportError:
    from types import ModuleType
    pa = ModuleType('pyarrow')
    pa.ChunkedArray = type('ChunkedArray', (), {})
    pa.Table = type('Table', (), {})
    sys.modules['pyarrow'] = pa

from main import load_and_prepare_data, train_model, evaluate_cost_savings

# --- General System Setup ---
st.set_page_config(page_title="Supply Chain Risk & Cost Analyzer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        font-family: Arial, Helvetica, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("Supply Chain Delay Prediction & Cost Optimizer")
st.write("A predictive application explicitly designed for analyzing Yarn Supply Chain datasets. It forecasts delivery delays by racing standard, explainable machine learning models (**Logistic Regression, Decision Trees, Random Forests**) to find the most accurate algorithm. It then utilizes those probabilities to optimize your shipping constraints via an expected-value cost strategy.")

with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload Yarn Shipment Data (CSV)", type="csv", help="Ensure file structure mirrors yarn_supplychain_surat.csv parameters.")

if uploaded_file:
    with st.spinner("Processing supply chain data..."):
        try:
            # 1. Load and prep data
            df = load_and_prepare_data(uploaded_file)
            st.sidebar.success(f"✓ {len(df):,} records successfully configured.")
            data_loaded = True
        except KeyError as e:
            st.error(f"Dataset structure error. Expected a Supply Chain column not found. ({str(e)})")
            data_loaded = False
    
    if data_loaded:
        if st.button("Run Predictive Model", type="primary"):
            with st.spinner("Evaluating machine learning models across logistics constraints..."):
                
                # 2. Race Models explicitly built for this dataset (LR, DT, RF)
                clf, features, roc_auc, best_model_name = train_model(df)
                
                # 3. Process Expediting Cost Logic
                upcoming_df, baseline_cost, optimized_cost = evaluate_cost_savings(df, clf, features)
                savings = baseline_cost - optimized_cost
            
            st.markdown("---")
            
            # --- EXPANDED EDA FOR NEW DATA ---
            st.subheader("1. Exploratory Geographic Metrics")
            st.write("Extracting structural delay risks directly from the unique shipment history recorded in this specific file upload:")
            
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                if "carrier" in df.columns:
                    carrier_delay = df.groupby("carrier")["delay_flag"].mean().reset_index()
                    # Using safe lists to avoid Arrow attributes error
                    fig_c = px.bar(
                        x=carrier_delay["carrier"].tolist(), 
                        y=carrier_delay["delay_flag"].tolist(), 
                        title="Actual Failure Rate per Carrier", 
                        color=carrier_delay["carrier"].tolist(),
                        labels={"x": "Carrier Network", "y": "Delay Probability", "color": "Carrier"}
                    )
                    fig_c.update_layout(yaxis_title="Delay Probability", xaxis_title="Carrier Network")
                    st.plotly_chart(fig_c, use_container_width=True)
            with g_col2:
                if "source_city" in df.columns and "destination_city" in df.columns:
                    plot_df = df.copy()
                    plot_df["Route"] = plot_df["source_city"] + " → " + plot_df["destination_city"]
                    route_delay = plot_df.groupby("Route")["delay_flag"].mean().reset_index().sort_values("delay_flag", ascending=False).head(5)
                    # Using safe lists
                    fig_r = px.bar(
                        x=route_delay["delay_flag"].tolist(), 
                        y=route_delay["Route"].tolist(), 
                        orientation='h', 
                        title="Top 5 Most High-Risk Corridors", 
                        color=route_delay["delay_flag"].tolist(), 
                        color_continuous_scale="Reds",
                        labels={"x": "Delay Probability", "y": "Route", "color": "Risk Level"}
                    )
                    fig_r.update_layout(xaxis_title="Delay Probability")
                    st.plotly_chart(fig_r, use_container_width=True)
                    
            st.markdown("---")
            
            # --- UI LAYOUT EXPLAINABLE IN AN INTERVIEW ---
            st.subheader("2. Business Impact & Cost Savings")
            st.write("The active algorithm mathematically calculates the *Expected Value* of a delay penalty (`Probability of Delay * Stockout Penalty`). If risking the penalty is projected to be more expensive than paying the explicit expedited freight surcharge, the system triggers an immediate shipment upgrade.")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Baseline Cost", f"INR {baseline_cost:,.0f}")
            col2.metric("Optimized Cost (Using ML)", f"INR {optimized_cost:,.0f}")
            col3.metric("Total Capital Preserved", f"INR {savings:,.0f}", f"{savings/baseline_cost*100:.1f}% Margin Retention")
            
            # Visualize the explicit cost delta 
            cost_data = {
                "Execution Strategy": ["Baseline Freight Execution", "ML Expected-Value Intervention"],
                "Total Logistics Overhead (INR)": [baseline_cost, optimized_cost]
            }
            fig_cost = px.bar(
                x=cost_data["Execution Strategy"], 
                y=cost_data["Total Logistics Overhead (INR)"], 
                color=cost_data["Execution Strategy"], 
                color_discrete_sequence=["#e74c3c", "#2ecc71"], 
                text=cost_data["Total Logistics Overhead (INR)"],
                labels={"x": "Strategy", "y": "logistics Overhead", "color": "Strategy"}
            )
            fig_cost.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(fig_cost, use_container_width=True)
            
            st.markdown("---")
            st.subheader("3. Operational Root Causes")
            st.write(f"Why do these shipments fail? Because our dynamic system selected the transparent **{best_model_name}** architecture, we can visually extract and rank the exact mathematical weights determining shipment delays.")
            
            # Extract Feature Importance (Interpretability Step)
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                imp_df = pd.DataFrame({
                    "Feature Indicator": [f.replace("_", " ").title() for f in features], 
                    "Relative Importance Weight": importances
                }).sort_values(by="Relative Importance Weight", ascending=False).head(10)
                
                fig_imp = px.bar(
                    x=imp_df["Relative Importance Weight"].tolist(), 
                    y=imp_df["Feature Indicator"].tolist(), 
                    orientation='h', 
                    color_discrete_sequence=["#3498db"],
                    labels={"x": "Importance Weight", "y": "Feature"}
                )
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
                
            elif hasattr(clf, 'named_steps'):
                # It's a pipeline (Logistic Regression)
                lr = clf.named_steps.get('logisticregression')
                if lr is not None:
                    coefs = np.abs(lr.coef_[0])
                    imp_df = pd.DataFrame({
                        "Feature Indicator": [f.replace("_", " ").title() for f in features], 
                        "Relative Impact Vector": coefs
                    }).sort_values(by="Relative Impact Vector", ascending=False).head(10)
                    
                    fig_imp = px.bar(
                        x=imp_df["Relative Impact Vector"].tolist(), 
                        y=imp_df["Feature Indicator"].tolist(), 
                        orientation='h', 
                        color_discrete_sequence=["#f39c12"], 
                        title="Logistic Regression Intercept Vectors",
                        labels={"x": "Impact Vector", "y": "Feature"}
                    )
                    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown("---")
            st.subheader("4. Model Benchmarking Diagnostics")
            st.write(f"**Winning Algorithm Selected:** `{best_model_name}`")
            st.write(f"**ROC-AUC Accuracy Benchmark:** `{roc_auc:.3f}`")
            st.write("We evaluate the backend pipelines utilizing `ROC-AUC` rather than gross `Accuracy` because physical supply chain delay data is naturally heavily imbalanced towards successful arrivals. A score bounding closer to `1.0` means the algorithm is highly effective at distinguishing a delayed truck from an on-time truck intrinsically without failing or over-predicting the majority class.")

else:
    st.info("Upload your Yarn Supply Chain datasets via the sidebar tool to begin constraint processing.")
