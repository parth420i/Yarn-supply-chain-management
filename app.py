import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

# --- Configuration & Styling ---
st.set_page_config(page_title="Supply Chain Analytics", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling ---
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("Supply Chain Delay & Cost Analytics")
st.markdown("Upload your shipment data to forecast potential delays, evaluate carrier performance, and calculate stockout cost optimizations.")

# --- Sidebar ---
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload Shipment CSV", type="csv", help="Ensure your files follow the standard supply chain template.")
    
    if uploaded_file is not None:
        st.success("File uploaded perfectly.")

if uploaded_file is not None:
    # --- Loading & Processing ---
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(
            file,
            parse_dates=["booking_date", "scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
        )
        
        # Feature Engineering 
        df = df.sort_values(["carrier", "scheduled_departure"]).reset_index(drop=True)
        df["lead_time_days"] = (df["scheduled_departure"] - df["booking_date"]).dt.days
        df["planned_transit_days"] = (df["scheduled_arrival"] - df["scheduled_departure"]).dt.days
        df["weekday_dep"] = df["scheduled_departure"].dt.weekday
        df["month_dep"] = df["scheduled_departure"].dt.month
        df["carrier_delay_30"] = df.groupby("carrier")["delay_flag"].transform(lambda x: x.rolling(window=60, min_periods=1).mean()).fillna(0)
        df["temp_sens_flag"] = df["temperature_sensitive"].map({"Yes": 1, "No": 0}).fillna(0)
        return df

    with st.spinner("Processing dataset..."):
        df = load_data(uploaded_file)
        
    features = [
        "planned_transit_days", "lead_time_days", "weekday_dep", "month_dep",
        "carrier_delay_30", "quantity_tonnes", "shipping_cost",
        "expedite_surcharge", "stockout_cost_per_tonne", "temp_sens_flag"
    ]
    
    st.divider()

    # --- Training Models ---
    with st.spinner("Analyzing operational trends..."):
        X = df[features].fillna(-1)
        y_clf = df["delay_flag"]
        y_reg = df["delay_days"]
        
        X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = train_test_split(
            X, y_clf, y_reg, test_size=0.2, random_state=42
        )
        
        clf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train_clf)
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test_clf, probs)
        
        reg = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
        reg.fit(X_train, y_train_reg)

    # --- Main Dashboard Setup ---
    
    # 1. Executive Summary metrics
    st.subheader("Executive Summary")
    
    # Analyze an explicit 300 batch of test records so we can simulate real impact
    upcoming_idx = df.sample(min(300, len(df)), random_state=99).index 
    upcoming = df.loc[upcoming_idx].copy()
    X_up = upcoming[features].fillna(-1)
    
    upcoming["pred_delay_prob"] = clf.predict_proba(X_up)[:, 1]
    upcoming["expected_stockout_cost"] = upcoming["pred_delay_prob"] * upcoming["quantity_tonnes"] * upcoming["stockout_cost_per_tonne"]
    upcoming["expedite_cost"] = upcoming["shipping_cost"] + upcoming["expedite_surcharge"]
    upcoming["baseline_expected_cost"] = upcoming["shipping_cost"] + upcoming["expected_stockout_cost"]
    
    # Logical decision definition: ONLY expedite if it is literally cheaper
    upcoming["expedite_decision"] = (upcoming["expedite_cost"] < upcoming["baseline_expected_cost"]).astype(int)
    
    total_baseline = upcoming["baseline_expected_cost"].sum()
    total_after = (upcoming["expedite_decision"] * upcoming["expedite_cost"] + (1 - upcoming["expedite_decision"]) * upcoming["baseline_expected_cost"]).sum()
    savings = total_baseline - total_after
    
    # Create top-level dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shipments Analyzed", f"{len(df):,}")
    col2.metric("Projected Baseline Cost", f"INR {total_baseline:,.0f}")
    col3.metric("Cost After Optimization", f"INR {total_after:,.0f}")
    col4.metric("Potential Capital Savings", f"INR {savings:,.0f}", f"{savings/total_baseline*100:.1f}% reduction")
    
    st.info(f"**Recommendation Engine:** Based on historically analogous data, expediting **{upcoming['expedite_decision'].sum()} out of {len(upcoming)}** flagged shipments will minimize your overall stockout penalties while saving capital.")
    
    st.write("---")
    # Tabs for detailed views
    tab1, tab2, tab3 = st.tabs(["Performance Drivers", "Carrier Analysis", "Financial Breakdown"])
    
    with tab1:
        st.write("Understand which supply chain factors are contributing most to delivery delays.")
        importances = clf.feature_importances_
        
        # Sort for better plotting in a dataframe
        importance_df = pd.DataFrame({
            'Feature': [f.replace("_", " ").title() for f in features],
            'Impact': importances
        }).sort_values('Impact', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='Impact', 
            y='Feature', 
            orientation='h',
            title="Key Drivers of Shipment Delays",
            color='Impact',
            color_continuous_scale="Teal"
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Relative Impact on Delay", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show Technical Model Details"):
            st.write(f"**Classification Accuracy (ROC-AUC):** {auc:.3f}")
            st.write("A model trained on standard parameters evaluated your operational metrics. High scores indicate a high reliability in anticipating delays before they occur.")

    with tab2:
        st.write("Compare average delay duration across your different logistics partners.")
        carrier_perf = df.groupby("carrier")["delay_days"].mean().reset_index().sort_values("delay_days", ascending=False)
        fig = px.bar(
            carrier_perf,
            x='carrier',
            y='delay_days',
            title="Historical Average Delay by Carrier",
            color='delay_days',
            color_continuous_scale="Blues"
        )
        fig.update_layout(xaxis_title="Shipping Carrier Framework", yaxis_title="Average Delay (Days)", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.write("A scenario comparison demonstrating your total expenditures before and after implementing predictive expedites.")
        
        pie_col, bar_col = st.columns([1, 2])
        
        with pie_col:
            # Decisions pie chart
            decisions = upcoming['expedite_decision'].value_counts().reset_index()
            decisions['Decision'] = decisions['expedite_decision'].map({0: 'Standard Shipping', 1: 'Expedite Required'})
            
            fig_pie = px.pie(
                decisions, 
                values='count', 
                names='Decision', 
                title="Expedite Strategy Ratio",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with bar_col:
            # Decisions bar chart projection
            cost_df = pd.DataFrame({
                "Scenario": ["Current Strategy Baseline", "AI Optimized Strategy"],
                "Total Cost (INR)": [total_baseline, total_after]
            })
            fig_bar = px.bar(
                cost_df, 
                x="Scenario", 
                y="Total Cost (INR)", 
                title="Aggregate Cost Projection",
                text="Total Cost (INR)",
                color="Scenario",
                color_discrete_sequence=["#ef553b", "#00cc96"]
            )
            fig_bar.update_traces(texttemplate='INR %{text:,.0f}', textposition='outside')
            fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Net Cost Requirement (INR)")
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    # Landing page placeholder
    st.info("👈 Please use the sidebar to connect your `yarn_supplychain_surat.csv` data-lake.")
    
    st.markdown("""
    ### Why use Predictive Analytics?
    By anticipating network congestion and carrier delays *before* they occur, organizations can make proactive decisions to upgrade shipping services only when it is mathematically cheaper than absorbing the penalty of arriving late.
    
    **Dashboard Features:**
    - **Carrier Diagnostics**: Instantly identify which transportation providers commonly miss their SLA.
    - **Driver Transparency**: Understand exactly what variables cause delays across your network.
    - **Calculated Savings**: See side-by-side what your supply chain expenses look like before and after adopting intelligent expedite recommendations.
    """)
