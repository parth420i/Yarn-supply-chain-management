import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Yarn Supply Chain Dashboard", page_icon="🧶", layout="wide")

# Custom Title
st.title("🧶 Yarn Supply Chain: Delay Predictor & Cost Optimizer")
st.markdown("Analyze shipment data, predict delays using Random Forest AI, and automatically calculate stockout cost savings.")

uploaded_file = st.file_uploader("Upload 'yarn_supplychain_surat.csv'", type="csv")

if uploaded_file is not None:
    # Load data
    with st.spinner("Processing data..."):
        df = pd.read_csv(
            uploaded_file,
            parse_dates=["booking_date", "scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
        )
        
        # Consistent Feature Engineering
        df = df.sort_values(["carrier", "scheduled_departure"]).reset_index(drop=True)
        df["lead_time_days"] = (df["scheduled_departure"] - df["booking_date"]).dt.days
        df["planned_transit_days"] = (df["scheduled_arrival"] - df["scheduled_departure"]).dt.days
        df["weekday_dep"] = df["scheduled_departure"].dt.weekday
        df["month_dep"] = df["scheduled_departure"].dt.month
        df["carrier_delay_30"] = df.groupby("carrier")["delay_flag"].transform(lambda x: x.rolling(window=60, min_periods=1).mean()).fillna(0)
        df["temp_sens_flag"] = df["temperature_sensitive"].map({"Yes": 1, "No": 0}).fillna(0)
        
        features = [
            "planned_transit_days", "lead_time_days", "weekday_dep", "month_dep",
            "carrier_delay_30", "quantity_tonnes", "shipping_cost",
            "expedite_surcharge", "stockout_cost_per_tonne", "temp_sens_flag"
        ]
        
    st.success(f"Data successfully loaded. Extracted {len(features)} metrics for {len(df)} shipments.")
    
    if st.button("Run Automated AI Analysis", type="primary"):
        with st.spinner("Training Random Forest Subsystems..."):
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
            
        st.subheader("🤖 Model Performance Insights")
        col1, col2 = st.columns(2)
        col1.metric("Classifier Accuracy (ROC-AUC)", f"{auc:.3f}", "Delay Propensity Quality")
        col2.info("A ROC-AUC score closer to 1.0 represents a perfect classification of delayed vs on-time deliveries.")
        
        st.divider()
        st.subheader("📊 Key Supply Chain Analytics")
        tab1, tab2, tab3 = st.tabs(["Carrier Diagnostics", "Why are delays happening?", "Financial Optimization"])
        
        with tab1:
            st.write("Review which shipping partners are historically prone to missing their delivery targets.")
            fig, ax = plt.subplots(figsize=(10, 5))
            carrier_perf = df.groupby("carrier")["delay_days"].mean().sort_values(ascending=False)
            sns.barplot(x=carrier_perf.index, y=carrier_perf.values, hue=carrier_perf.index, palette="mako", legend=False, ax=ax)
            ax.set_title("Average Delay by Carrier (Days)")
            ax.set_ylabel("Days Late")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
        with tab2:
            st.write("The underlying Random Forest engine mathematically extracts what operational factors lead to the failure of on-time delivery constraints.")
            fig, ax = plt.subplots(figsize=(10, 5))
            importances = clf.feature_importances_
            idx = np.argsort(importances)
            ax.barh(range(len(idx)), importances[idx], color='teal', align='center')
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels([features[i] for i in idx])
            ax.set_xlabel("Relative Feature Importance Score")
            ax.set_title("Root Cause Drivers for Delays")
            st.pyplot(fig)
            
        with tab3:
            st.write("### AI-Recommended Expedite Actions")
            st.write("Using a test sample of 300 upcoming shipments, the software calculates if expediting the truck is cheaper than absorbing customer penalty fees for missing the deadline.")
            
            upcoming_idx = df.sample(min(300, len(df)), random_state=99).index 
            upcoming = df.loc[upcoming_idx].copy()
            X_up = upcoming[features].fillna(-1)
            
            upcoming["pred_delay_prob"] = clf.predict_proba(X_up)[:, 1]
            upcoming["expected_stockout_cost"] = upcoming["pred_delay_prob"] * upcoming["quantity_tonnes"] * upcoming["stockout_cost_per_tonne"]
            upcoming["expedite_cost"] = upcoming["shipping_cost"] + upcoming["expedite_surcharge"]
            upcoming["baseline_expected_cost"] = upcoming["shipping_cost"] + upcoming["expected_stockout_cost"]
            
            # Predict Logic
            upcoming["expedite_decision"] = (upcoming["expedite_cost"] < upcoming["baseline_expected_cost"]).astype(int)
            total_baseline = upcoming["baseline_expected_cost"].sum()
            total_after = (upcoming["expedite_decision"] * upcoming["expedite_cost"] + (1 - upcoming["expedite_decision"]) * upcoming["baseline_expected_cost"]).sum()
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Baseline Estimated Cost", f"INR {total_baseline:,.0f}")
            m2.metric("Optimized Action Cost", f"INR {total_after:,.0f}")
            m3.metric("Projected Savings", f"INR {(total_baseline - total_after):,.0f}", f"{(total_baseline-total_after)/total_baseline*100:.1f}% Reduction")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            cost_df = pd.DataFrame({
                "Scenario": ["Standard Baseline", "AI Optimized Strategy"],
                "Total_Cost": [total_baseline, total_after]
            })
            sns.barplot(x="Scenario", y="Total_Cost", data=cost_df, hue="Scenario", palette="crest", legend=False, ax=ax)
            ax.set_ylabel("Total Cost (INR)")
            ax.set_title("Financial Savings from AI Actions")
            st.pyplot(fig)
            
else:
    st.info("Awaiting CSV upload to begin analysis.")
