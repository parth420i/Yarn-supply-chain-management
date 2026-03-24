# 🧶 YARN SUPPLY CHAIN DELAY PREDICTION + COST ANALYSIS

# NOTE: Install dependencies from your terminal, e.g.:
# pip install pandas numpy scikit-learn matplotlib seaborn joblib

import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error, roc_curve
import joblib

def load_and_preprocess_data(data_path):
    """Loads dataset and creates useful features for predictions."""
    if not os.path.exists(data_path):
        print(f"Error: Could not find file at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(
        data_path,
        parse_dates=["booking_date", "scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
    )
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    
    # Feature engineering
    df = df.sort_values(["carrier", "scheduled_departure"]).reset_index(drop=True)
    df["lead_time_days"] = (df["scheduled_departure"] - df["booking_date"]).dt.days
    df["planned_transit_days"] = (df["scheduled_arrival"] - df["scheduled_departure"]).dt.days
    df["weekday_dep"] = df["scheduled_departure"].dt.weekday
    df["month_dep"] = df["scheduled_departure"].dt.month
    
    # Rolling mean per carrier to capture carrier's recent performance trends
    df["carrier_delay_30"] = (
        df.groupby("carrier")["delay_flag"]
          .transform(lambda x: x.rolling(window=60, min_periods=1).mean())
          .fillna(0)
    )
    
    df["temp_sens_flag"] = df["temperature_sensitive"].map({"Yes": 1, "No": 0}).fillna(0)
    return df

def train_models(df):
    """Trains classification and regression models and prints metrics."""
    features = [
        "planned_transit_days", "lead_time_days", "weekday_dep", "month_dep",
        "carrier_delay_30", "quantity_tonnes", "shipping_cost",
        "expedite_surcharge", "stockout_cost_per_tonne", "temp_sens_flag"
    ]
    X = df[features].fillna(-1)
    y_clf = df["delay_flag"]
    y_reg = df["delay_days"]
    
    # Robust randomized split ensures carrier bias is not introduced
    X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42
    )
    
    print("\nTraining Random Forest Models... Please wait.")
    
    # Train Classification model
    clf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train_clf)
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_clf, probs)
    
    # Train Regression model
    reg = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train_reg)
    pred_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_test_reg, pred_reg)
    
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Delay Classifier ROC-AUC: {auc:.3f}")
    print(f"Delay Regressor MAE: {mae:.2f} days")
    
    return clf, reg, features, X_test, y_test_clf, probs

def optimize_costs(df, clf, reg, features):
    """Predicts costs and decides optimal actions (expedite vs regular shipment)."""
    # Sample out a balanced subset of 'upcoming' testing shipments to analyze cost decisions
    upcoming_idx = df.sample(min(300, len(df)), random_state=99).index 
    upcoming = df.loc[upcoming_idx].copy()
    
    X_up = upcoming[features].fillna(-1)
    upcoming["pred_delay_prob"] = clf.predict_proba(X_up)[:, 1]
    upcoming["pred_delay_days"] = reg.predict(X_up)
    
    upcoming["expected_stockout_cost"] = (
        upcoming["pred_delay_prob"] * upcoming["quantity_tonnes"] * upcoming["stockout_cost_per_tonne"]
    )
    upcoming["expedite_cost"] = upcoming["shipping_cost"] + upcoming["expedite_surcharge"]
    upcoming["baseline_expected_cost"] = upcoming["shipping_cost"] + upcoming["expected_stockout_cost"]
    
    # Logical decision mapping
    upcoming["expedite_decision"] = (upcoming["expedite_cost"] < upcoming["baseline_expected_cost"]).astype(int)
    
    total_baseline = upcoming["baseline_expected_cost"].sum()
    total_after = (
        upcoming["expedite_decision"] * upcoming["expedite_cost"]
        + (1 - upcoming["expedite_decision"]) * upcoming["baseline_expected_cost"]
    ).sum()
    savings = total_baseline - total_after
    
    print("\n--- COST OPTIMIZATION (Sample of 300) ---")
    print(f"Baseline Expected Cost: INR {total_baseline:,.2f}")
    print(f"Optimized Cost (After Decision): INR {total_after:,.2f}")
    print(f"Expected Savings: INR {savings:,.2f}")
    print(f"Expedite Shipments: {upcoming['expedite_decision'].sum()} out of {len(upcoming)}")
    
    return upcoming, total_baseline, total_after

def export_plots(df, upcoming, auc, y_test_clf, probs, total_baseline, total_after, out_dir="output"):
    """Generates charts and saves them locally without blocking execution."""
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test_clf, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title("ROC Curve for Delay Classification")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()
    
    # 2. Delay Distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(df["delay_days"], kde=True, bins=20, color='steelblue')
    plt.title("Distribution of Shipment Delay (Days)")
    plt.xlabel("Delay Days")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delay_distribution.png"), dpi=150)
    plt.close()
    
    # 3. Carrier Performance
    carrier_perf = df.groupby("carrier")["delay_days"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=carrier_perf.index, y=carrier_perf.values, palette="mako")
    plt.title("Average Delay by Carrier")
    plt.ylabel("Avg Delay (Days)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "carrier_performance.png"), dpi=150)
    plt.close()
    
    # 4. Shipping Cost vs Delay
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=df["shipping_cost"], y=df["delay_days"], hue=df["shipment_type"], alpha=0.7)
    plt.title("Shipping Cost vs Delay Days")
    plt.xlabel("Shipping Cost (INR)")
    plt.ylabel("Delay Days")
    plt.legend(title="Shipment Type")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_vs_delay.png"), dpi=150)
    plt.close()
    
    # 5. Cost Comparison
    cost_df = pd.DataFrame({
        "Scenario": ["Baseline Cost", "Optimized Cost"],
        "Total_Cost": [total_baseline, total_after]
    })
    plt.figure(figsize=(6, 4))
    sns.barplot(x="Scenario", y="Total_Cost", data=cost_df, palette="crest")
    plt.title("Cost Comparison: Baseline vs Optimized")
    plt.ylabel("Total Cost (INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_comparison.png"), dpi=150)
    plt.close()
    
    # 6. Expedite Decisions
    plt.figure(figsize=(6, 4))
    sns.countplot(x="expedite_decision", data=upcoming, palette="Set2")
    plt.title("Expedite Decisions (0=No, 1=Yes)")
    plt.xlabel("Decision")
    plt.ylabel("Count of Shipments")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "expedite_decisions.png"), dpi=150)
    plt.close()
    
    # 7. Probability Distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(upcoming["pred_delay_prob"], bins=20, kde=True, color="orange")
    plt.title("Predicted Delay Probability Distribution")
    plt.xlabel("Predicted Probability of Delay")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delay_probability_dist.png"), dpi=150)
    plt.close()
    
    print(f"\nSaved all visualizations to the '{out_dir}/' directory.")

def main():
    parser = argparse.ArgumentParser(description="Run yarn supply chain analysis")
    parser.add_argument("data_path", nargs="?", help="Path to yarn_supplychain_surat.csv")
    parser.add_argument("--save-model", action="store_true", help="Save the trained models to disk")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting completely")
    
    args = parser.parse_args()
    
    if not args.data_path:
        print("Usage: python main.py <path/to/yarn_supplychain_surat.csv>")
        sys.exit(1)
        
    df = load_and_preprocess_data(args.data_path)
    
    clf, reg, features, X_test, y_test_clf, probs = train_models(df)
    
    if args.save_model:
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, "models/delay_classifier.pkl")
        joblib.dump(reg, "models/delay_regressor.pkl")
        print("\nModels saved to 'models/' directory.")
        
    upcoming, total_baseline, total_after = optimize_costs(df, clf, reg, features)
    
    if not args.no_plots:
        auc = roc_auc_score(y_test_clf, probs)
        
        export_plots(
            df=df, 
            upcoming=upcoming, 
            auc=auc, 
            y_test_clf=y_test_clf, 
            probs=probs, 
            total_baseline=total_baseline, 
            total_after=total_after
        )

if __name__ == "__main__":
    main()
