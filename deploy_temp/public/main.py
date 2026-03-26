import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path_or_buffer):
    if hasattr(data_path_or_buffer, 'getvalue'):
        data = io.BytesIO(data_path_or_buffer.getvalue())
    else:
        data = data_path_or_buffer
        if hasattr(data, 'seek'):
            data.seek(0)
            
    df = pd.read_csv(data)
    df.columns = [str(c).strip() for c in df.columns]
    
    date_cols = [
        'actual_arrival', 'actual_departure',
        'booking_date', 'scheduled_arrival', 'scheduled_departure'
    ]
    
    existing_date_cols = [col for col in date_cols if col in df.columns]
    for col in existing_date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    if "scheduled_departure" in df.columns and "booking_date" in df.columns:
        df["lead_time_days"] = (df["scheduled_departure"] - df["booking_date"]).dt.days
    else:
        df["lead_time_days"] = 0
        
    if "scheduled_arrival" in df.columns and "scheduled_departure" in df.columns:
        df["planned_transit_days"] = (df["scheduled_arrival"] - df["scheduled_departure"]).dt.days
    else:
        df["planned_transit_days"] = 0
        
    if "scheduled_departure" in df.columns:
        df["weekday_dep"] = df["scheduled_departure"].dt.weekday
    else:
        df["weekday_dep"] = 0
    
    if "carrier" in df.columns and "delay_flag" in df.columns:
        df = df.sort_values(["carrier", "scheduled_departure"] if "scheduled_departure" in df.columns else ["carrier"]).reset_index(drop=True)
        # Shift(1) prevents algorithmic target leakage by blinding the model to the current row's label
        df["carrier_delay_rate"] = df.groupby("carrier")["delay_flag"].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
    else:
        df["carrier_delay_rate"] = 0
    
    if "temperature_sensitive" in df.columns:
        # Dynamically map multiple variations of "Yes" and "No" for seamless integration with entirely new datasets
        mapping_dict = {"Yes": 1, "No": 0, "True": 1, "False": 0, "Y": 1, "N": 0, "1": 1, "0": 0, True: 1, False: 0, 1: 1, 0: 0}
        
        # Convert values to strings safely for comparison, strip whitespace, and title case
        def robust_map(val):
            if type(val) in [bool, int, float]: return val
            clean_val = str(val).strip().title()
            return mapping_dict.get(clean_val, 0)
            
        df["temp_sens_flag"] = df["temperature_sensitive"].apply(robust_map)
    else:
        df["temp_sens_flag"] = 0
        
    # Mathematical encoding of powerful textual features (Geographic locations, Carriers)
    cat_columns = ["carrier", "source_city", "destination_city", "material_type", "shipment_type"]
    existing_cats = [c for c in cat_columns if c in df.columns]
    if existing_cats:
        # Preserve original categorical text for Streamlit UI Graphing
        for c in existing_cats:
            df[c + "_raw_str"] = df[c]
            
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True)
        
        # Restore the raw text columns natively
        for c in existing_cats:
            df.rename(columns={c + "_raw_str": c}, inplace=True)
    
    return df

def train_model(df):
    expected_features = [
        "planned_transit_days", "lead_time_days", "weekday_dep", 
        "carrier_delay_rate", "quantity_tonnes", "shipping_cost",
        "expedite_surcharge", "stockout_cost_per_tonne", "temp_sens_flag"
    ]
    
    # Dynamically inject the newly encoded categorical variables (Dummy variables) into the ML algorithms
    for col in df.columns:
        if col.startswith(("carrier_", "source_city_", "destination_city_", "material_type_", "shipment_type_")) and col != "carrier_delay_rate":
            expected_features.append(col)
    
    # DYNAMIC FILTER: Keep only features that actually exist in the current dataset!
    features = [f for f in expected_features if f in df.columns]
    
    if len(features) > 0:
        X = df[features].fillna(df[features].mean()) 
    else:
        # Failsafe if absolutely zero supply chain columns are found
        X = pd.DataFrame(np.zeros((len(df), 1)), columns=["dummy_feature"])
        features = ["dummy_feature"]
        
    target_col = "delay_flag" if "delay_flag" in df.columns else df.columns[-1]
    
    # Safe Target fallback for categorical conversions if user inputs unmapped strings
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].astype('category').cat.codes
        
    y = df[target_col].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hardened hyperparameters to massively prevent tree overfitting 
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    }
    
    best_roc_auc = 0
    best_clf = None
    best_name = ""
    
    # Fallback to pure accuracy if ROC-AUC fails (due to continuous target variables mistakenly imported)
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        try:
            probs = clf.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, probs)
        except Exception:
            score = clf.score(X_test, y_test)
            
        if score >= best_roc_auc:
            best_roc_auc = score
            best_clf = clf
            best_name = name
            
    return best_clf, features, best_roc_auc, best_name

def evaluate_cost_savings(df, clf, features):
    upcoming = df.sample(min(300, len(df)), random_state=42).copy()
    X_upcoming = upcoming[features].fillna(upcoming[features].mean())
    
    try:
        upcoming["delay_probability"] = clf.predict_proba(X_upcoming)[:, 1]
    except Exception:
        upcoming["delay_probability"] = clf.predict(X_upcoming)
    
    # DYNAMIC FALLBACK: Safely assign 0 if financial variables are missing from the dataset upload
    q_col = upcoming["quantity_tonnes"] if "quantity_tonnes" in upcoming.columns else 1.0
    stock_col = upcoming["stockout_cost_per_tonne"] if "stockout_cost_per_tonne" in upcoming.columns else 0.0
    ship_col = upcoming["shipping_cost"] if "shipping_cost" in upcoming.columns else 0.0
    exp_col = upcoming["expedite_surcharge"] if "expedite_surcharge" in upcoming.columns else 0.0
    
    upcoming["expected_stockout_penalty"] = upcoming["delay_probability"] * q_col * stock_col
    upcoming["cost_do_nothing"] = ship_col + upcoming["expected_stockout_penalty"]
    
    upcoming["cost_expedite"] = ship_col + exp_col
    
    upcoming["should_expedite"] = upcoming["cost_expedite"] < upcoming["cost_do_nothing"]
    
    total_baseline_cost = upcoming["cost_do_nothing"].sum()
    
    total_optimized_cost = (
        upcoming["should_expedite"] * upcoming["cost_expedite"] + 
        (~upcoming["should_expedite"]) * upcoming["cost_do_nothing"]
    ).sum()
    
    return upcoming, total_baseline_cost, total_optimized_cost
