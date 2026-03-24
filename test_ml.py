import pandas as pd
from main import load_and_prepare_data, train_model

df = load_and_prepare_data("yarn_supplychain_surat.csv")
clf, features, roc_auc, best_name = train_model(df)
print(f"Features used ({len(features)}): {features[:10]}...")
print(f"Best Model: {best_name}")
print(f"ROC-AUC: {roc_auc}")

