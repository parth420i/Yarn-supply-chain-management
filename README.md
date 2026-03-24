# Yarn Supply Chain Delay Prediction & Cost Analysis

This tool analyzes a synthetic yarn supply chain dataset to:
1. Predict the probability and expected duration of shipment delays.
2. Determine if it makes financial sense to expedite "at risk" shipments.
3. Quantify baseline vs. optimized costs and potential savings.

## Features
- Modular code architecture.
- Model Training (`RandomForestClassifier` for Delay Prediction, `RandomForestRegressor` for Delay Days).
- Cost Optimization Logic to offset stockout costs versus expedite surcharges.
- Output visualizations exported directly to an `/output` folder without blocking runtime.
- Proper Train/Test segregation using `scikit-learn`.

## Setup
1. Use Python 3.8+
2. Install pip dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script using the CSV path:
```bash
python main.py yarn_supplychain_surat.csv
```

### Options
- `--save-model`: Exports the trained classification and regression Random Forest models inside a local `models/` directory for later inference.
- `--no-plots`: Completely skip generating charts.
