import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

print("Loading ML-ready dataset...")
df = pd.read_csv(r"datasets/processed/urbanev_ml_ready.csv")

print("Performing Chronological 80/20 Train-Test Split...")
# We must split based on the global timeline, NOT row index, because rows are grouped by grid
df['hour_floor'] = pd.to_datetime(df['hour_floor'])
unique_dates = sorted(df['hour_floor'].unique())
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

train_df = df[df['hour_floor'] < split_date].copy()
test_df = df[df['hour_floor'] >= split_date].copy()

# Initialize BOTH DataFrames (Test set for submission, Full set for Agent)
predictions_test_df = test_df[['grid', 'hour_floor']].copy()
predictions_all_df = df[['grid', 'hour_floor']].copy()

# Drop pre-modeling features from the whole dataset for predictions
drop_cols = ['grid', 'hour_floor', 'time_bucket']
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)
X_all = df.drop(columns=drop_cols)

targets = [
    'congestion_probability', 
    'charger_util', 
    'chargingload', 
    'charging_demand_kwh', 
    'occupancy_density',
    'queue_length_proxy'
]

# Ensure output directory exists
os.makedirs('plots/models/lightgbm', exist_ok=True)

for target in targets:
    print(f"\n{'='*50}")
    print(f"Training LightGBM Model for: {target}")
    print(f"{'='*50}")
    
    # Isolate X and y
    X_train = train_df.drop(columns=targets)
    y_train = train_df[target]
    
    X_test = test_df.drop(columns=targets)
    y_test = test_df[target]
    
    # Initialize and train LightGBM Regressor
    model = lgb.LGBMRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on training set and unseen test set
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Predict on entire 30-day timeline
    X_all_target = X_all.drop(columns=targets)
    all_preds = model.predict(X_all_target)
    
    # Save Test set predictions for submission (predictions.csv)
    predictions_test_df[f'{target}_actual'] = y_test.values
    predictions_test_df[f'{target}_predicted'] = test_preds
    
    # Save FULL timeline predictions for Pricing Agent (predictions2.csv)
    predictions_all_df[f'{target}_actual'] = df[target].values
    predictions_all_df[f'{target}_predicted'] = all_preds
    
    # Calculate exhaustive metrics for ALL 5 targets
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    
    print(f"Metric -> Train R2 Score: {train_r2:.4f}")
    print(f"Metric -> Test R2 Score: {test_r2:.4f}")
    print(f"Metric -> Test RMSE: {test_rmse:.4f}")
    print(f"Metric -> Test MAE: {test_mae:.4f}")
    
    # Append to scores.txt
    with open(r'results/scores.txt', 'a') as f:
        f.write(f"\n--- LightGBM: {target} ---\n")
        f.write(f"Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}\n")
        f.write(f"Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}\n")
        
    # Generate Feature Importance Plot
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, max_num_features=15, importance_type='gain', title=f"Feature Importance: {target}")
    plt.tight_layout()
    plt.savefig(fr"plots/models/lightgbm/{target}_importance.png")
    plt.close('all')

# Save both DataFrames
predictions_test_df.to_csv(r"results/predictions.csv", index=False)
print(f"\nFinal test set predictions saved to models/predictions.csv")

predictions_all_df.to_csv(r"results/predictions2.csv", index=False)
print(f"Final predictions for all 5 targets (FULL 30-DAY TIMELINE) saved to models/predictions2.csv")

print("\nModel training and exhaustive evaluation complete! Feature importance plots saved to plots/models/")
