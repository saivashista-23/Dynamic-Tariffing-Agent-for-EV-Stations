import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

print("Loading ML-ready dataset for XGBoost...")
df = pd.read_csv(r"datasets/processed/urbanev_ml_ready.csv")

print("Performing Chronological 80/20 Train-Test Split...")
df['hour_floor'] = pd.to_datetime(df['hour_floor'])
unique_dates = sorted(df['hour_floor'].unique())
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

train_df = df[df['hour_floor'] < split_date].copy()
test_df = df[df['hour_floor'] >= split_date].copy()

# Drop pre-modeling features
drop_cols = ['grid', 'hour_floor', 'time_bucket']
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

targets = [
    'congestion_probability', 
    'charger_util', 
    'chargingload', 
    'charging_demand_kwh', 
    'occupancy_density',
    'queue_length_proxy'
]

os.makedirs('plots/models/xgboost', exist_ok=True)

for target in targets:
    print(f"\n{'='*50}")
    print(f"Training XGBoost Model for: {target}")
    print(f"{'='*50}")
    
    X_train = train_df.drop(columns=targets)
    y_train = train_df[target]
    
    X_test = test_df.drop(columns=targets)
    y_test = test_df[target]
    
    # Train XGBoost
    model = xgb.XGBRegressor(n_estimators=200, random_state=42, tree_method='hist')
    model.fit(X_train, y_train)
    
    # Predict
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate exhaustive metrics
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    
    print(f"Metric -> Train R2 Score: {train_r2:.4f}")
    print(f"Metric -> Test R2 Score: {test_r2:.4f}")
    print(f"Metric -> Test RMSE: {test_rmse:.4f}")
    print(f"Metric -> Test MAE: {test_mae:.4f}")
    
    with open(r'results/scores.txt', 'a') as f:
        f.write(f"\n--- XGBoost: {target} ---\n")
        f.write(f"Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}\n")
        f.write(f"Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}\n")
        
    # Feature Importance
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15, importance_type='gain', title=f"XGBoost Feature Importance: {target}")
    plt.tight_layout()
    plt.savefig(fr"plots/models/xgboost/{target}_importance.png")
    plt.close('all')

print("\nXGBoost training complete! Plots saved to plots/models/xgboost/")
