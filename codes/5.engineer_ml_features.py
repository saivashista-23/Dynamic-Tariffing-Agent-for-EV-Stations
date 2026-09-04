import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv(r"datasets/processed/urbanev_hourly_v2.csv")

# Ensure datetime and sort chronologically to prevent cross-pollination
df['hour_floor'] = pd.to_datetime(df['hour_floor'])
df = df.sort_values(by=['grid', 'hour_floor']).reset_index(drop=True)

print("Engineering Time-Series Lag & Rolling Features by Grid...")

# Group by grid
g = df.groupby('grid')

# Group 1: charger_util, charging_demand_kwh
for col in ['charger_util', 'charging_demand_kwh']:
    for lag in [1, 2, 3, 12, 24, 168]:
        df[f'{col}_lag_{lag}'] = g[col].shift(lag)
    df[f'{col}_rolling_mean_6h'] = g[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df[f'{col}_rolling_mean_24h'] = g[col].transform(lambda x: x.rolling(24, min_periods=1).mean())

# Group 2: chargingload
for lag in [1, 2, 3, 12, 24]:
    df[f'chargingload_lag_{lag}'] = g['chargingload'].shift(lag)

# Group 3: occupancy_density, congestion_probability, queue_length_proxy
for col in ['occupancy_density', 'congestion_probability', 'queue_length_proxy']:
    for lag in [1, 2, 3, 12, 24]:
        df[f'{col}_lag_{lag}'] = g[col].shift(lag)

# Specific smoothing for congestion extreme anomalies
df['congestion_rolling_mean_6h'] = g['congestion_probability'].transform(lambda x: x.rolling(6, min_periods=1).mean())

print("Reordering columns to place target variables at the end...")
targets = ['congestion_probability', 'charger_util', 'chargingload', 'charging_demand_kwh', 'occupancy_density', 'queue_length_proxy']
features = [c for c in df.columns if c not in targets]
df = df[features + targets]

print("Saving final Machine Learning dataset...")
# Save the final ML-ready dataset
df.to_csv(r"datasets/processed/urbanev_ml_ready.csv", index=False)

print(f"Successfully generated urbanev_ml_ready.csv with shape {df.shape}!")
