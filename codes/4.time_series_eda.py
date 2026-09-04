import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set styling
plt.style.use('seaborn-v0_8')
os.makedirs('plots/edaforpredictionmodel/eda', exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(r"datasets/processed/urbanev_hourly_v2.csv")

# Ensure hour_floor is datetime
df['hour_floor'] = pd.to_datetime(df['hour_floor'])

# Generalize: Group by hour_floor across all grids
print("Creating Generalized Grid...")
targets = ['congestion_probability', 'charger_util', 'chargingload', 'charging_demand_kwh', 'occupancy_density', 'queue_length_proxy']
df_gen = df.groupby('hour_floor')[targets].mean().reset_index()

# Add time helpers for plots
df_gen['hour'] = df_gen['hour_floor'].dt.hour
df_gen['date'] = df_gen['hour_floor'].dt.date

# Function to run the 6 steps for a given target
def run_eda(target):
    print(f"Running EDA for {target}...")
    
    # 1. Descriptive Statistics
    stats = df_gen[target].describe()
    with open(fr'plots/edaforpredictionmodel/{target}_stats.txt', 'w') as f:
        f.write(f"Descriptive Statistics for {target}:\n")
        f.write(stats.to_string())
        
    # 2. Time Plot
    plt.figure(figsize=(15, 5))
    plt.plot(df_gen['hour_floor'], df_gen[target], color='royalblue', linewidth=1)
    plt.title(f"{target} - 30 Day Time Plot")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(fr'plots/edaforpredictionmodel/{target}_1_timeplot.png')
    plt.close()
    
    # 3. Seasonal Plot (Daily Overlap)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_gen, x='hour', y=target, hue='date', legend=False, palette='viridis', alpha=0.6)
    # Add mean line
    mean_hourly = df_gen.groupby('hour')[target].mean()
    plt.plot(mean_hourly.index, mean_hourly.values, color='red', linewidth=3, label='Mean Trend')
    plt.title(f"{target} - Intraday Seasonal Overlap")
    plt.xlabel("Hour of Day")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fr'plots/edaforpredictionmodel/{target}_2_seasonal.png')
    plt.close()
    
    # 4. Box Plot (Volatility per hour)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_gen, x='hour', y=target, color='lightblue')
    plt.title(f"{target} - Hourly Volatility (Box Plot)")
    plt.xlabel("Hour of Day")
    plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(fr'plots/edaforpredictionmodel/{target}_3_boxplot.png')
    plt.close()
    
    # 5. Decomposition
    # Set index to datetime for statsmodels
    ts = df_gen.set_index('hour_floor')[target]
    decomp = seasonal_decompose(ts, model='additive', period=24) # 24 hours = 1 day cycle
    fig = decomp.plot()
    fig.set_size_inches(15, 8)
    fig.suptitle(f"{target} - Time Series Decomposition", y=1.02)
    plt.tight_layout()
    plt.savefig(fr'plots/edaforpredictionmodel/{target}_4_decomposition.png')
    plt.close()
    
    # 6. Lag Analysis (ACF/PACF)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(ts, lags=72, ax=axes[0], title=f"ACF (Autocorrelation) - {target}")
    plot_pacf(ts, lags=72, ax=axes[1], title=f"PACF (Partial Autocorr) - {target}", method='ywm')
    plt.tight_layout()
    plt.savefig(fr'plots/edaforpredictionmodel/{target}_5_lag_analysis.png')
    plt.close()

for t in targets:
    run_eda(t)

print("EDA completely finished! All plots saved to plots/edaforpredictionmodel/eda/")
