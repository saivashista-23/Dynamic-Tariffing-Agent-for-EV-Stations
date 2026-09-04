import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys

# Ensure the 'codes' folder is in sys.path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = r"."
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "eda_demand_trends")
os.makedirs(PLOTS_DIR, exist_ok=True)

ACN_FILE = os.path.join(BASE_DIR, "datasets", "processed", "acn_hourly.csv")
URBAN_FILE = os.path.join(BASE_DIR, "datasets", "processed", "urbanev_hourly_v2.csv")
INFO_FILE = os.path.join(BASE_DIR, "datasets", "raw", "UrbanEV_ SZ_districts", "information.csv")


import longrundemandplots
import plot_intraday_utilization

def plot_long_run_demand():
    """
    INSIGHTS:
    1. Scale Differences: Average demand per pile in CBD areas is surprisingly low (2-4 kWh per day). 
       Non-CBD areas see much higher per-pile utilization (5-10 kWh per day). CBD grids handle more 
       total energy but are heavily over-provisioned.
    2. Volatility: The CBD graph has sharper, more dramatic relative drops (weekends/holidays), 
       whereas non-CBD areas show a slightly more stable oscillation pattern.
    3. Prediction Strategy: Workplace/Residential charging (Non-CBD) exhibits massive, highly 
       predictable patterns based on the workweek. 
    """
    print("Generating Long Run Demand Plots...")
    longrundemandplots.generate_all_plots()

def run_plot_intraday_utilization():
    """
    INSIGHTS:
    1. ACN data models workplace/campus charging behavior. Peak hours differ from public city (CBD) charging.
    2. CBD grids suffer from severe underutilization per pile. Tariff Agent must force charging into off-peak to save grid.
    3. Non-CBD grids are healthier, meaning distinct alpha/delta surge limits must apply per zone.
    """
    print("Generating Intraday Utilization Plots...")
    plot_intraday_utilization.generate_intraday_plots()


def plot_all_grids():
    """
    INSIGHTS (Grid-Specific Surging Thresholds):
    1. Extreme variance exists in structural utilization rates between grids. 
       Some grids peak around 10% utilization, while others exceed 50%.
    2. A static surge threshold (e.g., > 80%) is sub-optimal. It leaves revenue on the table 
       for quiet grids and over-penalizes busy grids. Dynamic thresholds are required.
    """
    print("Generating All Grids 30-Day Trend Plot...")
    df = pd.read_csv(URBAN_FILE)
    df['date'] = pd.to_datetime(df['hour_floor']).dt.date
    daily_grid_util = df.groupby(['date', 'grid'])['charger_util'].mean().reset_index()
    plot_data = daily_grid_util.pivot(index='date', columns='grid', values='charger_util')

    plt.figure(figsize=(16, 8))
    plt.plot(plot_data.index, plot_data.values, color='gray', alpha=0.15, linewidth=1)
    plt.axhline(y=0.8, color='red', linestyle='--', linewidth=2.5, label='Static 80% Threshold')
    plt.title('Daily Average Charger Utilization Across All Grids')
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(PLOTS_DIR, "all_grids_30_day_trend.png"), dpi=300)
    plt.close()


def plot_grouped_thresholds():
    """
    INSIGHTS (Universal Network Load Balancing Thresholds):
    1. To enable true 'Network Balancing' (moving cars between zones using prices), thresholds must be grouped.
    2. CBD Thresholds: Surge > 29.38% (80th Pct), Discount < 8.66% (20th Pct).
    3. Non-CBD Thresholds: Surge > 34.94% (80th Pct), Discount < 9.80% (20th Pct).
    4. A CBD station surging at 30% acts as a financial magnet pushing cars to a Non-CBD station at 30% (normal price).
    """
    print("Generating Grouped Thresholds Plot...")
    df = pd.read_csv(URBAN_FILE)
    df['date'] = pd.to_datetime(df['hour_floor']).dt.date

    cbd_df = df[df['CBD'] == 1]
    non_cbd_df = df[df['CBD'] == 0]

    cbd_p20, cbd_p80 = cbd_df['charger_util'].quantile(0.20), cbd_df['charger_util'].quantile(0.80)
    non_cbd_p20, non_cbd_p80 = non_cbd_df['charger_util'].quantile(0.20), non_cbd_df['charger_util'].quantile(0.80)

    daily_grouped = df.groupby(['date', 'CBD'])['charger_util'].mean().reset_index()
    cbd_daily = daily_grouped[daily_grouped['CBD'] == 1].set_index('date')['charger_util']
    non_cbd_daily = daily_grouped[daily_grouped['CBD'] == 0].set_index('date')['charger_util']

    plt.figure(figsize=(14, 7))
    plt.plot(cbd_daily.index, cbd_daily.values, color='blue', linewidth=2.5, label='CBD Daily Avg')
    plt.plot(non_cbd_daily.index, non_cbd_daily.values, color='green', linewidth=2.5, label='Non-CBD Daily Avg')
    plt.axhline(y=cbd_p80, color='blue', linestyle='--', alpha=0.7)
    plt.axhline(y=cbd_p20, color='blue', linestyle=':', alpha=0.7)
    plt.axhline(y=non_cbd_p80, color='green', linestyle='--', alpha=0.7)
    plt.axhline(y=non_cbd_p20, color='green', linestyle=':', alpha=0.7)
    plt.title('Daily Average Utilization with Surge & Discount Thresholds')
    plt.savefig(os.path.join(PLOTS_DIR, "grouped_thresholds_trend.png"), dpi=300)
    plt.close()


def plot_volatility():
    """
    INSIGHTS (Volatility of Charging Demand):
    1. The Peak Squeeze: Volatility (Standard Deviation) is incredibly high during Peak and Off-Peak hours 
       for Non-CBD areas (Workplaces/Residential). Demand varies wildly day-to-day.
    2. High Coefficient of Variation (CV): The ML Prediction Model will have a much harder time predicting 
       Peak hours than Shoulder hours because the data inherently fluctuates more.
    """
    print("Generating Volatility Plot...")
    df = pd.read_csv(URBAN_FILE)
    
    def map_to_3_buckets(bucket):
        if 'Peak' in bucket: return 'Peak'
        elif bucket == 'Midday': return 'Shoulder'
        else: return 'Off-Peak'
        
    df['broad_bucket'] = df['time_bucket'].apply(map_to_3_buckets)
    volatility_df = df.groupby(['broad_bucket', 'CBD'])['charger_util'].std().reset_index()
    volatility_df.rename(columns={'charger_util': 'volatility_std'}, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=volatility_df, x='broad_bucket', y='volatility_std', hue='CBD', order=['Peak', 'Shoulder', 'Off-Peak'], palette={1: 'blue', 0: 'green'})
    plt.title('Volatility of Charging Demand Across Time Periods')
    plt.savefig(os.path.join(PLOTS_DIR, "volatility_by_bucket.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_long_run_demand()
    run_plot_intraday_utilization()
    plot_all_grids()
    plot_grouped_thresholds()
    plot_volatility()
    print("\nSuccessfully generated all EDA plots!")
