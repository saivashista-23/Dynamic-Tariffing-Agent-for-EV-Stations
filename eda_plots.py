import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

BASE_DIR = r"c:\Users\vashi\OneDrive\Desktop\opensoc"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ACN_FILE = os.path.join(BASE_DIR, "acn_hourly.csv")
URBAN_FILE = os.path.join(BASE_DIR, "urbanev_hourly_v2.csv")
INFO_FILE = os.path.join(BASE_DIR, "UrbanEV_ SZ_districts", "information.csv")


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
    out_acn = os.path.join(PLOTS_DIR, "charging_demand_at_us.png")
    out_noncbd = os.path.join(PLOTS_DIR, "non_cbd_demand.png")
    out_cbd = os.path.join(PLOTS_DIR, "cbd_demand.png")
    out_combined = os.path.join(PLOTS_DIR, "longrundemandtrends.png")

    df_acn = pd.read_csv(ACN_FILE)
    df_acn['timestamp'] = pd.to_datetime(df_acn['timestamp'])
    df_acn_overall = df_acn.copy().sort_values('timestamp').set_index('timestamp')
    
    daily_acn_overall = df_acn_overall['total_kwh_this_hour'].resample('D').sum().reset_index()
    daily_acn_overall['avg_demand_per_charger'] = daily_acn_overall['total_kwh_this_hour'] / 54
    daily_acn_overall['rolling_demand_7d'] = daily_acn_overall['avg_demand_per_charger'].rolling(window=7).mean()
    
    plt.figure(figsize=(18, 6)) 
    plt.plot(daily_acn_overall['timestamp'], daily_acn_overall['avg_demand_per_charger'], label='Daily Demand per Charger', color='#1f77b4', alpha=0.4)
    plt.plot(daily_acn_overall['timestamp'], daily_acn_overall['rolling_demand_7d'], label='7-Day Rolling Avg', color='#d62728', linewidth=2.0)
    plt.title('Long-Run Charging Demand per Charger (ACN / US Sites)')
    plt.savefig(out_acn, dpi=300)
    plt.close()

    df_urban = pd.read_csv(URBAN_FILE)
    df_urban['hour_floor'] = pd.to_datetime(df_urban['hour_floor'])
    
    start_urban = pd.to_datetime('2022-06-18')
    end_urban = pd.to_datetime('2022-07-19')
    mask_urban = (df_urban['hour_floor'] >= start_urban) & (df_urban['hour_floor'] <= end_urban)
    df_urban_filtered = df_urban.loc[mask_urban].copy()

    hourly_urban_agg = df_urban_filtered.groupby(['hour_floor', 'CBD']).agg(
        total_demand=('charging_demand_kwh', 'sum'),
        total_piles=('total_piles', 'sum')
    ).reset_index()

    hourly_urban_agg['avg_demand_per_pile'] = hourly_urban_agg['total_demand'] / hourly_urban_agg['total_piles']

    df_non_cbd = hourly_urban_agg[hourly_urban_agg['CBD'] == 0].set_index('hour_floor')
    df_cbd = hourly_urban_agg[hourly_urban_agg['CBD'] == 1].set_index('hour_floor')

    daily_non_cbd = df_non_cbd['avg_demand_per_pile'].resample('D').mean().reset_index()
    daily_cbd = df_cbd['avg_demand_per_pile'].resample('D').mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_non_cbd['hour_floor'], daily_non_cbd['avg_demand_per_pile'], color='#1f77b4', marker='o')
    plt.title('Average Daily Demand per Pile: Non-CBD')
    plt.savefig(out_noncbd, dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_cbd['hour_floor'], daily_cbd['avg_demand_per_pile'], color='#d62728', marker='s')
    plt.title('Average Daily Demand per Pile: CBD')
    plt.savefig(out_cbd, dpi=300)
    plt.close()


def plot_intraday_utilization():
    """
    INSIGHTS:
    1. ACN data models workplace/campus charging behavior. Peak hours differ from public city (CBD) charging.
    2. CBD grids suffer from severe underutilization per pile. Tariff Agent must focus on 'Off-Peak Discounts'.
    3. Non-CBD areas trigger 'surge pricing' during peak morning arrivals.
    4. Learning Agent Focus: CBD prioritizes 'Off-Peak Uplift'. Non-CBD prioritizes 'Congestion Reduction'.
    """
    print("Generating Intraday Utilization Plots...")
    out_acn = os.path.join(PLOTS_DIR, "acn_intraday_utilization.png")
    out_noncbd = os.path.join(PLOTS_DIR, "urbanev_noncbd_intraday_utilization.png")
    out_cbd = os.path.join(PLOTS_DIR, "urbanev_cbd_intraday_utilization.png")

    df_acn = pd.read_csv(ACN_FILE)
    df_acn['charger_util'] = np.where(df_acn['active_sessions'] > 0, df_acn['actively_charging'] / df_acn['active_sessions'], 0)
    acn_agg = df_acn.groupby(['hour', 'is_weekend'])['charger_util'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(acn_agg[acn_agg['is_weekend'] == 0]['hour'], acn_agg[acn_agg['is_weekend'] == 0]['charger_util'], label='Weekday', color='#1f77b4', marker='o')
    plt.plot(acn_agg[acn_agg['is_weekend'] == 1]['hour'], acn_agg[acn_agg['is_weekend'] == 1]['charger_util'], label='Weekend', color='#d62728', marker='s', linestyle='--')
    plt.title('Intraday Charger Utilization Rate (ACN / Workplace)')
    plt.savefig(out_acn, dpi=300)
    plt.close()

    df_urban = pd.read_csv(URBAN_FILE)
    urban_agg = df_urban.groupby(['hour', 'is_weekend', 'CBD'])['charger_util'].mean().reset_index()
    
    noncbd_weekday = urban_agg[(urban_agg['CBD'] == 0) & (urban_agg['is_weekend'] == 0)]
    noncbd_weekend = urban_agg[(urban_agg['CBD'] == 0) & (urban_agg['is_weekend'] == 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(noncbd_weekday['hour'], noncbd_weekday['charger_util'], label='Weekday', color='#1f77b4', marker='o')
    plt.plot(noncbd_weekend['hour'], noncbd_weekend['charger_util'], label='Weekend', color='#d62728', marker='s', linestyle='--')
    plt.title('Intraday Charger Utilization Rate (UrbanEV / Non-CBD)')
    plt.savefig(out_noncbd, dpi=300)
    plt.close()

    cbd_weekday = urban_agg[(urban_agg['CBD'] == 1) & (urban_agg['is_weekend'] == 0)]
    cbd_weekend = urban_agg[(urban_agg['CBD'] == 1) & (urban_agg['is_weekend'] == 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(cbd_weekday['hour'], cbd_weekday['charger_util'], label='Weekday', color='#1f77b4', marker='o')
    plt.plot(cbd_weekend['hour'], cbd_weekend['charger_util'], label='Weekend', color='#d62728', marker='s', linestyle='--')
    plt.title('Intraday Charger Utilization Rate (UrbanEV / CBD)')
    plt.savefig(out_cbd, dpi=300)
    plt.close()


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
    plot_intraday_utilization()
    plot_all_grids()
    plot_grouped_thresholds()
    plot_volatility()
    print("\nSuccessfully generated all EDA plots!")
