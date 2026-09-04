import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def generate_all_plots():
    base_dir = r"."
    plots_dir = os.path.join(base_dir, "plots", "eda_demand_trends")
    
    # Input files
    acn_file = os.path.join(base_dir, "datasets", "processed", "acn_hourly.csv")
    hourly_file = os.path.join(base_dir, "datasets", "processed", "urbanev_hourly_v2.csv")
    info_file = os.path.join(base_dir, "datasets", "raw", "UrbanEV_ SZ_districts", "information.csv")
    
    # Output files
    out_acn = os.path.join(plots_dir, "charging_demand_at_us.png")
    out_noncbd = os.path.join(plots_dir, "non_cbd_demand.png")
    out_cbd = os.path.join(plots_dir, "cbd_demand.png")
    out_combined = os.path.join(plots_dir, "longrundemandtrends.png")

    try:
        # ==========================================
        # 1. Process ACN Dataset (US Workplaces)
        # ==========================================
        print("Processing ACN dataset...")
        df_acn = pd.read_csv(acn_file)
        df_acn['timestamp'] = pd.to_datetime(df_acn['timestamp'])
        
        # For the overall ACN plot, we use all available data
        df_acn_overall = df_acn.copy().sort_values('timestamp')
        df_acn_overall.set_index('timestamp', inplace=True)
        
        # Calculate daily average per charger (Caltech has 54 chargers)
        daily_acn_overall = df_acn_overall['total_kwh_this_hour'].resample('D').sum().reset_index()
        daily_acn_overall['avg_demand_per_charger'] = daily_acn_overall['total_kwh_this_hour'] / 54
        daily_acn_overall['rolling_demand_7d'] = daily_acn_overall['avg_demand_per_charger'].rolling(window=7).mean()
        
        # Plot 1: ACN Overall Demand
        plt.figure(figsize=(18, 6)) 
        plt.plot(daily_acn_overall['timestamp'], daily_acn_overall['avg_demand_per_charger'], label='Daily Demand per Charger', color='#1f77b4', alpha=0.4)
        plt.plot(daily_acn_overall['timestamp'], daily_acn_overall['rolling_demand_7d'], label='7-Day Rolling Avg', color='#d62728', linewidth=2.0)
        plt.title('Long-Run Charging Demand per Charger (ACN / US Sites)', fontsize=18)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Avg Energy Demand per Charger (kWh)', fontsize=14)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(out_acn, dpi=300)
        plt.close()
        print(f"-> Saved {out_acn}")

        # ==========================================
        # 2. Process UrbanEV Dataset (Shenzhen)
        # ==========================================
        print("Processing UrbanEV dataset...")
        df_urban = pd.read_csv(hourly_file)
        if 'CBD' in df_urban.columns:
            df_urban = df_urban.drop(columns=['CBD'])
        df_info = pd.read_csv(info_file)
        df_urban = pd.merge(df_urban, df_info[['grid', 'CBD']], on='grid', how='left')
        df_urban['hour_floor'] = pd.to_datetime(df_urban['hour_floor'])
        
        # Filter UrbanEV for June 18 - July 19 (2022)
        start_urban = pd.to_datetime('2022-06-18')
        end_urban = pd.to_datetime('2022-07-19')
        mask_urban = (df_urban['hour_floor'] >= start_urban) & (df_urban['hour_floor'] <= end_urban)
        df_urban_filtered = df_urban.loc[mask_urban].copy()

        # Group by hour and CBD to get total demand AND total piles
        hourly_urban_agg = df_urban_filtered.groupby(['hour_floor', 'CBD']).agg(
            total_demand=('charging_demand_kwh', 'sum'),
            total_piles=('total_piles', 'sum')
        ).reset_index()

        hourly_urban_agg['avg_demand_per_pile'] = hourly_urban_agg['total_demand'] / hourly_urban_agg['total_piles']

        df_non_cbd = hourly_urban_agg[hourly_urban_agg['CBD'] == 0].set_index('hour_floor')
        df_cbd = hourly_urban_agg[hourly_urban_agg['CBD'] == 1].set_index('hour_floor')

        daily_non_cbd = df_non_cbd['avg_demand_per_pile'].resample('D').mean().reset_index()
        daily_cbd = df_cbd['avg_demand_per_pile'].resample('D').mean().reset_index()

        # Plot 2: Non-CBD Demand
        plt.figure(figsize=(12, 6))
        plt.plot(daily_non_cbd['hour_floor'], daily_non_cbd['avg_demand_per_pile'], color='#1f77b4', marker='o', linestyle='-', linewidth=2)
        plt.title('Average Daily Demand per Pile: Non-CBD (Workplaces/Residential)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Avg Demand per Pile (kWh)', fontsize=12)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_noncbd, dpi=300)
        plt.close()
        print(f"-> Saved {out_noncbd}")

        # Plot 3: CBD Demand
        plt.figure(figsize=(12, 6))
        plt.plot(daily_cbd['hour_floor'], daily_cbd['avg_demand_per_pile'], color='#d62728', marker='s', linestyle='-', linewidth=2)
        plt.title('Average Daily Demand per Pile: CBD (Central Business District)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Avg Demand per Pile (kWh)', fontsize=12)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_cbd, dpi=300)
        plt.close()
        print(f"-> Saved {out_cbd}")

        # ==========================================
        # 3. Process Combined Plot
        # ==========================================
        print("Generating combined comparison plot...")
        # Align UrbanEV to dummy year (2022)
        daily_non_cbd['plot_date'] = daily_non_cbd['hour_floor'].apply(lambda dt: dt.replace(year=2022))
        daily_cbd['plot_date'] = daily_cbd['hour_floor'].apply(lambda dt: dt.replace(year=2022))

        # Filter ACN for exactly June 18 - July 19 (2018)
        start_acn = pd.to_datetime('2018-06-18')
        end_acn = pd.to_datetime('2018-07-19')
        mask_acn = (df_acn['timestamp'] >= start_acn) & (df_acn['timestamp'] <= end_acn)
        df_acn_filtered = df_acn.loc[mask_acn].copy()
        
        df_acn_filtered.set_index('timestamp', inplace=True)
        daily_acn_combined = df_acn_filtered['total_kwh_this_hour'].resample('D').sum().reset_index()
        daily_acn_combined['avg_demand_per_pile'] = daily_acn_combined['total_kwh_this_hour'] / 54

        # Align ACN to dummy year (2022)
        daily_acn_combined['plot_date'] = daily_acn_combined['timestamp'].apply(lambda dt: dt.replace(year=2022))

        # Plot 4: Combined Demand (Original)
        plt.figure(figsize=(14, 7))
        plt.plot(daily_cbd['plot_date'], daily_cbd['avg_demand_per_pile'], label='UrbanEV CBD (Avg per Pile)', color='#d62728', marker='s', linewidth=2)
        plt.plot(daily_non_cbd['plot_date'], daily_non_cbd['avg_demand_per_pile'], label='UrbanEV Non-CBD (Avg per Pile)', color='#1f77b4', marker='o', linewidth=2)
        plt.plot(daily_acn_combined['plot_date'], daily_acn_combined['avg_demand_per_pile'], label='ACN Workplace (Avg per Charger)', color='#2ca02c', marker='^', linewidth=2, linestyle='--')

        plt.title('Daily Demand Trend Comparison per Pile/Charger (June 18 - July 19)', fontsize=16)
        plt.xlabel('Date (Aligned Month/Day)', fontsize=14)
        plt.ylabel('Average Daily Demand per Pile (kWh)', fontsize=14)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(out_combined, dpi=300)
        plt.close()
        print(f"-> Saved {out_combined}")
        
        # Plot 5: ACN Specific Time Window Isolated Plot
        out_acn_isolated = os.path.join(plots_dir, "acn_isolated_demand.png")
        plt.figure(figsize=(14, 7))
        plt.plot(daily_acn_combined['plot_date'], daily_acn_combined['avg_demand_per_pile'], label='ACN Workplace (Avg per Charger)', color='#2ca02c', marker='^', linewidth=2, linestyle='--')
        
        plt.title('Daily Demand Trend per Charger: ACN Workplace (June 18 - July 19)', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Average Daily Demand per Charger (kWh)', fontsize=14)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(out_acn_isolated, dpi=300)
        plt.close()
        print(f"-> Saved {out_acn_isolated}")

        # ==========================================
        # 4. Process Intraday Demand per Pile Comparison
        # ==========================================
        print("Generating intraday demand comparison plot...")
        # ACN Intraday
        acn_intraday = df_acn.groupby('hour')['total_kwh_this_hour'].mean().reset_index()
        acn_intraday['avg_demand_per_pile'] = acn_intraday['total_kwh_this_hour'] / 54.0
        
        # UrbanEV Non-CBD Intraday
        non_cbd_intraday_agg = df_urban[df_urban['CBD'] == 0].groupby('hour').agg(
            total_demand=('charging_demand_kwh', 'sum'),
            total_piles=('total_piles', 'sum')
        ).reset_index()
        non_cbd_intraday_agg['avg_demand_per_pile'] = non_cbd_intraday_agg['total_demand'] / non_cbd_intraday_agg['total_piles']
        
        # UrbanEV CBD Intraday
        cbd_intraday_agg = df_urban[df_urban['CBD'] == 1].groupby('hour').agg(
            total_demand=('charging_demand_kwh', 'sum'),
            total_piles=('total_piles', 'sum')
        ).reset_index()
        cbd_intraday_agg['avg_demand_per_pile'] = cbd_intraday_agg['total_demand'] / cbd_intraday_agg['total_piles']

        out_intraday_comparison = os.path.join(plots_dir, "intraday_demand_comparison.png")
        plt.figure(figsize=(12, 7))
        plt.plot(non_cbd_intraday_agg['hour'], non_cbd_intraday_agg['avg_demand_per_pile'], color='#1f77b4', marker='o', linewidth=2.5, label='UrbanEV Non-CBD (Avg per Pile)')
        plt.plot(cbd_intraday_agg['hour'], cbd_intraday_agg['avg_demand_per_pile'], color='#d62728', marker='s', linewidth=2.5, label='UrbanEV CBD (Avg per Pile)')
        plt.plot(acn_intraday['hour'], acn_intraday['avg_demand_per_pile'], color='#2ca02c', marker='^', linewidth=2.5, linestyle='--', label='ACN Workplace (Avg per Charger)')
        
        plt.title("Intraday Average Demand per Pile Comparison", fontsize=16)
        plt.xlabel("Hour of Day (0-23)", fontsize=14)
        plt.ylabel("Average Hourly Demand per Pile (kWh)", fontsize=14)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(out_intraday_comparison, dpi=300)
        plt.close()
        print(f"-> Saved {out_intraday_comparison}")

        print("\nAll plots generated successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_all_plots()
