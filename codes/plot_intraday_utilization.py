import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_intraday_plots():
    base_dir = r"."
    plots_dir = os.path.join(base_dir, "plots", "eda_demand_trends")
    # Input files
    acn_file = os.path.join(base_dir, "datasets", "processed", "acn_hourly.csv")
    urban_file = os.path.join(base_dir, "datasets", "processed", "urbanev_hourly_v2.csv")
    # Output files
    out_acn = os.path.join(plots_dir, "acn_intraday_utilization.png")
    out_noncbd = os.path.join(plots_dir, "urbanev_noncbd_intraday_utilization.png")
    out_cbd = os.path.join(plots_dir, "urbanev_cbd_intraday_utilization.png")
    try:
        # ==========================================
        # 1. Process ACN Dataset (US Workplaces)
        # ==========================================
        print("Processing ACN dataset...")
        df_acn = pd.read_csv(acn_file)
        
        # Calculate Active Charging Ratio (actively_charging / active_sessions)
        # Handle division by zero
        df_acn['charger_util'] = np.where(
            df_acn['active_sessions'] > 0, 
            df_acn['actively_charging'] / df_acn['active_sessions'], 
            0
        )
        
        acn_agg = df_acn.groupby(['hour', 'is_weekend'])['charger_util'].mean().reset_index()
        
        acn_weekday = acn_agg[acn_agg['is_weekend'] == 0]
        acn_weekend = acn_agg[acn_agg['is_weekend'] == 1]
        
        plt.figure(figsize=(10, 6))
        plt.plot(acn_weekday['hour'], acn_weekday['charger_util'], label='Weekday', color='#1f77b4', marker='o', linewidth=2)
        plt.plot(acn_weekend['hour'], acn_weekend['charger_util'], label='Weekend', color='#d62728', marker='s', linestyle='--', linewidth=2)
        
        plt.title('Intraday Charger Utilization Rate (ACN / Workplace)', fontsize=16)
        plt.xlabel('Hour of Day (0-23)', fontsize=14)
        plt.ylabel('Active Charging Ratio (Charging/Connected)', fontsize=14)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(out_acn, dpi=300)
        plt.close()
        print(f"-> Saved {out_acn}")
        # ==========================================
        # 2. Process UrbanEV Dataset (Shenzhen)
        # ==========================================
        print("Processing UrbanEV dataset...")
        df_urban = pd.read_csv(urban_file)
        
        # The charger_util is already calculated in process_urbanev.py as:
        # total_connected_duration_mins / (total_piles * 60)
        
        urban_agg = df_urban.groupby(['hour', 'is_weekend', 'CBD'])['charger_util'].mean().reset_index()
        
        # Non-CBD
        noncbd_weekday = urban_agg[(urban_agg['CBD'] == 0) & (urban_agg['is_weekend'] == 0)]
        noncbd_weekend = urban_agg[(urban_agg['CBD'] == 0) & (urban_agg['is_weekend'] == 1)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(noncbd_weekday['hour'], noncbd_weekday['charger_util'], label='Weekday', color='#1f77b4', marker='o', linewidth=2)
        plt.plot(noncbd_weekend['hour'], noncbd_weekend['charger_util'], label='Weekend', color='#d62728', marker='s', linestyle='--', linewidth=2)
        
        plt.title('Intraday Charger Utilization Rate (UrbanEV / Non-CBD)', fontsize=16)
        plt.xlabel('Hour of Day (0-23)', fontsize=14)
        plt.ylabel('Average Charger Utilization Rate', fontsize=14)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(out_noncbd, dpi=300)
        plt.close()
        print(f"-> Saved {out_noncbd}")
        # CBD
        cbd_weekday = urban_agg[(urban_agg['CBD'] == 1) & (urban_agg['is_weekend'] == 0)]
        cbd_weekend = urban_agg[(urban_agg['CBD'] == 1) & (urban_agg['is_weekend'] == 1)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(cbd_weekday['hour'], cbd_weekday['charger_util'], label='Weekday', color='#1f77b4', marker='o', linewidth=2)
        plt.plot(cbd_weekend['hour'], cbd_weekend['charger_util'], label='Weekend', color='#d62728', marker='s', linestyle='--', linewidth=2)
        
        plt.title('Intraday Charger Utilization Rate (UrbanEV / CBD)', fontsize=16)
        plt.xlabel('Hour of Day (0-23)', fontsize=14)
        plt.ylabel('Average Charger Utilization Rate', fontsize=14)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(out_cbd, dpi=300)
        plt.close()
        print(f"-> Saved {out_cbd}")
        print("\nAll intraday utilization plots generated successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    generate_intraday_plots()