import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})

# --- CONFIGURATION & HYPERPARAMETERS ---
CBD_SURGE_TH = 0.2938
CBD_DISCOUNT_TH = 0.0866
NON_CBD_SURGE_TH = 0.3494
NON_CBD_DISCOUNT_TH = 0.0980

CBD_ELASTICITY = -0.8
NON_CBD_ELASTICITY = -0.3

MAX_STEP_INTRA = 0.10
MAX_MULTIPLIER = 1.85
MIN_MULTIPLIER = 0.50
STEPS_PER_HOUR = 5 # 60 mins / 12 mins

LEARNING_RATE = 0.005
EPOCHS = 10

def load_and_merge_data():
    print("Loading 100% full-timeline predictions and merging with spatial features...")
    preds = pd.read_csv(r"results/predictions2.csv")
    
    # We only need the prediction variables, but we need CBD and time variables from the original dataset
    orig = pd.read_csv(r"datasets/processed/urbanev_hourly_v2.csv", usecols=['grid', 'hour_floor', 'CBD', 'day_of_week', 'hour'])
    
    df = pd.merge(preds, orig, on=['grid', 'hour_floor'], how='inner')
    
    df['hour_floor'] = pd.to_datetime(df['hour_floor'])
    df = df.sort_values(by=['grid', 'hour_floor']).reset_index(drop=True)
    
    df['hour_of_week'] = df['day_of_week'] * 24 + df['hour']
    return df

def train_optimizer(df):
    print(f"Training Parameterized Heuristic Optimizer over {EPOCHS} Epochs...")
    
    # Matrices: Alpha (Surge) and Delta (Discount) for [CBD_Zone (0/1), HourOfWeek]
    alpha_matrix = np.ones((2, 168)) * 0.5
    delta_matrix = np.ones((2, 168)) * 0.5
    
    all_metrics = []
    
    for epoch in range(EPOCHS):
        total_reward = 0
        
        # We need to simulate chronological stepping per grid to track prev_price
        for grid_id, group in df.groupby('grid'):
            prev_price = 1.0
            
            for idx, row in group.iterrows():
                how = int(row['hour_of_week'])
                is_cbd = row['CBD'] == 1
                cbd_idx = 1 if is_cbd else 0
                
                pred_util = row['charger_util_predicted']
                pred_demand = row['charging_demand_kwh_predicted']
                pred_queue = row['queue_length_proxy_predicted']
                pred_congestion = row['congestion_probability_predicted']
                pred_load = row['chargingload_predicted']
                
                surge_th = CBD_SURGE_TH if is_cbd else NON_CBD_SURGE_TH
                disc_th = CBD_DISCOUNT_TH if is_cbd else NON_CBD_DISCOUNT_TH
                elasticity = CBD_ELASTICITY if is_cbd else NON_CBD_ELASTICITY
                
                alpha = alpha_matrix[cbd_idx, how]
                delta = delta_matrix[cbd_idx, how]
                
                # Heuristic Equation with Stress/Vacancy Scoring
                target_price = 1.0
                if pred_util > surge_th:
                    stress_score = (pred_util - surge_th) + pred_congestion + (pred_load / 240.0)
                    target_price = 1.0 + alpha * stress_score
                elif pred_util < disc_th:
                    vacancy_score = (disc_th - pred_util) + (1.0 - pred_congestion) + (1.0 - (pred_load / 240.0))
                    target_price = 1.0 - delta * vacancy_score
                
                target_price = np.clip(target_price, MIN_MULTIPLIER, MAX_MULTIPLIER)
                
                # Elasticity Simulator uses target_price directly
                multiplier_ratio = target_price / 1.0
                elastic_factor = multiplier_ratio ** elasticity
                
                shifted_demand = pred_demand * elastic_factor
                shifted_util = pred_util * elastic_factor
                shifted_queue = pred_queue * elastic_factor
                
                # Unified Reward Function
                if pred_demand > 0:
                    rev_gain_rate = ((shifted_demand * target_price) - (pred_demand * 1.0)) / (pred_demand * 1.0)
                else:
                    rev_gain_rate = 0.0
                    
                delta_util = shifted_util - pred_util
                uplift = shifted_demand - pred_demand
                delta_queue = shifted_queue - pred_queue
                
                if target_price < 1.0: # Discounting
                    R = (1.0 * rev_gain_rate) + (2.0 * delta_util) + (2.0 * uplift) - (0.0 * delta_queue)
                    # Update Delta
                    delta_matrix[cbd_idx, how] += LEARNING_RATE * R
                    delta_matrix[cbd_idx, how] = max(0.01, delta_matrix[cbd_idx, how])
                else: # Surging (or Holding)
                    R = (1.5 * rev_gain_rate) + (-2.0 * delta_util) + (0.0 * uplift) - (2.0 * delta_queue)
                    # Update Alpha
                    alpha_matrix[cbd_idx, how] += LEARNING_RATE * R
                    alpha_matrix[cbd_idx, how] = max(0.01, alpha_matrix[cbd_idx, how])
                
                
                total_reward += R
                
        print(f"\nEpoch {epoch + 1:>2} - Total Reward: {total_reward:,.1f}")
        # Run a silent production simulation to get the KPIs for this epoch
        res_df = run_production_simulation(df, alpha_matrix, delta_matrix, silent=True)
        epoch_rows = print_kpi_table(res_df)
        
        for r in epoch_rows:
            r['Epoch'] = epoch + 1
            r['Total Reward'] = round(total_reward, 1)
            all_metrics.append(r)
            
    metrics_df = pd.DataFrame(all_metrics)
    cols = ['Epoch', 'Total Reward', 'ZONE', 'Rev Gain', 'Q Drop', 'Uplift', 'Avg Price', 'Util %', 'CRR %']
    metrics_df = metrics_df[cols]
    out_path = r"results/loopevaluationmetrics.csv"
    metrics_df.to_csv(out_path, index=False)
    print(f"\nSaved loop evaluation metrics to {out_path}")
        
    return alpha_matrix, delta_matrix

def run_production_simulation(df, alpha_matrix, delta_matrix, silent=False):
    if not silent:
        print("Running Final Production Simulation with Optimized Matrices...")
    results = []
    
    for grid_id, group in df.groupby('grid'):
        prev_price = 1.0
        
        for idx, row in group.iterrows():
            how = int(row['hour_of_week'])
            is_cbd = row['CBD'] == 1
            cbd_idx = 1 if is_cbd else 0
            
            pred_util = row['charger_util_predicted']
            pred_demand = row['charging_demand_kwh_predicted']
            pred_queue = row['queue_length_proxy_predicted']
            pred_congestion = row['congestion_probability_predicted']
            pred_load = row['chargingload_predicted']
            
            # For accurate KPI metrics, we compare against the ACTUAL historical data, not the prediction
            actual_util = row['charger_util_actual']
            actual_demand = row['charging_demand_kwh_actual']
            actual_queue = row['queue_length_proxy_actual']
            
            surge_th = CBD_SURGE_TH if is_cbd else NON_CBD_SURGE_TH
            disc_th = CBD_DISCOUNT_TH if is_cbd else NON_CBD_DISCOUNT_TH
            elasticity = CBD_ELASTICITY if is_cbd else NON_CBD_ELASTICITY
            
            alpha = alpha_matrix[cbd_idx, how]
            delta = delta_matrix[cbd_idx, how]
            
            # Heuristic Equation with Stress/Vacancy Scoring
            target_price = 1.0
            if pred_util > surge_th:
                stress_score = (pred_util - surge_th) + pred_congestion + (pred_load / 240.0)
                target_price = 1.0 + alpha * stress_score
            elif pred_util < disc_th:
                vacancy_score = (disc_th - pred_util) + (1.0 - pred_congestion) + (1.0 - (pred_load / 240.0))
                target_price = 1.0 - delta * vacancy_score
            
            target_price = np.clip(target_price, MIN_MULTIPLIER, MAX_MULTIPLIER)
            
            # Elasticity Simulator uses target_price directly
            multiplier_ratio = target_price / 1.0
            elastic_factor = multiplier_ratio ** elasticity
            
            shifted_demand = actual_demand * elastic_factor
            shifted_util = actual_util * elastic_factor
            shifted_queue = actual_queue * elastic_factor
            
            results.append({
                'timestamp': row['hour_floor'],
                'grid': grid_id,
                'CBD': is_cbd,
                'original_demand': actual_demand,
                'shifted_demand': shifted_demand,
                'multiplier': target_price,
                'original_util': actual_util,
                'shifted_util': shifted_util,
                'original_queue': actual_queue,
                'shifted_queue': shifted_queue
            })
            
    return pd.DataFrame(results)

def print_kpi_table(res_df):
    BASE_RATE = 15.0
    print("="*95)
    print(" | ZONE                       | Rev Gain | Q Drop  | Uplift  | Avg Price | Util % | CRR % |")
    print(" |========================================================================================|")
    
    rows = []
    for is_cbd in [True, False]:
        zone_df = res_df[res_df['CBD'] == is_cbd]
        if len(zone_df) == 0: continue
        
        scenario_name = "CBD" if is_cbd else "Non-CBD"
        
        old_rev = (zone_df['original_demand'] * BASE_RATE).sum()
        new_rev = (zone_df['shifted_demand'] * BASE_RATE * zone_df['multiplier']).sum()
        rev_gain = ((new_rev - old_rev) / max(1, old_rev)) * 100
        
        old_q = zone_df['original_queue'].sum()
        new_q = zone_df['shifted_queue'].sum()
        q_drop = ((old_q - new_q) / max(1, old_q)) * 100
        
        surge_th = CBD_SURGE_TH if is_cbd else NON_CBD_SURGE_TH
        off_peak_df = zone_df[zone_df['original_util'] < surge_th]
        if len(off_peak_df) > 0 and off_peak_df['original_demand'].sum() > 0:
            uplift = ((off_peak_df['shifted_demand'].sum() - off_peak_df['original_demand'].sum()) / off_peak_df['original_demand'].sum()) * 100
        else:
            uplift = 0.0
            
        avg_price = (zone_df['multiplier'] * BASE_RATE).mean()
        util_pct = zone_df['shifted_util'].mean() * 100
        
        # Customer Response Rate (CRR): Percentage of total demand that was actively shifted
        total_demand = zone_df['original_demand'].sum()
        total_shifted_abs = np.abs(zone_df['shifted_demand'] - zone_df['original_demand']).sum()
        crr_pct = (total_shifted_abs / max(1, total_demand)) * 100
        
        print(f" | {scenario_name:<26} | {rev_gain:>+7.2f}% | {q_drop:>+6.1f}% | {uplift:>+6.2f}% | ¥{avg_price:>7.2f} | {util_pct:>5.1f}% | {crr_pct:>4.1f}% |")
        rows.append({
            'ZONE': scenario_name,
            'Rev Gain': f"{rev_gain:+.2f}%",
            'Q Drop': f"{q_drop:+.1f}%",
            'Uplift': f"{uplift:+.2f}%",
            'Avg Price': f"¥{avg_price:.2f}",
            'Util %': f"{util_pct:.1f}%",
            'CRR %': f"{crr_pct:.1f}%"
        })
        
    print(" " + "="*95)
    return rows

def generate_dashboard(res_df):
    os.makedirs(r'plots/agent', exist_ok=True)
    
    # --- VISUALIZATION DASHBOARD ---
    # 1. Popular Grids (Peak Shaving + Off-Peak Uplift)
    res_df['demand_drop'] = np.maximum(0, res_df['original_demand'] - res_df['shifted_demand'])
    res_df['demand_gain'] = np.maximum(0, res_df['shifted_demand'] - res_df['original_demand'])
    
    grid_stats = res_df.groupby('grid')[['demand_drop', 'demand_gain']].sum()
    grid_stats['combined_score'] = grid_stats['demand_drop'] * grid_stats['demand_gain']
    top_3_popular = grid_stats.nlargest(3, 'combined_score').index.tolist()
    
    print(f"Top 3 POPULAR Grids (Massive Peak Shaving + Uplift): {top_3_popular}")
    
    # 2. Unpopular Grids (Low Overall Utilization, Deep Discounts, High Demand Gain)
    max_util_per_grid = res_df.groupby('grid')['original_util'].max()
    # Filter grids that rarely get busy (Max historical utilization is under 30%)
    unpopular_candidates = max_util_per_grid[max_util_per_grid < 0.30].index.tolist()
    
    if len(unpopular_candidates) >= 3:
        top_3_unpopular = grid_stats.loc[unpopular_candidates].nlargest(3, 'demand_gain').index.tolist()
    else:
        # Fallback: Just grab the ones with the absolute lowest demand_drop (least surging) and highest demand_gain
        top_3_unpopular = grid_stats.nsmallest(20, 'demand_drop').nlargest(3, 'demand_gain').index.tolist()
        
    print(f"Top 3 UNPOPULAR Grids (Deep Discounts driving Utilization Uplift): {top_3_unpopular}")
    
    all_target_grids = [(g, "popular") for g in top_3_popular] + [(g, "unpopular") for g in top_3_unpopular]
    
    for target_grid, grid_type in all_target_grids:
        sample_df = res_df[res_df['grid'] == target_grid].sort_values('timestamp').head(168) # Plot 1 full week
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Panel 1: Demand
        axes[0].plot(sample_df['timestamp'], sample_df['original_demand'], label='Original Historical Demand (kWh)', color='red', linestyle='--', alpha=0.6)
        axes[0].plot(sample_df['timestamp'], sample_df['shifted_demand'], label='AI Shifted Demand (kWh)', color='green', linewidth=2.5)
        
        title_prefix = "Peak Shaving & Balancing" if grid_type == "popular" else "Deep Discount Demand Generation"
            
        axes[0].set_title(f"{title_prefix} (Grid {target_grid}): Original vs AI Shifted Demand")
        axes[0].set_ylabel("Demand (kWh)")
        axes[0].legend(loc="upper left")
        
        # Panel 2: Utilization & Congestion
        is_cbd = sample_df['CBD'].iloc[0]
        surge_th = CBD_SURGE_TH if is_cbd else NON_CBD_SURGE_TH
        disc_th = CBD_DISCOUNT_TH if is_cbd else NON_CBD_DISCOUNT_TH
        
        axes[1].plot(sample_df['timestamp'], sample_df['original_util'], label='Original Utilization', color='red', linestyle='--', alpha=0.6)
        axes[1].plot(sample_df['timestamp'], sample_df['shifted_util'], label='AI Optimized Utilization', color='green', linewidth=2.5)
        axes[1].axhline(surge_th, color='orange', linestyle=':', linewidth=2, label=f'Surge Threshold ({surge_th:.1%})')
        axes[1].axhline(disc_th, color='blue', linestyle=':', linewidth=2, label=f'Discount Threshold ({disc_th:.1%})')
        axes[1].set_title("Utilization Balancing")
        axes[1].set_ylabel("Utilization Rate")
        axes[1].legend(loc="upper left")
        
        # Panel 3: Tariff Multiplier
        axes[2].step(sample_df['timestamp'], sample_df['multiplier'], label='AI Tariff Multiplier', color='#023E73', linewidth=3, where='mid')
        axes[2].set_title("The Parameterized Dynamic Tariff (Intra-Hour Smoothed)")
        axes[2].set_ylabel("Price Multiplier")
        axes[2].set_ylim(0.4, 2.0)
        axes[2].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_xlabel("Time (168 Hour Snapshot)")
        axes[2].legend(loc="upper left")
        
        plt.tight_layout()
        save_path = fr"plots/agent/rl_dashboard_{grid_type}_grid_{target_grid}.png"
        plt.savefig(save_path, dpi=200)
        print(f"Dashboard saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    df = load_and_merge_data()
    alpha_matrix, delta_matrix = train_optimizer(df)
    res_df = run_production_simulation(df, alpha_matrix, delta_matrix)
    print("\nFINAL PRODUCTION METRICS:")
    print_kpi_table(res_df)
    generate_dashboard(res_df)
    print("\nPricing, Monitoring, and Learning Agent pipeline successfully completed!")
