"""
ACN Hourly Aggregation Script
==============================
Converts session-level preprocessed ACN data into 1-hour interval time windows.

For each hourly window:
  - Identifies which sessions are ACTIVE (overlapping that hour)
  - Computes aggregated metrics across all active sessions

Output: acn_hourly.csv
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# ── 1. Load and Preprocess Raw Data ──────────────────────────────────────────
print("Loading raw acn_sessions.csv...")
raw_file_path = r"datasets/raw/acn_sessions.csv"
df = pd.read_csv(raw_file_path)

print("Preprocessing raw ACN data...")
df.drop(columns=['timezone','_meta','end','min_kWh','start','_items','userInputs'], inplace=True, errors='ignore')
df.drop(columns=['userID.1'], inplace=True, errors='ignore')
df.drop(columns=['_id','site','clusterID','stationID','siteID','kWhRequested','milesRequested','WhPerMile','minutesAvailable','userID','modifiedAt','paymentRequired','requestedDeparture'], inplace=True, errors='ignore')

df = df.dropna(subset=["connectionTime", "disconnectTime","doneChargingTime","spaceID" ,"kWhDelivered"], how="all")
df = df.dropna(subset=["doneChargingTime"])

df['connectionTime'] = pd.to_datetime(df['connectionTime'], utc=True, errors='coerce')
df['disconnectTime'] = pd.to_datetime(df['disconnectTime'], utc=True, errors='coerce')
df['doneChargingTime'] = pd.to_datetime(df['doneChargingTime'], utc=True, errors='coerce')

df['session_duration_mins'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 60
df['charging_duration_mins'] = (df['doneChargingTime'] - df['connectionTime']).dt.total_seconds() / 60
df['charger_utilization_rate'] = (df['charging_duration_mins'] / df['session_duration_mins']).clip(0, 1)

BASELINE_TARIFF = 15  # ₹ per kWh
df['revenue_baseline'] = df['kWhDelivered'] * BASELINE_TARIFF
df['avg_power_kw'] = (df['kWhDelivered'] / (df['charging_duration_mins'] / 60)).replace([np.inf, -np.inf], np.nan)
df['avg_power_kw'] = df['avg_power_kw'].fillna(df['avg_power_kw'].mean())

df['hour']       = df['connectionTime'].dt.hour
df['day_of_week']= df['connectionTime'].dt.dayofweek
df['month']      = df['connectionTime'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

def time_bucket(hour):
    if pd.isna(hour): return 'Unknown'
    if 7 <= hour < 10:   return 'Morning Peak'
    elif 10 <= hour < 16: return 'Midday'
    elif 16 <= hour < 20: return 'Evening Peak'
    elif 20 <= hour < 24: return 'Night'
    else:                 return 'Off-Peak'

df['time_bucket'] = df['hour'].apply(time_bucket)
df.drop(columns=['connectionTime','disconnectTime','doneChargingTime'], inplace=True, errors='ignore')

print(f"Total sessions after preprocessing: {len(df)}")

# ── 2. Extract start timestamps from sessionID ───────────────────────────────
# sessionID format: "2_39_78_362_2018-04-25 11:08:04.400812"
# The timestamp is everything after the 4th underscore

def extract_start_time(session_id):
    """Extract the datetime from the sessionID string."""
    parts = str(session_id).split('_')
    # The timestamp starts after the 4th underscore
    # e.g., "2_39_78_362_2018-04-25 11:08:04.400812"
    timestamp_str = '_'.join(parts[4:])  # "2018-04-25 11:08:04.400812"
    try:
        return pd.to_datetime(timestamp_str)
    except:
        return pd.NaT

print("\nExtracting start timestamps from sessionIDs...")
df['start_time'] = df['sessionID'].apply(extract_start_time)

# Drop rows where timestamp extraction failed
before = len(df)
df = df.dropna(subset=['start_time'])
print(f"Rows after dropping invalid timestamps: {len(df)} (dropped {before - len(df)})")

# ── 3. Compute end timestamps ────────────────────────────────────────────────
# end_time = start_time + session_duration_mins
df['end_time'] = df['start_time'] + pd.to_timedelta(df['session_duration_mins'], unit='m')

# Also compute charging end time for reference
df['charging_end_time'] = df['start_time'] + pd.to_timedelta(df['charging_duration_mins'], unit='m')

print(f"\nDate range: {df['start_time'].min()} to {df['end_time'].max()}")
print(f"Unique chargers (spaceIDs): {df['spaceID'].nunique()}")

# ── 4. Create the hourly time index ──────────────────────────────────────────
# Create hourly buckets spanning the entire dataset
hour_start = df['start_time'].min().floor('h')  # Round down to nearest hour
hour_end = df['end_time'].max().ceil('h')        # Round up to nearest hour

hourly_index = pd.date_range(start=hour_start, end=hour_end, freq='1h')
print(f"\nTotal hourly windows: {len(hourly_index) - 1}")
print(f"From: {hourly_index[0]}")
print(f"To:   {hourly_index[-1]}")

# ── 5. For each hourly window, find active sessions & aggregate ──────────────
print("\nComputing hourly aggregations (this may take a moment)...")

# Pre-convert to numpy arrays for speed
start_times = df['start_time'].values
end_times = df['end_time'].values
charging_end_times = df['charging_end_time'].values
utilization_rates = df['charger_utilization_rate'].values
avg_power = df['avg_power_kw'].values
kwh_delivered = df['kWhDelivered'].values
revenue = df['revenue_baseline'].values
session_durations = df['session_duration_mins'].values
charging_durations = df['charging_duration_mins'].values

def _get_time_bucket(hour):
    """Map hour to time bucket matching the preprocessed data."""
    if 0 <= hour < 7:
        return 'Off-Peak'
    elif 7 <= hour < 10:
        return 'Morning Peak'
    elif 10 <= hour < 16:
        return 'Midday'
    elif 16 <= hour < 20:
        return 'Evening Peak'
    else:
        return 'Night'

results = []

for i in range(len(hourly_index) - 1):
    window_start = hourly_index[i]
    window_end = hourly_index[i + 1]
    
    ws = np.datetime64(window_start)
    we = np.datetime64(window_end)
    
    # A session is ACTIVE during this hour if:
    #   session_start < window_end AND session_end > window_start
    # (i.e., the session interval overlaps with the hourly window)
    active_mask = (start_times < we) & (end_times > ws)
    
    # Also check which sessions are ACTIVELY CHARGING during this window
    # (charging hasn't ended yet)
    charging_mask = active_mask & (charging_end_times > ws)
    
    n_active = active_mask.sum()
    n_charging = charging_mask.sum()
    
    if n_active > 0:
        # ── Metrics for ACTIVE sessions ──
        active_util = utilization_rates[active_mask]
        active_power = avg_power[active_mask]
        active_kwh = kwh_delivered[active_mask]
        active_rev = revenue[active_mask]
        active_sess_dur = session_durations[active_mask]
        active_chrg_dur = charging_durations[active_mask]
        
        # Compute proportional kWh contribution to this specific hour
        # For each active session, what fraction of its total duration falls in this hour?
        sess_starts = start_times[active_mask]
        sess_ends = end_times[active_mask]
        
        # Overlap: max(start, window_start) to min(end, window_end)
        overlap_start = np.maximum(sess_starts, ws)
        overlap_end = np.minimum(sess_ends, we)
        overlap_mins = (overlap_end - overlap_start) / np.timedelta64(1, 'm')
        overlap_mins = np.maximum(overlap_mins, 0)
        
        # Proportional kWh = (overlap_mins / session_duration) * kWhDelivered
        proportional_kwh = np.where(
            active_sess_dur > 0,
            (overlap_mins / active_sess_dur) * active_kwh,
            0
        )
        
        # Proportional revenue
        proportional_revenue = np.where(
            active_sess_dur > 0,
            (overlap_mins / active_sess_dur) * active_rev,
            0
        )
        
        row = {
            'timestamp': window_start,
            'hour': window_start.hour,
            'day_of_week': window_start.dayofweek,
            'month': window_start.month,
            'is_weekend': int(window_start.dayofweek >= 5),
            'time_bucket': _get_time_bucket(window_start.hour),
            
            # Session counts
            'active_sessions': n_active,
            'actively_charging': n_charging,
            'idle_sessions': n_active - n_charging,
            
            # Utilization metrics
            'avg_utilization_rate': np.mean(active_util),
            'median_utilization_rate': np.median(active_util),
            'max_utilization_rate': np.max(active_util),
            'min_utilization_rate': np.min(active_util),
            
            # Power / Load metrics
            'avg_power_kw': np.mean(active_power),
            'total_power_kw': np.sum(active_power),  # total load across all active chargers
            'max_power_kw': np.max(active_power),
            
            # Energy metrics (proportional to this hour)
            'total_kwh_this_hour': np.sum(proportional_kwh),
            'avg_kwh_per_session': np.mean(proportional_kwh),
            
            # Revenue metrics (proportional to this hour)
            'total_revenue_this_hour': np.sum(proportional_revenue),
            'avg_revenue_per_session': np.mean(proportional_revenue),
            
            # Duration averages
            'avg_session_duration_mins': np.mean(active_sess_dur),
            'avg_charging_duration_mins': np.mean(active_chrg_dur),
        }
    else:
        row = {
            'timestamp': window_start,
            'hour': window_start.hour,
            'day_of_week': window_start.dayofweek,
            'month': window_start.month,
            'is_weekend': int(window_start.dayofweek >= 5),
            'time_bucket': _get_time_bucket(window_start.hour),
            
            'active_sessions': 0,
            'actively_charging': 0,
            'idle_sessions': 0,
            
            'avg_utilization_rate': 0.0,
            'median_utilization_rate': 0.0,
            'max_utilization_rate': 0.0,
            'min_utilization_rate': 0.0,
            
            'avg_power_kw': 0.0,
            'total_power_kw': 0.0,
            'max_power_kw': 0.0,
            
            'total_kwh_this_hour': 0.0,
            'avg_kwh_per_session': 0.0,
            
            'total_revenue_this_hour': 0.0,
            'avg_revenue_per_session': 0.0,
            
            'avg_session_duration_mins': 0.0,
            'avg_charging_duration_mins': 0.0,
        }
    
    results.append(row)


# ── 6. Build output DataFrame ────────────────────────────────────────────────
hourly_df = pd.DataFrame(results)

# Sort by timestamp
hourly_df = hourly_df.sort_values('timestamp').reset_index(drop=True)

print(f"\nHourly aggregation complete!")
print(f"Shape: {hourly_df.shape}")
print(f"\nSample rows (first 10):")
print(hourly_df.head(10).to_string())

print(f"\n\nBasic stats:")
print(hourly_df[['active_sessions', 'avg_utilization_rate', 'avg_power_kw', 
                  'total_kwh_this_hour', 'total_revenue_this_hour']].describe())

# ── 7. Save ──────────────────────────────────────────────────────────────────
output_path = r"datasets/processed/acn_hourly.csv"
hourly_df.to_csv(output_path, index=False)
print(f"\n[DONE] Saved to: {output_path}")
print(f"Total rows: {len(hourly_df)}")
