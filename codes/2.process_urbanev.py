import pandas as pd
import numpy as np

print("Loading UrbanEV datasets...")
time_df = pd.read_csv(r"datasets/raw/UrbanEV_ SZ_districts/time.csv")
occ_df = pd.read_csv(r"datasets/raw/UrbanEV_ SZ_districts/occupancy.csv")
vol_df = pd.read_csv(r"datasets/raw/UrbanEV_ SZ_districts/volume.csv")
dur_df = pd.read_csv(r"datasets/raw/UrbanEV_ SZ_districts/duration.csv")
info_df = pd.read_csv(r"datasets/raw/UrbanEV_ SZ_districts/information.csv")

print("Calculating Spatial Network Features (Haversine Distance)...")
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

coords = info_df[['grid', 'lon', 'la']].dropna().reset_index(drop=True)
n = len(coords)
dist_matrix = np.zeros((n, n))
for i in range(n):
    dist_matrix[i, :] = haversine_np(coords['lon'].iloc[i], coords['la'].iloc[i], coords['lon'].values, coords['la'].values)

info_df['grid_adj_count'] = np.sum((dist_matrix <= 3) & (dist_matrix > 0), axis=1)

mean_dists = []
for i in range(n):
    nearest_5 = np.sort(dist_matrix[i, :])[1:6]
    mean_dists.append(np.mean(nearest_5))
info_df['grid_mean_dist'] = mean_dists

print("Processing time features...")
def _get_time_bucket(hour):
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

# Create a proper datetime
time_df['datetime'] = pd.to_datetime(
    time_df[['year', 'month', 'day', 'hour', 'minute', 'second']]
)
time_df['timestamp_id'] = time_df.index + 1
time_df['hour_floor'] = time_df['datetime'].dt.floor('h')
time_df['is_weekend'] = (time_df['datetime'].dt.dayofweek >= 5).astype(int)
time_df['day_of_week'] = time_df['datetime'].dt.dayofweek
time_df['month'] = time_df['datetime'].dt.month
time_df['time_bucket'] = time_df['hour'].apply(_get_time_bucket)

print("Melting occupancy, volume, and duration...")
# The first column is 'timestamp', rest are grid IDs
occ_melt = occ_df.melt(id_vars=['timestamp'], var_name='grid', value_name='occupancy')
vol_melt = vol_df.melt(id_vars=['timestamp'], var_name='grid', value_name='volume')
dur_melt = dur_df.melt(id_vars=['timestamp'], var_name='grid', value_name='duration')

# Convert grid to int
occ_melt['grid'] = occ_melt['grid'].astype(int)
vol_melt['grid'] = vol_melt['grid'].astype(int)
dur_melt['grid'] = dur_melt['grid'].astype(int)

# Merge occupancy, volume, and duration
print("Merging data...")
df = pd.merge(occ_melt, vol_melt, on=['timestamp', 'grid'])
df = pd.merge(df, dur_melt, on=['timestamp', 'grid'])

# Merge with time
df = pd.merge(df, time_df[['timestamp_id', 'datetime', 'hour_floor', 'hour', 'is_weekend', 'day_of_week', 'month', 'time_bucket']], left_on='timestamp', right_on='timestamp_id')

# Merge with info
df = pd.merge(df, info_df[['grid', 'count', 'CBD', 'area', 'lon', 'la', 'grid_adj_count', 'grid_mean_dist']], on='grid')
df = df.rename(columns={'count': 'total_piles'})

# Utilization
df['utilization'] = df['occupancy'] / df['total_piles']
df['utilization'] = df['utilization'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Congestion indicator (occupancy > 80)
df['is_congested'] = (df['occupancy'] > 80).astype(int)

# Queue Length Proxy: Bounded +1 / -1 dynamic counter
# Sort by grid and chronological order to ensure correct time-series counting
df = df.sort_values(['grid', 'timestamp'])

# A window is full if utilization is 1.0 or greater
is_full_arr = (df['utilization'] >= 1.0).values
grids_arr = df['grid'].values

n = len(grids_arr)
queue_arr = np.zeros(n, dtype=int)

current_grid = -1
current_q = 0

# Fast numpy loop to calculate the stateful bounded queue
for i in range(n):
    # Reset queue if we switch to a new grid
    if grids_arr[i] != current_grid:
        current_grid = grids_arr[i]
        current_q = 0
    
    if is_full_arr[i]:
        current_q += 1
    else:
        if current_q > 0:
            current_q -= 1
            
    queue_arr[i] = current_q

df['queue_length_proxy'] = queue_arr

print("Aggregating to hourly windows...")
hourly = df.groupby(['grid', 'hour_floor']).agg(
    charging_demand_kwh=('volume', 'sum'),
    congestion_probability=('is_congested', 'mean'),
    queue_length_proxy=('queue_length_proxy', 'last'),
    occupancy_density=('utilization', 'mean'),
    total_charging_hours=('duration', 'sum'),
    total_piles=('total_piles', 'first'),
    CBD=('CBD', 'first'),
    area=('area', 'first'),
    lon=('lon', 'first'),
    la=('la', 'first'),
    grid_adj_count=('grid_adj_count', 'first'),
    grid_mean_dist=('grid_mean_dist', 'first'),
    hour=('hour', 'first'),
    day_of_week=('day_of_week', 'first'),
    month=('month', 'first'),
    is_weekend=('is_weekend', 'first'),
    time_bucket=('time_bucket', 'first')
).reset_index()

# The duration is in hours. Summing across the hour gives total charging hours for the grid.
# Dividing by total_piles gives the average charging hours per individual charger.
# Since the time window is 1 hour, (charging hours per charger / 1 hour) gives the exact charger_util rate!
hourly['charger_util'] = hourly['total_charging_hours'] / hourly['total_piles']

# Calculate charging load (Average Power Draw in kW)
# charging_demand_kwh (energy) / total_charging_hours (time) = kW
hourly['chargingload'] = hourly['charging_demand_kwh'] / hourly['total_charging_hours'].replace(0, np.nan)
hourly['chargingload'] = hourly['chargingload'].fillna(0)



# Calculate cyclic time features
hourly['hour_sin'] = np.sin(2 * np.pi * hourly['hour'] / 24)
hourly['hour_cos'] = np.cos(2 * np.pi * hourly['hour'] / 24)
hourly['dayofweek_sin'] = np.sin(2 * np.pi * hourly['day_of_week'] / 7)
hourly['dayofweek_cos'] = np.cos(2 * np.pi * hourly['day_of_week'] / 7)

# Removed historical_intensity and turnover_rate based on user preference to use lag features instead.

# Reorder columns as requested
ordered_cols = ['grid', 'hour_floor', 'hour', 'hour_sin', 'hour_cos', 'day_of_week', 'dayofweek_sin', 'dayofweek_cos', 'month', 'is_weekend', 'time_bucket', 'CBD', 'lon', 'la', 'area', 'grid_adj_count', 'grid_mean_dist']
remaining_cols = [col for col in hourly.columns if col not in ordered_cols]
hourly = hourly[ordered_cols + remaining_cols]

print(f"Hourly dataset shape: {hourly.shape}")
hourly.to_csv(r"datasets/processed/urbanev_hourly_v2.csv", index=False)
print("Saved to urbanev_hourly_v2.csv!")

