import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 1: Loading Dataset")
print("=" * 80)
df = pd.read_csv('Dataset/EVcharging.csv')
print(f"Original dataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Data Preprocessing")
print("=" * 80)

# Create a copy to work with
df_processed = df.copy()

# 1. Convert time duration strings to numeric (minutes)
def parse_duration(duration_str):
    """Convert '0 days 08:45:22' to total minutes"""
    try:
        parts = str(duration_str).split()
        if len(parts) >= 3:
            days = int(parts[0])
            time_parts = parts[2].split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = int(time_parts[2])
            total_minutes = days * 24 * 60 + hours * 60 + minutes + seconds / 60
            return total_minutes
        return 0
    except:
        return 0

print("Converting time duration strings to numeric...")
df_processed['Total Duration (minutes)'] = df_processed['Total Duration (hh:mm:ss)'].apply(parse_duration)
df_processed['Charging Time (minutes)'] = df_processed['Charging Time (hh:mm:ss)'].apply(parse_duration)

# 2. Parse timestamps and extract time features
print("Extracting time-based features from timestamps...")

# Convert timestamp columns to datetime
df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], format='%d-%m-%Y %H:%M', errors='coerce')
df_processed['Start Date'] = pd.to_datetime(df_processed['Start Date'], format='%d-%m-%Y %H:%M', errors='coerce')
df_processed['End Date'] = pd.to_datetime(df_processed['End Date'], format='%d-%m-%Y %H:%M', errors='coerce')
df_processed['timestamp_hourly'] = pd.to_datetime(df_processed['timestamp_hourly'], format='%d-%m-%Y %H:%M', errors='coerce')

# Extract time features
df_processed['hour'] = df_processed['timestamp'].dt.hour
df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
df_processed['day_of_month'] = df_processed['timestamp'].dt.day
df_processed['month'] = df_processed['timestamp'].dt.month
df_processed['year'] = df_processed['timestamp'].dt.year
df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
df_processed['is_weekday'] = (df_processed['day_of_week'] < 5).astype(int)

# Create time of day categories
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df_processed['time_of_day'] = df_processed['hour'].apply(get_time_of_day)

# Create season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df_processed['season'] = df_processed['month'].apply(get_season)

# Peak hours (typical 8-10 AM and 5-7 PM)
df_processed['is_peak_hour'] = ((df_processed['hour'] >= 8) & (df_processed['hour'] <= 10)) | \
                               ((df_processed['hour'] >= 17) & (df_processed['hour'] <= 19))
df_processed['is_peak_hour'] = df_processed['is_peak_hour'].astype(int)

# 3. Encode categorical variables
print("Encoding categorical variables...")

# Port Type encoding (Level 1 = 0, Level 2 = 1)
df_processed['Port Type Encoded'] = (df_processed['Port Type'] == 'Level 2').astype(int)

# Plug Type encoding
plug_type_map = {'J1772': 1, 'NEMA 5-20R': 0}
df_processed['Plug Type Encoded'] = df_processed['Plug Type'].map(plug_type_map)

# Ended By - one-hot encode or label encode (we'll use label encode for correlation)
from sklearn.preprocessing import LabelEncoder
le_ended = LabelEncoder()
df_processed['Ended By Encoded'] = le_ended.fit_transform(df_processed['Ended By'])

# County encoding
df_processed['County Encoded'] = (df_processed['County'] == 'Santa Clara County').astype(int)

# Time of day encoding
time_of_day_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
df_processed['time_of_day_encoded'] = df_processed['time_of_day'].map(time_of_day_map)

# Season encoding
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
df_processed['season_encoded'] = df_processed['season'].map(season_map)

# 4. Handle missing values (if any)
print("Checking for missing values...")
missing = df_processed.isnull().sum()
if missing.sum() > 0:
    print(f"Found {missing.sum()} missing values")
    # Fill numeric missing values with median
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
else:
    print("No missing values found!")

print("[OK] Preprocessing complete!")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Feature Engineering")
print("=" * 80)

# 1. Interaction Features
print("Creating interaction features...")
df_processed['Energy per Hour'] = df_processed['Energy (kWh)'] / (df_processed['Charging Time (minutes)'] / 60 + 1e-6)  # Avoid division by zero
df_processed['Fee per kWh'] = df_processed['Fee'] / (df_processed['Energy (kWh)'] + 1e-6)
df_processed['Energy per Minute'] = df_processed['Energy (kWh)'] / (df_processed['Charging Time (minutes)'] + 1e-6)

# Efficiency: Energy / Duration ratio
df_processed['Charging Efficiency'] = df_processed['Energy (kWh)'] / (df_processed['Total Duration (minutes)'] / 60 + 1e-6)

# 2. User Behavior Features
print("Creating user behavior features...")
user_stats = df_processed.groupby('User ID').agg({
    'Energy (kWh)': ['mean', 'std', 'count'],
    'Charging Time (minutes)': 'mean',
    'Fee': 'mean'
}).reset_index()
user_stats.columns = ['User ID', 'User Avg Energy', 'User Std Energy', 'User Frequency', 
                      'User Avg Charging Time', 'User Avg Fee']
df_processed = df_processed.merge(user_stats, on='User ID', how='left')

# 3. Station Features
print("Creating station features...")
station_stats = df_processed.groupby('Station Name').agg({
    'Energy (kWh)': ['mean', 'std', 'count'],
    'Charging Time (minutes)': 'mean',
    'Fee': 'mean',
    'User ID': 'nunique'
}).reset_index()
station_stats.columns = ['Station Name', 'Station Avg Energy', 'Station Std Energy', 'Station Popularity',
                         'Station Avg Charging Time', 'Station Avg Fee', 'Station Unique Users']
df_processed = df_processed.merge(station_stats, on='Station Name', how='left')

# 4. Location-based features (if needed - keeping lat/lon for now)
print("Location features (Latitude/Longitude) already present...")

# 5. Time-based aggregations
print("Creating time-based aggregated features...")
# Average energy by hour of day
hourly_avg = df_processed.groupby('hour')['Energy (kWh)'].mean().to_dict()
df_processed['Hourly Avg Energy'] = df_processed['hour'].map(hourly_avg)

# Average energy by day of week
dow_avg = df_processed.groupby('day_of_week')['Energy (kWh)'].mean().to_dict()
df_processed['Day of Week Avg Energy'] = df_processed['day_of_week'].map(dow_avg)

# Average energy by month
monthly_avg = df_processed.groupby('month')['Energy (kWh)'].mean().to_dict()
df_processed['Monthly Avg Energy'] = df_processed['month'].map(monthly_avg)

print("[OK] Feature engineering complete!")

# ============================================================================
# STEP 4: Prepare Data for Correlation Analysis
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Preparing Data for Correlation Analysis")
print("=" * 80)

# Select numeric columns for correlation analysis
numeric_features = [
    # Original numeric features
    'Energy (kWh)', 'GHG Savings (kg)', 'Gasoline Savings (gallons)', 'Fee',
    'Latitude', 'Longitude', 'Port Number', 'EVSE ID', 'User ID',
    
    # Preprocessed time features
    'Total Duration (minutes)', 'Charging Time (minutes)',
    'hour', 'day_of_week', 'day_of_month', 'month', 'year',
    'is_weekend', 'is_weekday', 'is_peak_hour',
    
    # Encoded categorical features
    'Port Type Encoded', 'Plug Type Encoded', 'Ended By Encoded',
    'County Encoded', 'time_of_day_encoded', 'season_encoded',
    
    # Engineered features
    'Energy per Hour', 'Fee per kWh', 'Energy per Minute', 'Charging Efficiency',
    'User Avg Energy', 'User Std Energy', 'User Frequency',
    'User Avg Charging Time', 'User Avg Fee',
    'Station Avg Energy', 'Station Std Energy', 'Station Popularity',
    'Station Avg Charging Time', 'Station Avg Fee', 'Station Unique Users',
    'Hourly Avg Energy', 'Day of Week Avg Energy', 'Monthly Avg Energy'
]

# Filter to only include columns that exist
available_features = [col for col in numeric_features if col in df_processed.columns]
correlation_df = df_processed[available_features].copy()

print(f"Selected {len(available_features)} numeric features for correlation analysis")

# ============================================================================
# STEP 5: Compute Correlation Matrix
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Computing Correlation Matrix")
print("=" * 80)

correlation_matrix = correlation_df.corr()
print(f"Correlation matrix shape: {correlation_matrix.shape}")
print("[OK] Correlation matrix computed!")

# ============================================================================
# STEP 6: Visualize Correlation Matrix
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Creating Correlation Visualizations")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 16)

# Full correlation heatmap
print("Creating full correlation heatmap...")
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Full Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: correlation_heatmap_full.png")
plt.close()

# Correlation with Energy (kWh) - target variable
print("Creating correlation with Energy (kWh)...")
energy_corr = correlation_matrix['Energy (kWh)'].sort_values(ascending=False)
energy_corr = energy_corr[energy_corr.index != 'Energy (kWh)']  # Remove self-correlation

plt.figure(figsize=(12, 10))
energy_corr.plot(kind='barh', color='steelblue')
plt.title('Correlation with Energy (kWh)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('correlation_with_energy.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: correlation_with_energy.png")
plt.close()

# Top correlations (absolute value)
print("Creating top correlations visualization...")
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_val = correlation_matrix.loc[col1, col2]
        if not np.isnan(corr_val):
            corr_pairs.append((col1, col2, abs(corr_val)))

corr_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)
top_correlations = corr_pairs[:30]  # Top 30 correlations

fig, ax = plt.subplots(figsize=(14, 10))
pair_names = [f"{pair[0][:20]} vs {pair[1][:20]}" for pair in top_correlations]
corr_values = [pair[2] for pair in top_correlations]
colors = ['red' if pair[2] > 0.8 else 'orange' if pair[2] > 0.5 else 'blue' for pair in top_correlations]

ax.barh(range(len(pair_names)), corr_values, color=colors)
ax.set_yticks(range(len(pair_names)))
ax.set_yticklabels(pair_names, fontsize=9)
ax.set_xlabel('Absolute Correlation Coefficient', fontsize=12)
ax.set_title('Top 30 Feature Correlations (Absolute Values)', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('top_correlations.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: top_correlations.png")
plt.close()

# Highly correlated pairs (threshold > 0.8 or < -0.8)
print("\nIdentifying highly correlated pairs...")
high_corr_pairs = [(pair[0], pair[1], pair[2]) for pair in corr_pairs if pair[2] > 0.8]
print(f"Found {len(high_corr_pairs)} pairs with correlation > 0.8")

if high_corr_pairs:
    print("\nHighly Correlated Pairs (> 0.8):")
    for pair in high_corr_pairs[:20]:  # Show top 20
        print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

print("\n[OK] All visualizations created!")

# ============================================================================
# STEP 7: Save Preprocessed Dataset
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Saving Preprocessed Dataset")
print("=" * 80)

# Save the full preprocessed dataset with all new features
df_processed.to_csv('Dataset/EVcharging_preprocessed.csv', index=False)
print(f"[OK] Saved preprocessed dataset: Dataset/EVcharging_preprocessed.csv")
print(f"  Final shape: {df_processed.shape}")
print(f"  Total columns: {len(df_processed.columns)}")

print("\n" + "=" * 80)
print("ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nFiles created:")
print("  1. Dataset/EVcharging_preprocessed.csv - Preprocessed dataset with new features")
print("  2. correlation_heatmap_full.png - Full correlation matrix heatmap")
print("  3. correlation_with_energy.png - Correlation with Energy (kWh)")
print("  4. top_correlations.png - Top 30 feature correlations")
print("\nNext: Review the correlation visualizations to identify important features!")

