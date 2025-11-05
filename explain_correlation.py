"""
EXPLANATION: How Correlation Works and Math Correction
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("MATH CORRECTION - Column Count")
print("=" * 80)

df = pd.read_csv('Dataset/EVcharging_preprocessed.csv')
total_columns = len(df.columns)
original_columns = 35
new_features = total_columns - original_columns

print(f"\nTotal columns in preprocessed dataset: {total_columns}")
print(f"Original columns: {original_columns}")
print(f"New features added: {new_features}")
print(f"\nIf we keep 35-40 columns:")
print(f"  We need to REMOVE: {total_columns - 40} to {total_columns - 35} columns")
print(f"  That is: {total_columns - 40} to {total_columns - 35} columns to remove")
print(f"\nCORRECTION: I was wrong! We need to remove 31-36 columns, not 10-15!")

print("\n" + "=" * 80)
print("HOW CORRELATION WORKS")
print("=" * 80)

print("\n1. WHAT IS CORRELATION?")
print("   - Correlation measures how two features move together")
print("   - Range: -1.0 to +1.0")
print("   - +1.0 = Perfect positive relationship (when one goes up, other goes up)")
print("   - -1.0 = Perfect negative relationship (when one goes up, other goes down)")
print("   - 0.0 = No relationship (they don't move together)")

print("\n2. HOW DID I CALCULATE IT?")
print("   - I used pandas DataFrame.corr() method")
print("   - This calculates Pearson Correlation Coefficient")
print("   - Formula: correlation = (how much features move together) / (variability of each feature)")

print("\n3. THE METHOD USED:")
print("   Step 1: Select all numeric columns from the dataset")
print("   Step 2: Calculate correlation between every pair of columns")
print("   Step 3: Create a correlation matrix (like a table)")
print("   Step 4: Each cell shows correlation between two features")

print("\n4. EXAMPLE:")
print("   If Energy (kWh) and Charging Time have correlation = 0.908:")
print("   - This means: When Charging Time increases, Energy usually increases")
print("   - 0.908 is close to 1.0, so it's a STRONG positive relationship")
print("   - This makes sense: longer charging = more energy consumed")

print("\n5. WHAT I USED IN THE CODE:")
print("   correlation_matrix = correlation_df.corr()")
print("   - This calculates correlation for all numeric columns")
print("   - Returns a matrix where:")
print("     - Rows = features")
print("     - Columns = features")
print("     - Values = correlation coefficients (-1 to +1)")

print("\n6. WHY REMOVE HIGHLY CORRELATED FEATURES?")
print("   - If two features have correlation > 0.8 or < -0.8:")
print("     - They provide the SAME information")
print("     - Keeping both is redundant")
print("     - Can cause overfitting in ML models")
print("     - Example: Energy and GHG Savings have 1.0 correlation")
print("       -> GHG Savings is calculated from Energy, so they're the same!")

print("\n7. VISUALIZATION:")
print("   - Heatmap: Shows correlation matrix as colors")
print("     - Red = positive correlation")
print("     - Blue = negative correlation")
print("     - White = no correlation")
print("   - Bar chart: Shows correlation of each feature with target (Energy)")

print("\n" + "=" * 80)
print("DEMONSTRATION: Let me show you the actual correlation calculation")
print("=" * 80)

# Load the data
df = pd.read_csv('Dataset/EVcharging_preprocessed.csv')

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nTotal numeric features: {len(numeric_cols)}")

# Calculate correlation
corr_matrix = df[numeric_cols].corr()

# Show correlation between Energy and Charging Time
print("\nExample: Correlation between Energy (kWh) and Charging Time (minutes):")
energy = df['Energy (kWh)']
charging_time = df['Charging Time (minutes)']

# Manual calculation explanation
print(f"\nManual calculation (for understanding):")
print(f"  Energy (kWh) mean: {energy.mean():.2f}")
print(f"  Charging Time (minutes) mean: {charging_time.mean():.2f}")
print(f"  Using pandas.corr(): {energy.corr(charging_time):.4f}")
print(f"  This means: STRONG positive correlation (0.908)")

print("\n" + "=" * 80)
print("CORRECTED SUMMARY")
print("=" * 80)
print(f"\nTotal columns: {total_columns}")
print(f"Keep: 35-40 columns")
print(f"Remove: {total_columns - 40} to {total_columns - 35} columns")
print(f"  = 31 to 36 columns to remove")

print("\nBreakdown:")
print(f"  - Remove redundant features (highly correlated): ~10-15 columns")
print(f"  - Remove low-value features (low correlation): ~15-20 columns")
print(f"  - Remove non-predictive features (IDs, etc.): ~5-10 columns")
print(f"  Total to remove: ~31-36 columns")

