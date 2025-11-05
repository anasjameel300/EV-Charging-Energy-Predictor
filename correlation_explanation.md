# Correlation Explanation - How I Found It

## Math Correction

You're absolutely right! I made a math error:

- **Total columns**: 71
- **Keep**: 35-40 columns
- **Remove**: 71 - 40 = **31 columns** to 71 - 35 = **36 columns**

**CORRECTION**: We need to remove **31-36 columns**, not 10-15!

---

## How I Calculated Correlation

### Method Used: Pearson Correlation Coefficient

I used pandas' built-in `.corr()` method which calculates the **Pearson Correlation Coefficient**.

### The Code:

```python
# Step 1: Select all numeric columns
numeric_features = df.select_dtypes(include=[np.number]).columns

# Step 2: Calculate correlation matrix
correlation_matrix = df[numeric_features].corr()

# Step 3: This creates a matrix where:
#   - Each row = a feature
#   - Each column = a feature
#   - Each cell = correlation coefficient (-1 to +1)
```

### What `.corr()` Does:

1. Takes each pair of numeric columns
2. Calculates how much they move together
3. Returns a value between -1.0 and +1.0:
   - **+1.0** = Perfect positive relationship (when one goes up, other goes up)
   - **-1.0** = Perfect negative relationship (when one goes up, other goes down)
   - **0.0** = No relationship (they don't move together)

### The Formula (Behind the Scenes):

```
Correlation = Σ((X - X_mean) × (Y - Y_mean)) / √(Σ(X - X_mean)² × Σ(Y - Y_mean)²)
```

Where:
- X = values of first feature
- Y = values of second feature
- X_mean = average of X
- Y_mean = average of Y

### Example:

**Energy (kWh)** and **Charging Time (minutes)**:
- Correlation = 0.908
- This means: When Charging Time increases, Energy usually increases
- 0.908 is close to 1.0, so it's a **STRONG positive relationship**
- This makes sense: longer charging = more energy consumed

---

## What the Correlation Matrix Shows

The correlation matrix is like a table:

| Feature | Energy | Charging Time | Fee | ... |
|---------|--------|---------------|-----|-----|
| Energy | 1.000 | 0.908 | 0.848 | ... |
| Charging Time | 0.908 | 1.000 | 0.750 | ... |
| Fee | 0.848 | 0.750 | 1.000 | ... |
| ... | ... | ... | ... | ... |

- **Diagonal** = 1.000 (a feature is perfectly correlated with itself)
- **Off-diagonal** = correlation between two different features
- **Values close to 1.0 or -1.0** = highly correlated (redundant)
- **Values close to 0.0** = not correlated (independent)

---

## Why This Matters

### High Correlation (>0.8 or <-0.8):
- Two features provide the **same information**
- Example: Energy (kWh) and GHG Savings (kg) = 1.000
  - GHG Savings is calculated directly from Energy
  - They're the same thing!
  - **Solution**: Remove one (keep Energy, remove GHG Savings)

### Low Correlation (<0.3):
- Features are **independent** (don't move together)
- Example: Station Popularity and Energy = 0.148
  - They don't have a strong relationship
  - **Solution**: May not be useful for prediction

### Moderate Correlation (0.3-0.8):
- Features are **related but not redundant**
- Example: User Avg Energy and Energy = 0.760
  - User's past behavior predicts their current behavior
  - **Solution**: Keep both (they provide complementary information)

---

## Visualizations Created

1. **correlation_heatmap_full.png**: Shows the entire correlation matrix as colors
   - Red = positive correlation
   - Blue = negative correlation
   - White = no correlation

2. **correlation_with_energy.png**: Shows correlation of each feature with Energy (kWh)
   - Helps identify which features are important for predicting Energy

3. **top_correlations.png**: Shows the top 30 most correlated pairs
   - Helps identify redundant features to remove

---

## Summary

**How I Found Correlation:**
- Used `pandas.DataFrame.corr()` method
- This calculates Pearson Correlation Coefficient
- Returns a matrix showing relationships between all features

**What It Tells Us:**
- Which features are redundant (high correlation)
- Which features are important (high correlation with target)
- Which features are independent (low correlation)

**Corrected Math:**
- Total columns: 71
- Keep: 35-40 columns
- Remove: **31-36 columns** (not 10-15!)

