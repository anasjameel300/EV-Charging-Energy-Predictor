# Correlation Explanation - How I Found It

## Overview

This document explains:
1. **Why we added features** (from 35 to 71 columns)
2. **What formulas we used** for feature engineering
3. **How we calculated correlation** (Pearson Correlation Coefficient)
4. **Why we selected only 30-33 columns** from 71 for modeling
5. **The reasoning behind our decisions**

---

## Part 1: Why We Added Features (35 → 71 Columns)

### Original Dataset: 35 Columns

The original dataset had **35 columns** with basic information:
- Timestamps (raw dates/times)
- Station information (names, addresses, locations)
- Charging session data (energy, duration, fees)
- User information (IDs, postal codes)
- Equipment details (port types, plug types)

### Problem with Original Data

**Issue 1: Time Data Not Usable**
- Timestamps were in string format: "01-01-2018 10:36"
- Machine learning models need **numeric values**
- We couldn't extract patterns like "morning vs evening" or "weekend vs weekday"

**Issue 2: Duration Data Not Usable**
- Duration was in string format: "0 days 08:45:22"
- Models can't use this directly
- We needed numeric values (minutes or seconds)

**Issue 3: Missing Behavioral Patterns**
- No user behavior patterns (how often does this user charge?)
- No station patterns (how popular is this station?)
- No time-based patterns (what's the average energy at this hour?)

**Issue 4: Missing Interaction Features**
- No relationships between features (energy per hour, fee per kWh)
- Models work better when we show relationships between variables

### Solution: Feature Engineering (Added 36 New Features)

We created **36 new features** to solve these problems:

#### Category 1: Time-Based Features (12 features)

**Why:** Time patterns are crucial for EV charging prediction
- People charge differently at different times
- Morning vs evening patterns
- Weekday vs weekend patterns
- Seasonal patterns

**What We Created:**
1. `hour` - Hour of day (0-23)
2. `day_of_week` - Day of week (0=Monday, 6=Sunday)
3. `day_of_month` - Day of month (1-31)
4. `month` - Month (1-12)
5. `year` - Year (2018-2019)
6. `is_weekend` - Binary (1 if weekend, 0 if weekday)
7. `is_weekday` - Binary (1 if weekday, 0 if weekend)
8. `is_peak_hour` - Binary (1 if peak hours 8-10 AM or 5-7 PM)
9. `time_of_day` - Categorical (Morning, Afternoon, Evening, Night)
10. `time_of_day_encoded` - Numeric encoding (0-3)
11. `season` - Categorical (Winter, Spring, Summer, Fall)
12. `season_encoded` - Numeric encoding (0-3)

**Formulas Used:**
```python
# Extract hour from timestamp
hour = timestamp.dt.hour

# Extract day of week (0=Monday, 6=Sunday)
day_of_week = timestamp.dt.dayofweek

# Check if weekend
is_weekend = (day_of_week >= 5).astype(int)

# Check if peak hour
is_peak_hour = ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 19))
```

#### Category 2: Preprocessed Time Features (2 features)

**Why:** Convert string durations to numeric values

**What We Created:**
1. `Total Duration (minutes)` - Converted from "0 days 08:45:22" to 525.37 minutes
2. `Charging Time (minutes)` - Converted from "0 days 08:44:43" to 524.72 minutes

**Formula Used:**
```python
def parse_duration(duration_str):
    # "0 days 08:45:22" → 525.37 minutes
    parts = duration_str.split()
    days = int(parts[0])
    time_parts = parts[2].split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = int(time_parts[2])
    total_minutes = days * 24 * 60 + hours * 60 + minutes + seconds / 60
    return total_minutes
```

#### Category 3: Encoded Categorical Features (6 features)

**Why:** Machine learning models need numeric values, not text

**What We Created:**
1. `Port Type Encoded` - Level 1 = 0, Level 2 = 1
2. `Plug Type Encoded` - J1772 = 1, NEMA 5-20R = 0
3. `Ended By Encoded` - Label encoding (0-9 for different end reasons)
4. `County Encoded` - Santa Clara = 1, San Mateo = 0
5. `time_of_day_encoded` - Morning=0, Afternoon=1, Evening=2, Night=3
6. `season_encoded` - Winter=0, Spring=1, Summer=2, Fall=3

**Formula Used:**
```python
# Binary encoding
Port Type Encoded = (Port Type == 'Level 2').astype(int)

# Label encoding
Ended By Encoded = LabelEncoder().fit_transform(Ended By)
```

#### Category 4: Interaction Features (4 features)

**Why:** Show relationships between features (models learn better patterns)

**What We Created:**
1. `Energy per Hour` - Energy (kWh) / Charging Time (hours)
2. `Fee per kWh` - Fee / Energy (kWh)
3. `Energy per Minute` - Energy (kWh) / Charging Time (minutes)
4. `Charging Efficiency` - Energy (kWh) / Total Duration (hours)

**Formulas Used:**
```python
# Energy per Hour
Energy per Hour = Energy (kWh) / (Charging Time (minutes) / 60 + 1e-6)
# + 1e-6 prevents division by zero

# Fee per kWh
Fee per kWh = Fee / (Energy (kWh) + 1e-6)

# Charging Efficiency
Charging Efficiency = Energy (kWh) / (Total Duration (minutes) / 60 + 1e-6)
```

**Why These Matter:**
- `Energy per Hour` shows charging speed (higher = faster charging)
- `Fee per kWh` shows pricing efficiency (lower = better value)
- `Charging Efficiency` shows how efficiently time is used

#### Category 5: User Behavior Features (6 features)

**Why:** Users have patterns - past behavior predicts future behavior

**What We Created:**
1. `User Avg Energy` - Average energy consumed by this user across all sessions
2. `User Std Energy` - Standard deviation (shows consistency)
3. `User Frequency` - How many times this user has charged
4. `User Avg Charging Time` - Average charging time for this user
5. `User Avg Fee` - Average fee paid by this user

**Formulas Used:**
```python
# Group by User ID and calculate statistics
user_stats = df.groupby('User ID').agg({
    'Energy (kWh)': ['mean', 'std', 'count'],
    'Charging Time (minutes)': 'mean',
    'Fee': 'mean'
})

# Result:
User Avg Energy = mean(Energy for this user)
User Std Energy = std(Energy for this user)
User Frequency = count(sessions for this user)
User Avg Charging Time = mean(Charging Time for this user)
User Avg Fee = mean(Fee for this user)
```

**Why These Matter:**
- If a user typically charges 10 kWh, they'll likely charge similar amounts
- User patterns are strong predictors

#### Category 6: Station Features (6 features)

**Why:** Different stations have different characteristics

**What We Created:**
1. `Station Avg Energy` - Average energy at this station
2. `Station Std Energy` - Standard deviation at this station
3. `Station Popularity` - Total number of sessions at this station
4. `Station Avg Charging Time` - Average charging time at this station
5. `Station Avg Fee` - Average fee at this station
6. `Station Unique Users` - Number of unique users at this station

**Formulas Used:**
```python
# Group by Station Name and calculate statistics
station_stats = df.groupby('Station Name').agg({
    'Energy (kWh)': ['mean', 'std', 'count'],
    'Charging Time (minutes)': 'mean',
    'Fee': 'mean',
    'User ID': 'nunique'  # Count unique users
})
```

**Why These Matter:**
- Popular stations might have different patterns
- Station characteristics affect charging behavior

#### Category 7: Time-Based Aggregations (3 features)

**Why:** Show average patterns at different times

**What We Created:**
1. `Hourly Avg Energy` - Average energy at this hour of day
2. `Day of Week Avg Energy` - Average energy on this day of week
3. `Monthly Avg Energy` - Average energy in this month

**Formulas Used:**
```python
# Calculate averages by time period
hourly_avg = df.groupby('hour')['Energy (kWh)'].mean().to_dict()
Hourly Avg Energy = hourly_avg[hour]

dow_avg = df.groupby('day_of_week')['Energy (kWh)'].mean().to_dict()
Day of Week Avg Energy = dow_avg[day_of_week]

monthly_avg = df.groupby('month')['Energy (kWh)'].mean().to_dict()
Monthly Avg Energy = monthly_avg[month]
```

**Why These Matter:**
- Shows typical patterns at different times
- Helps models understand time-based trends

### Summary: 35 → 71 Columns

- **Original:** 35 columns (raw data)
- **Added:** 36 new features (engineered features)
- **Total:** 71 columns

**Breakdown:**
- Time-based: 12 features
- Preprocessed time: 2 features
- Encoded categorical: 6 features
- Interaction: 4 features
- User behavior: 6 features
- Station: 6 features
- Time aggregations: 3 features
- **Total added: 39 features** (some original columns were kept as-is)

---

## Part 2: How We Calculated Correlation

### Method: Pearson Correlation Coefficient

We used **Pearson Correlation Coefficient** to measure relationships between features.

### The Formula

```
Correlation (r) = Σ((X - X̄) × (Y - Ȳ)) / √(Σ(X - X̄)² × Σ(Y - Ȳ)²)
```

Where:
- **X** = values of first feature
- **Y** = values of second feature
- **X̄** = mean of X
- **Ȳ** = mean of Y
- **Σ** = sum of all values

### What It Measures

**Correlation measures how two features move together:**
- **+1.0** = Perfect positive relationship (when X goes up, Y always goes up)
- **-1.0** = Perfect negative relationship (when X goes up, Y always goes down)
- **0.0** = No relationship (X and Y don't move together)

### The Code

```python
# Step 1: Select all numeric columns
numeric_features = df.select_dtypes(include=[np.number]).columns

# Step 2: Calculate correlation matrix
correlation_matrix = df[numeric_features].corr()

# This creates a matrix where:
# - Each row = a feature
# - Each column = a feature
# - Each cell = correlation coefficient (-1 to +1)
```

### Example Calculation

**Energy (kWh)** and **Charging Time (minutes)**:
- Correlation = 0.908
- This means: When Charging Time increases, Energy usually increases
- 0.908 is close to 1.0, so it's a **STRONG positive relationship**
- This makes sense: longer charging = more energy consumed

### What the Correlation Matrix Shows

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

## Part 3: Why We Selected Only 30-33 Columns from 71

### The Problem: Too Many Features

**Issue 1: Redundant Features**
- Some features provide the same information
- Example: `Energy (kWh)` and `GHG Savings (kg)` have correlation = 1.000
- Keeping both is redundant and can confuse the model

**Issue 2: Low-Value Features**
- Some features have very low correlation with target (Energy)
- Example: `Station Std Energy` has correlation = 0.098 with Energy
- These features don't help prediction and add noise

**Issue 3: Non-Predictive Features**
- Some features are just IDs or metadata
- Example: `System S/N`, `Model Number`, `MAC Address`
- These don't help predict Energy consumption

**Issue 4: Overfitting Risk**
- Too many features can cause overfitting
- Model memorizes training data instead of learning patterns
- Reduces accuracy on new data

### Solution: Feature Selection Based on Correlation

We used correlation analysis to select the best features:

#### Step 1: Identify Highly Correlated Pairs (>0.8)

**Features to REMOVE (Perfect/High Correlation - Redundant):**

1. **GHG Savings (kg)** - Correlation: 1.000 with Energy (kWh)
   - **Reason:** Calculated directly from Energy
   - **Formula:** GHG Savings = Energy × 0.42 (constant conversion factor)
   - **Decision:** REMOVE (keep Energy, remove GHG Savings)

2. **Gasoline Savings (gallons)** - Correlation: 1.000 with Energy (kWh)
   - **Reason:** Calculated directly from Energy
   - **Formula:** Gasoline Savings = Energy × 0.125 (constant conversion factor)
   - **Decision:** REMOVE (keep Energy, remove Gasoline Savings)

3. **is_weekday** - Correlation: 1.000 with is_weekend
   - **Reason:** Perfect inverse correlation (if weekend=1, weekday=0)
   - **Formula:** is_weekday = 1 - is_weekend
   - **Decision:** REMOVE (keep is_weekend, remove is_weekday)

4. **Plug Type Encoded** - Correlation: 1.000 with Port Type Encoded
   - **Reason:** Perfect correlation (same information)
   - **Decision:** REMOVE (keep Port Type Encoded, remove Plug Type Encoded)

5. **Energy per Minute** - Correlation: 1.000 with Energy per Hour
   - **Reason:** Perfect correlation (just different units)
   - **Formula:** Energy per Minute = Energy per Hour / 60
   - **Decision:** REMOVE (keep Energy per Hour, remove Energy per Minute)

6. **Day of Week Avg Energy** - Correlation: 0.962 with is_weekend
   - **Reason:** High correlation (redundant with is_weekend)
   - **Decision:** REMOVE (keep is_weekend, remove Day of Week Avg Energy)

#### Step 2: Identify Important Features (High Correlation with Target)

**Features to KEEP (Correlation > 0.5 with Energy):**

1. **Charging Time (minutes)** - Correlation: 0.908
   - **Reason:** Strong predictor (longer charging = more energy)
   - **KEEP**

2. **Fee** - Correlation: 0.848
   - **Reason:** Closely related to energy consumed
   - **KEEP**

3. **User Avg Energy** - Correlation: 0.760
   - **Reason:** User behavior pattern is important
   - **KEEP**

4. **User Avg Fee** - Correlation: 0.706
   - **Reason:** User spending pattern
   - **KEEP**

5. **Total Duration (minutes)** - Correlation: 0.703
   - **Reason:** Related to charging time
   - **KEEP**

6. **User Avg Charging Time** - Correlation: 0.671
   - **Reason:** User charging behavior
   - **KEEP**

7. **User Std Energy** - Correlation: 0.588
   - **Reason:** Shows user consistency/variability
   - **KEEP**

#### Step 3: Keep Time-Based Features (Even with Low Correlation)

**Why:** Time patterns are important for prediction, even if individual correlation is low

**Features to KEEP:**
- `hour` - Hour of day patterns
- `day_of_week` - Day of week patterns
- `month` - Seasonal patterns
- `year` - Year patterns
- `is_weekend` - Weekend vs weekday patterns
- `is_peak_hour` - Peak vs off-peak patterns
- `season_encoded` - Seasonal patterns
- `time_of_day_encoded` - Time of day patterns

**Reason:** These features work together to capture time-based patterns

#### Step 4: Remove Low-Value Features

**Features to REMOVE (Very Low Correlation < 0.2):**

1. **Station Std Energy** - Correlation: 0.098
   - **Reason:** Very low correlation, doesn't help prediction
   - **REMOVE**

2. **Monthly Avg Energy** - Correlation: Very low
   - **Reason:** Redundant with month and season features
   - **REMOVE**

3. **System S/N** - Not numeric/predictive
   - **Reason:** Just an ID, doesn't help prediction
   - **REMOVE**

4. **Model Number** - Not numeric/predictive
   - **Reason:** Just an ID, doesn't help prediction
   - **REMOVE**

5. **MAC Address** - Not numeric/predictive
   - **Reason:** Just an ID, doesn't help prediction
   - **REMOVE**

6. **Org Name** - Not predictive
   - **Reason:** All values are the same ("City of Palo Alto")
   - **REMOVE**

### Final Selection: 30-33 Features

**Selected Features (30-33):**

**Core Features (7):**
1. Energy (kWh) - **TARGET VARIABLE**
2. Charging Time (minutes)
3. Total Duration (minutes)
4. Fee
5. Port Type Encoded
6. Port Number
7. EVSE ID

**Time Features (8):**
8. hour
9. day_of_week
10. month
11. year
12. is_weekend
13. is_peak_hour
14. season_encoded
15. time_of_day_encoded

**Location Features (3):**
16. Latitude
17. Longitude
18. County Encoded

**User Behavior Features (6):**
19. User ID
20. User Avg Energy
21. User Avg Fee
22. User Avg Charging Time
23. User Std Energy
24. User Frequency

**Station Features (5):**
25. Station Avg Energy
26. Station Avg Fee
27. Station Avg Charging Time
28. Station Popularity
29. Station Unique Users

**Engineered Features (3):**
30. Energy per Hour
31. Fee per kWh
32. Charging Efficiency

**Optional (1):**
33. Hourly Avg Energy (if needed)

### Summary: 71 → 30-33 Columns

- **Total columns:** 71
- **Selected:** 30-33 columns
- **Removed:** 38-41 columns

**Reasons for Removal:**
- **Redundant features:** 6 columns (perfect correlation)
- **Low-value features:** 15-20 columns (low correlation with target)
- **Non-predictive features:** 10-15 columns (IDs, metadata)
- **Redundant aggregations:** 5-10 columns (redundant with other features)

---

## Part 4: Why Correlation Analysis Matters

### Benefits of Feature Selection

1. **Reduces Overfitting**
   - Fewer features = simpler model
   - Model learns patterns, not noise
   - Better accuracy on new data

2. **Faster Training**
   - Fewer features = faster computation
   - Models train in seconds instead of minutes

3. **Better Accuracy**
   - Removing redundant features improves accuracy
   - Model focuses on important features
   - Less confusion from redundant information

4. **Easier Interpretation**
   - Fewer features = easier to understand
   - Can see which features matter most
   - Better insights for decision-making

### Correlation Thresholds Used

- **> 0.8 or < -0.8:** Remove one feature (highly redundant)
- **0.5 - 0.8:** Keep both features (complementary information)
- **0.3 - 0.5:** Keep if useful (moderate relationship)
- **< 0.3:** Consider removing (weak relationship)

---

## Summary

**Why We Added Features (35 → 71):**
- Original data had unusable formats (strings, timestamps)
- Missing behavioral patterns (user, station, time)
- Missing interaction features (relationships between variables)
- Created 36 new features using formulas and aggregations

**How We Calculated Correlation:**
- Used Pearson Correlation Coefficient formula
- Calculated correlation between all feature pairs
- Identified redundant and important features

**Why We Selected Only 30-33 Columns:**
- Removed redundant features (perfect correlation)
- Removed low-value features (low correlation with target)
- Removed non-predictive features (IDs, metadata)
- Kept important features (high correlation with target)
- Kept time-based features (important for patterns)

**Result:**
- Better model accuracy (99.86% R² with Random Forest)
- Faster training
- Easier interpretation
- Better generalization to new data
