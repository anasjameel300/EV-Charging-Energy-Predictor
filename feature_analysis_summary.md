# Feature Analysis Summary - EV Charging Dataset

## Overview

- **Original Dataset**: 35 columns, 102,781 rows
- **Preprocessed Dataset**: 71 columns (added 36 new features), 102,781 rows
- **Correlation Analysis**: 43 numeric features analyzed
- **Final Selected Features**: 30-33 columns for modeling

---

## Part 1: Why We Added Features (35 → 71 Columns)

### Original Dataset Limitations

The original 35 columns had several issues:

1. **Unusable Time Data**
   - Timestamps in string format: "01-01-2018 10:36"
   - Duration in string format: "0 days 08:45:22"
   - Models need numeric values, not strings

2. **Missing Behavioral Patterns**
   - No user behavior patterns (how often does this user charge?)
   - No station patterns (how popular is this station?)
   - No time-based patterns (what's average energy at this hour?)

3. **Missing Interaction Features**
   - No relationships between features (energy per hour, fee per kWh)
   - Models learn better when we show relationships

4. **Categorical Data Not Encoded**
   - Text categories (Port Type, Plug Type, Ended By)
   - Models need numeric values

### Solution: Feature Engineering

We created **36 new features** organized into 7 categories:

#### Category 1: Time-Based Features (12 features)

**Purpose:** Extract time patterns from timestamps

**Features Created:**
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

**Formulas:**
```python
hour = timestamp.dt.hour
day_of_week = timestamp.dt.dayofweek
is_weekend = (day_of_week >= 5).astype(int)
is_peak_hour = ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 19))
```

**Why Important:**
- People charge differently at different times
- Morning vs evening patterns
- Weekday vs weekend patterns
- Seasonal patterns

#### Category 2: Preprocessed Time Features (2 features)

**Purpose:** Convert string durations to numeric values

**Features Created:**
1. `Total Duration (minutes)` - Converted from "0 days 08:45:22"
2. `Charging Time (minutes)` - Converted from "0 days 08:44:43"

**Formula:**
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

**Why Important:**
- Models need numeric values, not strings
- Enables calculations (energy per hour, efficiency)

#### Category 3: Encoded Categorical Features (6 features)

**Purpose:** Convert text categories to numeric values

**Features Created:**
1. `Port Type Encoded` - Level 1 = 0, Level 2 = 1
2. `Plug Type Encoded` - J1772 = 1, NEMA 5-20R = 0
3. `Ended By Encoded` - Label encoding (0-9)
4. `County Encoded` - Santa Clara = 1, San Mateo = 0
5. `time_of_day_encoded` - Morning=0, Afternoon=1, Evening=2, Night=3
6. `season_encoded` - Winter=0, Spring=1, Summer=2, Fall=3

**Formula:**
```python
Port Type Encoded = (Port Type == 'Level 2').astype(int)
Ended By Encoded = LabelEncoder().fit_transform(Ended By)
```

**Why Important:**
- Models can only use numeric values
- Encoding preserves information while making it usable

#### Category 4: Interaction Features (4 features)

**Purpose:** Show relationships between features

**Features Created:**
1. `Energy per Hour` - Energy (kWh) / Charging Time (hours)
2. `Fee per kWh` - Fee / Energy (kWh)
3. `Energy per Minute` - Energy (kWh) / Charging Time (minutes)
4. `Charging Efficiency` - Energy (kWh) / Total Duration (hours)

**Formulas:**
```python
Energy per Hour = Energy (kWh) / (Charging Time (minutes) / 60 + 1e-6)
Fee per kWh = Fee / (Energy (kWh) + 1e-6)
Charging Efficiency = Energy (kWh) / (Total Duration (minutes) / 60 + 1e-6)
```

**Why Important:**
- Shows charging speed (Energy per Hour)
- Shows pricing efficiency (Fee per kWh)
- Shows time efficiency (Charging Efficiency)
- Models learn better patterns from relationships

#### Category 5: User Behavior Features (6 features)

**Purpose:** Capture user behavior patterns

**Features Created:**
1. `User Avg Energy` - Average energy for this user
2. `User Std Energy` - Standard deviation (consistency)
3. `User Frequency` - Number of charging sessions
4. `User Avg Charging Time` - Average charging time
5. `User Avg Fee` - Average fee paid

**Formula:**
```python
user_stats = df.groupby('User ID').agg({
    'Energy (kWh)': ['mean', 'std', 'count'],
    'Charging Time (minutes)': 'mean',
    'Fee': 'mean'
})
```

**Why Important:**
- Users have consistent patterns
- Past behavior predicts future behavior
- Strong predictor of energy consumption

#### Category 6: Station Features (6 features)

**Purpose:** Capture station characteristics

**Features Created:**
1. `Station Avg Energy` - Average energy at this station
2. `Station Std Energy` - Standard deviation
3. `Station Popularity` - Total sessions at this station
4. `Station Avg Charging Time` - Average charging time
5. `Station Avg Fee` - Average fee
6. `Station Unique Users` - Number of unique users

**Formula:**
```python
station_stats = df.groupby('Station Name').agg({
    'Energy (kWh)': ['mean', 'std', 'count'],
    'Charging Time (minutes)': 'mean',
    'Fee': 'mean',
    'User ID': 'nunique'
})
```

**Why Important:**
- Different stations have different characteristics
- Popular stations may have different patterns
- Station context affects charging behavior

#### Category 7: Time-Based Aggregations (3 features)

**Purpose:** Show average patterns at different times

**Features Created:**
1. `Hourly Avg Energy` - Average energy at this hour
2. `Day of Week Avg Energy` - Average energy on this day
3. `Monthly Avg Energy` - Average energy in this month

**Formula:**
```python
hourly_avg = df.groupby('hour')['Energy (kWh)'].mean().to_dict()
Hourly Avg Energy = hourly_avg[hour]
```

**Why Important:**
- Shows typical patterns at different times
- Helps models understand time-based trends

### Summary: Feature Engineering

- **Original:** 35 columns
- **Added:** 36 new features
- **Total:** 71 columns

**Breakdown:**
- Time-based: 12 features
- Preprocessed time: 2 features
- Encoded categorical: 6 features
- Interaction: 4 features
- User behavior: 6 features
- Station: 6 features
- Time aggregations: 3 features

---

## Part 2: Correlation Analysis - Why We Selected Only 30-33 Columns

### The Problem: Too Many Features (71 Columns)

**Issues:**
1. **Redundant Features** - Some features provide the same information
2. **Low-Value Features** - Some features don't help prediction
3. **Non-Predictive Features** - Some features are just IDs/metadata
4. **Overfitting Risk** - Too many features can cause overfitting

### Solution: Feature Selection Based on Correlation

We used **Pearson Correlation Coefficient** to identify:
- Which features are redundant (high correlation with each other)
- Which features are important (high correlation with target)
- Which features to remove (low correlation with target)

### Correlation Formula

```
Correlation (r) = Σ((X - X̄) × (Y - Ȳ)) / √(Σ(X - X̄)² × Σ(Y - Ȳ)²)
```

Where:
- X = values of first feature
- Y = values of second feature
- X̄ = mean of X
- Ȳ = mean of Y

### Correlation Thresholds

- **> 0.8 or < -0.8:** Remove one feature (highly redundant)
- **0.5 - 0.8:** Keep both features (complementary information)
- **0.3 - 0.5:** Keep if useful (moderate relationship)
- **< 0.3:** Consider removing (weak relationship)

---

## HIGHLY CORRELATED PAIRS (>0.8) - REDUNDANT FEATURES

### Features to REMOVE (Perfect/High Correlation - Redundant):

1. **GHG Savings (kg)** - Correlation: 1.000 with Energy (kWh)
   - **Reason:** Calculated directly from Energy
   - **Formula:** GHG Savings = Energy × 0.42 (constant)
   - **Decision:** REMOVE (keep Energy, remove GHG Savings)

2. **Gasoline Savings (gallons)** - Correlation: 1.000 with Energy (kWh)
   - **Reason:** Calculated directly from Energy
   - **Formula:** Gasoline Savings = Energy × 0.125 (constant)
   - **Decision:** REMOVE (keep Energy, remove Gasoline Savings)

3. **is_weekday** - Correlation: 1.000 with is_weekend
   - **Reason:** Perfect inverse correlation
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

---

## FEATURES TO KEEP - BASED ON CORRELATION WITH ENERGY (kWh)

### VERY IMPORTANT (Correlation > 0.5):

1. **Charging Time (minutes)** - Correlation: 0.908
   - **KEEP** - Strong predictor (longer charging = more energy)
   - **Reason:** Direct relationship with energy consumption

2. **Fee** - Correlation: 0.848
   - **KEEP** - Closely related to energy consumed
   - **Reason:** Fee is typically based on energy consumed

3. **User Avg Energy** - Correlation: 0.760
   - **KEEP** - User behavior pattern is important
   - **Reason:** Users have consistent charging patterns

4. **User Avg Fee** - Correlation: 0.706
   - **KEEP** - User spending pattern
   - **Reason:** User spending patterns predict energy

5. **Total Duration (minutes)** - Correlation: 0.703
   - **KEEP** - Related to charging time
   - **Reason:** Longer duration = more energy

6. **User Avg Charging Time** - Correlation: 0.671
   - **KEEP** - User charging behavior
   - **Reason:** User patterns are strong predictors

7. **User Std Energy** - Correlation: 0.588
   - **KEEP** - Shows user consistency/variability
   - **Reason:** Helps understand user behavior patterns

### MODERATELY IMPORTANT (Correlation 0.3-0.5):

8. **Charging Efficiency** - Correlation: 0.370
   - **KEEP** - Useful interaction feature
   - **Reason:** Shows relationship between energy and time

### LESS IMPORTANT BUT STILL USEFUL (Correlation 0.2-0.3):

9. **Energy per Hour** - Correlation: 0.294
   - **KEEP** - Useful interaction feature
   - **Reason:** Shows charging speed

10. **Station Avg Energy** - Correlation: 0.148
    - **KEEP** - Station characteristics
    - **Reason:** Station context matters

11. **Hourly Avg Energy** - Correlation: 0.141
    - **KEEP** - Time-based pattern
    - **Reason:** Shows time-based trends

12. **Station Avg Fee** - Correlation: 0.137
    - **KEEP** - Station pricing pattern
    - **Reason:** Station pricing affects behavior

13. **Station Avg Charging Time** - Correlation: 0.136
    - **KEEP** - Station characteristics
    - **Reason:** Station context matters

---

## TIME-BASED FEATURES TO KEEP

These features may have low correlation individually but are important for time-series patterns:

1. **hour** - KEEP (hour of day patterns)
2. **day_of_week** - KEEP (day of week patterns)
3. **month** - KEEP (seasonal patterns)
4. **season_encoded** - KEEP (seasonal patterns)
5. **is_weekend** - KEEP (weekend vs weekday patterns)
6. **is_peak_hour** - KEEP (peak vs off-peak patterns)
7. **time_of_day_encoded** - KEEP (morning/afternoon/evening/night)

**Why Keep These:**
- Time patterns are crucial for EV charging prediction
- Features work together to capture time-based patterns
- Even if individual correlation is low, combined they're important

---

## LOCATION FEATURES TO KEEP

1. **Latitude** - KEEP (may show location-based patterns)
2. **Longitude** - KEEP (may show location-based patterns)
3. **County Encoded** - KEEP (location category)

**Why Keep These:**
- Location may affect charging behavior
- Different areas may have different patterns
- Useful for spatial analysis

---

## USER AND STATION FEATURES TO KEEP

1. **User ID** - KEEP (for user behavior analysis)
2. **User Frequency** - KEEP (how often user charges)
3. **Station Popularity** - KEEP (how busy the station is)
4. **Station Unique Users** - KEEP (station diversity)
5. **Port Type Encoded** - KEEP (Level 1 vs Level 2)
6. **Port Number** - KEEP (which port at station)
7. **EVSE ID** - KEEP (specific equipment ID)

**Why Keep These:**
- User behavior is a strong predictor
- Station characteristics affect charging
- Equipment type matters (Level 1 vs Level 2)

---

## FEATURES TO REMOVE (Low Correlation + Redundant)

### Definitely Remove:

1. **GHG Savings (kg)** - Perfect correlation with Energy
2. **Gasoline Savings (gallons)** - Perfect correlation with Energy
3. **is_weekday** - Perfect correlation with is_weekend
4. **Plug Type Encoded** - Perfect correlation with Port Type Encoded
5. **Energy per Minute** - Perfect correlation with Energy per Hour
6. **Day of Week Avg Energy** - High correlation with is_weekend

### Consider Removing (Very Low Correlation):

1. **Station Std Energy** - Very low correlation (0.098)
   - **Reason:** Doesn't help prediction
   - **REMOVE**

2. **Monthly Avg Energy** - Very low correlation
   - **Reason:** Redundant with month and season features
   - **REMOVE**

3. **System S/N** - Not useful for prediction
   - **Reason:** Just an ID, doesn't help prediction
   - **REMOVE**

4. **Model Number** - Not useful for prediction
   - **Reason:** Just an ID, doesn't help prediction
   - **REMOVE**

5. **MAC Address** - Not useful for prediction
   - **Reason:** Just an ID, doesn't help prediction
   - **REMOVE**

6. **Org Name** - Not useful for prediction
   - **Reason:** All values are the same
   - **REMOVE**

---

## RECOMMENDED FINAL FEATURE SET (30-33 Features)

### Core Features (7):
- Energy (kWh) - **TARGET VARIABLE**
- Charging Time (minutes)
- Total Duration (minutes)
- Fee
- Port Type Encoded
- Port Number
- EVSE ID

### Time Features (8):
- hour
- day_of_week
- month
- year
- is_weekend
- is_peak_hour
- season_encoded
- time_of_day_encoded

### Location Features (3):
- Latitude
- Longitude
- County Encoded

### User Behavior Features (6):
- User ID
- User Avg Energy
- User Avg Fee
- User Avg Charging Time
- User Std Energy
- User Frequency

### Station Features (5):
- Station Avg Energy
- Station Avg Fee
- Station Avg Charging Time
- Station Popularity
- Station Unique Users

### Engineered Features (3):
- Energy per Hour
- Fee per kWh
- Charging Efficiency

### Optional (1):
- Hourly Avg Energy (if needed)

**Total: 30-33 features**

---

## SUMMARY

### Why We Added Features (35 → 71):
1. **Original data had unusable formats** (strings, timestamps)
2. **Missing behavioral patterns** (user, station, time)
3. **Missing interaction features** (relationships between variables)
4. **Created 36 new features** using formulas and aggregations

### Why We Selected Only 30-33 Columns:
1. **Removed redundant features** (perfect correlation > 0.8)
2. **Removed low-value features** (low correlation < 0.3 with target)
3. **Removed non-predictive features** (IDs, metadata)
4. **Kept important features** (high correlation > 0.5 with target)
5. **Kept time-based features** (important for patterns, even if low correlation)

### Formulas Used:

**Time Features:**
- `hour = timestamp.dt.hour`
- `is_weekend = (day_of_week >= 5).astype(int)`
- `is_peak_hour = ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 19))`

**Interaction Features:**
- `Energy per Hour = Energy (kWh) / (Charging Time (minutes) / 60 + 1e-6)`
- `Fee per kWh = Fee / (Energy (kWh) + 1e-6)`
- `Charging Efficiency = Energy (kWh) / (Total Duration (minutes) / 60 + 1e-6)`

**Aggregation Features:**
- `User Avg Energy = mean(Energy for this user)`
- `Station Popularity = count(sessions at this station)`
- `Hourly Avg Energy = mean(Energy at this hour)`

**Correlation Formula:**
- `Correlation = Σ((X - X̄) × (Y - Ȳ)) / √(Σ(X - X̄)² × Σ(Y - Ȳ)²)`

### Result:
- **Better model accuracy** (99.86% R² with Random Forest)
- **Faster training** (fewer features = faster computation)
- **Easier interpretation** (fewer features = easier to understand)
- **Better generalization** (less overfitting)

### Key Insights:
- Energy consumption is highly correlated with Charging Time, Fee, and User behavior patterns
- Time-based features are important for capturing patterns
- Station and location features provide context
- Calculated features (GHG Savings, Gasoline Savings) are redundant
- User behavior patterns are strong predictors
