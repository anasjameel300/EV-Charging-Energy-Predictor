# Feature Analysis Summary - EV Charging Dataset

## Overview
- **Original Dataset**: 35 columns, 102,781 rows
- **Preprocessed Dataset**: 71 columns (added 36 new features), 102,781 rows
- **Correlation Analysis**: 43 numeric features analyzed

---

## HIGHLY CORRELATED PAIRS (>0.8) - REDUNDANT FEATURES

### Features to REMOVE (Perfect/High Correlation - Redundant):

1. **GHG Savings (kg)** - Correlation: 1.000 with Energy (kWh)
   - **Decision**: REMOVE (calculated directly from Energy)
   - **Reason**: Perfect correlation, adds no new information

2. **Gasoline Savings (gallons)** - Correlation: 1.000 with Energy (kWh)
   - **Decision**: REMOVE (calculated directly from Energy)
   - **Reason**: Perfect correlation, adds no new information

3. **is_weekday** - Correlation: 1.000 with is_weekend
   - **Decision**: REMOVE (keep is_weekend)
   - **Reason**: Perfect inverse correlation, redundant

4. **Plug Type Encoded** - Correlation: 1.000 with Port Type Encoded
   - **Decision**: REMOVE (keep Port Type Encoded)
   - **Reason**: Perfect correlation, same information

5. **Energy per Minute** - Correlation: 1.000 with Energy per Hour
   - **Decision**: REMOVE (keep Energy per Hour)
   - **Reason**: Perfect correlation, just different units

6. **Day of Week Avg Energy** - Correlation: 0.962 with is_weekend
   - **Decision**: REMOVE (redundant with is_weekend)
   - **Reason**: High correlation, redundant information

---

## FEATURES TO KEEP - BASED ON CORRELATION WITH ENERGY (kWh)

### VERY IMPORTANT (Correlation > 0.5):

1. **Charging Time (minutes)** - Correlation: 0.908
   - **KEEP** - Strong predictor of energy consumption

2. **Fee** - Correlation: 0.848
   - **KEEP** - Closely related to energy consumed

3. **User Avg Energy** - Correlation: 0.760
   - **KEEP** - User behavior pattern is important

4. **User Avg Fee** - Correlation: 0.706
   - **KEEP** - User spending pattern

5. **Total Duration (minutes)** - Correlation: 0.703
   - **KEEP** - Related to charging time

6. **User Avg Charging Time** - Correlation: 0.671
   - **KEEP** - User charging behavior

7. **User Std Energy** - Correlation: 0.588
   - **KEEP** - Shows user consistency/variability

### MODERATELY IMPORTANT (Correlation 0.3-0.5):

8. **Charging Efficiency** - Correlation: 0.370
   - **KEEP** - Useful interaction feature

### LESS IMPORTANT BUT STILL USEFUL (Correlation 0.2-0.3):

9. **Energy per Hour** - Correlation: 0.294
   - **KEEP** - Useful interaction feature

10. **Station Avg Energy** - Correlation: 0.148
    - **KEEP** - Station characteristics

11. **Hourly Avg Energy** - Correlation: 0.141
    - **KEEP** - Time-based pattern

12. **Station Avg Fee** - Correlation: 0.137
    - **KEEP** - Station pricing pattern

13. **Station Avg Charging Time** - Correlation: 0.136
    - **KEEP** - Station characteristics

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

---

## LOCATION FEATURES TO KEEP

1. **Latitude** - KEEP (may show location-based patterns)
2. **Longitude** - KEEP (may show location-based patterns)
3. **County Encoded** - KEEP (location category)

---

## USER AND STATION FEATURES TO KEEP

1. **User ID** - KEEP (for user behavior analysis)
2. **User Frequency** - KEEP (how often user charges)
3. **Station Popularity** - KEEP (how busy the station is)
4. **Station Unique Users** - KEEP (station diversity)
5. **Port Type Encoded** - KEEP (Level 1 vs Level 2)
6. **Port Number** - KEEP (which port at station)
7. **EVSE ID** - KEEP (specific equipment ID)

---

## FEATURES TO REMOVE (Low Correlation + Redundant)

### Definitely Remove:
1. GHG Savings (kg) - Perfect correlation with Energy
2. Gasoline Savings (gallons) - Perfect correlation with Energy
3. is_weekday - Perfect correlation with is_weekend
4. Plug Type Encoded - Perfect correlation with Port Type Encoded
5. Energy per Minute - Perfect correlation with Energy per Hour
6. Day of Week Avg Energy - High correlation with is_weekend

### Consider Removing (Very Low Correlation):
1. Station Std Energy - Very low correlation (0.098)
2. Monthly Avg Energy - Very low correlation
3. System S/N - Not useful for prediction
4. Model Number - Not useful for prediction
5. MAC Address - Not useful for prediction
6. Org Name - Not useful for prediction (all same)

---

## RECOMMENDED FINAL FEATURE SET

### Core Features (Must Keep):
- Energy (kWh) - **TARGET VARIABLE**
- Charging Time (minutes)
- Total Duration (minutes)
- Fee
- Port Type Encoded
- Port Number
- EVSE ID

### Time Features (Must Keep):
- hour
- day_of_week
- month
- year
- is_weekend
- is_peak_hour
- season_encoded
- time_of_day_encoded

### Location Features (Keep):
- Latitude
- Longitude
- County Encoded

### User Behavior Features (Keep):
- User ID
- User Avg Energy
- User Avg Fee
- User Avg Charging Time
- User Std Energy
- User Frequency

### Station Features (Keep):
- Station Name (or Station Popularity)
- Station Avg Energy
- Station Avg Fee
- Station Avg Charging Time
- Station Popularity
- Station Unique Users

### Engineered Features (Keep):
- Energy per Hour
- Fee per kWh
- Charging Efficiency

### Remove:
- GHG Savings (kg)
- Gasoline Savings (gallons)
- is_weekday
- Plug Type Encoded
- Energy per Minute
- Day of Week Avg Energy
- Station Std Energy
- Monthly Avg Energy
- System S/N
- Model Number
- MAC Address
- Org Name
- Hourly Avg Energy (if redundant)

---

## SUMMARY

**Total Features to Keep**: ~35-40 features
**Total Features to Remove**: ~10-15 features

**Key Insight**: 
- Energy consumption is highly correlated with Charging Time, Fee, and User behavior patterns
- Time-based features are important for capturing patterns
- Station and location features provide context
- Calculated features (GHG Savings, Gasoline Savings) are redundant

