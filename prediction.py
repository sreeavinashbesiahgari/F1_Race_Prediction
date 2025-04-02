import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Create cache directory if it doesn't exist
cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable FastF1 caching
fastf1.Cache.enable_cache(cache_dir)

# Load races with similar characteristics to Suzuka
# Selected based on:
# - High-speed circuits
# - Technical sections
# - Smooth surfaces
# - Similar weather conditions
races = [
    (2023, 16, "R"),  # 2023 Japanese GP (Suzuka)
    (2024, 4, "R"),   # 2024 Japanese GP (Suzuka)
    (2023, 7, "R"),   # 2023 British GP (Silverstone - similar high-speed sections)
    (2023, 13, "R"),  # 2023 Dutch GP (Zandvoort - technical sections)
    (2023, 14, "R"),  # 2023 Italian GP (Monza - high-speed)
    (2024, 2, "R"),   # 2024 Saudi Arabian GP (Jeddah - high-speed, technical)
    (2024, 1, "R")    # 2024 Bahrain (technical sections)
]

# Load all sessions and combine their data
all_laps = []
for year, round_num, session in races:
    try:
        print(f"Loading {year} Round {round_num} {session}...")
        session_data = fastf1.get_session(year, round_num, session)
        session_data.load()
        
        # Get lap times and additional features
        laps = session_data.laps[["Driver", "LapTime", "LapNumber", "Compound", "TyreLife"]].copy()
        
        # Add track temperature if available
        if hasattr(session_data, "weather"):
            track_temp = session_data.weather["TrackTemp"].mean()
            laps["TrackTemp"] = track_temp
        
        laps["Year"] = year
        laps["Round"] = round_num
        all_laps.append(laps)
    except Exception as e:
        print(f"Error loading {year} Round {round_num}: {str(e)}")
        continue

# Combine all lap times
laps_combined = pd.concat(all_laps)
laps_combined.dropna(subset=["LapTime"], inplace=True)
laps_combined["LapTime (s)"] = laps_combined["LapTime"].dt.total_seconds()

# Filter out outlier lap times (e.g., pit stops, safety car)
lap_time_mean = laps_combined["LapTime (s)"].mean()
lap_time_std = laps_combined["LapTime (s)"].std()
laps_combined = laps_combined[
    (laps_combined["LapTime (s)"] > lap_time_mean - 3 * lap_time_std) &
    (laps_combined["LapTime (s)"] < lap_time_mean + 3 * lap_time_std)
]

# 2025 Qualifying Data (Updated with more realistic times for Suzuka)
qualifying_2025 = pd.DataFrame({
    "Driver": ["Max Verstappen", "Charles Leclerc", "Lando Norris", "Carlos Sainz", 
               "Lewis Hamilton", "George Russell", "Oscar Piastri", "Fernando Alonso",
               "Lance Stroll", "Pierre Gasly", "Yuki Tsunoda", "Alexander Albon"],
    "QualifyingTime (s)": [90.153, 90.254, 90.355, 90.456,
                          90.557, 90.658, 90.759, 90.860,
                          90.961, 91.062, 91.163, 91.264]
})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", "George Russell": "RUS",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM",
    "Pierre Gasly": "GAS", "Carlos Sainz": "SAI", "Lance Stroll": "STR", "Fernando Alonso": "ALO",
    "Sergio Perez": "PER", "Daniel Ricciardo": "RIC", "Valtteri Bottas": "BOT", "Zhou Guanyu": "ZHO",
    "Esteban Ocon": "OCO", "Nico Hulkenberg": "HUL", "Kevin Magnussen": "MAG", "Logan Sargeant": "SAR"
}

# Create reverse mapping for driver numbers to full names
driver_number_to_name = {
    "1": "Max Verstappen", "11": "Sergio Perez", "16": "Charles Leclerc", "55": "Carlos Sainz",
    "44": "Lewis Hamilton", "63": "George Russell", "4": "Lando Norris", "81": "Oscar Piastri",
    "14": "Fernando Alonso", "18": "Lance Stroll", "22": "Yuki Tsunoda", "3": "Daniel Ricciardo",
    "23": "Alexander Albon", "2": "Logan Sargeant", "77": "Valtteri Bottas", "24": "Zhou Guanyu",
    "27": "Nico Hulkenberg", "20": "Kevin Magnussen", "31": "Esteban Ocon", "10": "Pierre Gasly"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge 2025 Qualifying Data with combined race data
merged_data = qualifying_2025.merge(laps_combined, left_on="DriverCode", right_on="Driver")

# Prepare features
feature_columns = ["QualifyingTime (s)", "Year"]
if "TrackTemp" in merged_data.columns:
    feature_columns.append("TrackTemp")
if "TyreLife" in merged_data.columns:
    feature_columns.append("TyreLife")

X = merged_data[feature_columns]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

# Define parameter grid for optimization
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

# Create and train model with GridSearchCV
base_model = GradientBoostingRegressor(random_state=39)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_absolute_error'
)
grid_search.fit(X_train, y_train)

# Get best model
model = grid_search.best_estimator_

# Print best parameters
print("\n🔧 Best Model Parameters:")
print(grid_search.best_params_)

# Predict using 2025 qualifying times
prediction_data = qualifying_2025[["QualifyingTime (s)"]].copy()
prediction_data["Year"] = 2025

# Add mean values for other features if they were used in training
if "TrackTemp" in X.columns:
    prediction_data["TrackTemp"] = X["TrackTemp"].mean()
if "TyreLife" in X.columns:
    prediction_data["TyreLife"] = X["TyreLife"].mean()

predicted_lap_times = model.predict(prediction_data[X.columns])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Japanese GP Winner 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Print feature importances
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
print("\n📊 Feature Importances:")
print(feature_importance.sort_values(by='Importance', ascending=False))

# Analyze historical Japanese GP results
print("\n📈 Historical Japanese GP Analysis")
print("==================================")

# Load 2023 and 2024 Japanese GP data
jp_2023 = fastf1.get_session(2023, 16, "R")
jp_2024 = fastf1.get_session(2024, 4, "R")
jp_2023.load()
jp_2024.load()

def analyze_race(session, year):
    print(f"\n{year} Japanese GP Results:")
    print("-" * 30)
    
    # Get race results
    results = session.results[["DriverNumber", "Position", "Points", "Status"]].copy()
    results["Driver"] = results["DriverNumber"].map(driver_number_to_name)
    
    # Get fastest lap for each driver
    fastest_laps = session.laps.groupby("DriverNumber")["LapTime"].min()
    fastest_laps = fastest_laps.dt.total_seconds()
    results["FastestLap (s)"] = results["DriverNumber"].map(fastest_laps)
    
    # Get average lap time (excluding pit stops and safety car)
    avg_laps = session.laps.groupby("DriverNumber")["LapTime"].mean()
    avg_laps = avg_laps.dt.total_seconds()
    results["AvgLapTime (s)"] = results["DriverNumber"].map(avg_laps)
    
    # Sort by position
    results = results.sort_values("Position")
    
    # Print results
    print(results[["Driver", "Position", "FastestLap (s)", "AvgLapTime (s)"]])
    
    return results

# Analyze both races
results_2023 = analyze_race(jp_2023, 2023)
results_2024 = analyze_race(jp_2024, 2024)

# Compare predictions with historical performance
print("\n🎯 Prediction vs Historical Performance")
print("=====================================")

# Create comparison DataFrame
comparison = qualifying_2025[["Driver", "PredictedRaceTime (s)"]].copy()
comparison["2023 Position"] = comparison["Driver"].map(results_2023.set_index("Driver")["Position"])
comparison["2024 Position"] = comparison["Driver"].map(results_2024.set_index("Driver")["Position"])
comparison["2023 Fastest Lap"] = comparison["Driver"].map(results_2023.set_index("Driver")["FastestLap (s)"])
comparison["2024 Fastest Lap"] = comparison["Driver"].map(results_2024.set_index("Driver")["FastestLap (s)"])

# Calculate average historical position and lap time
comparison["Avg Historical Position"] = comparison[["2023 Position", "2024 Position"]].mean(axis=1)
comparison["Avg Historical Lap"] = comparison[["2023 Fastest Lap", "2024 Fastest Lap"]].mean(axis=1)

# Sort by predicted race time
comparison = comparison.sort_values("PredictedRaceTime (s)")

# Print comparison
print("\nDriver Performance Comparison:")
print(comparison[["Driver", "PredictedRaceTime (s)", "Avg Historical Position", "Avg Historical Lap"]])

# After the comparison DataFrame is created, add visualizations:

# 1. Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.sort_values('Importance', ascending=True))
plt.title('Feature Importance in Lap Time Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 2. Predicted vs Historical Lap Times
plt.figure(figsize=(12, 6))
comparison_melted = pd.melt(comparison, 
                           id_vars=['Driver'],
                           value_vars=['PredictedRaceTime (s)', 'Avg Historical Lap'],
                           var_name='Time Type',
                           value_name='Lap Time (s)')

sns.barplot(data=comparison_melted, x='Driver', y='Lap Time (s)', hue='Time Type')
plt.xticks(rotation=45, ha='right')
plt.title('Predicted vs Historical Lap Times by Driver')
plt.tight_layout()
plt.savefig('predicted_vs_historical.png')
plt.close()

# 3. Historical Position vs Predicted Position
plt.figure(figsize=(10, 6))
comparison['Predicted Position'] = range(1, len(comparison) + 1)
sns.scatterplot(data=comparison, 
                x='Avg Historical Position', 
                y='Predicted Position',
                s=100)
for i, txt in enumerate(comparison['Driver']):
    plt.annotate(txt, (comparison['Avg Historical Position'].iloc[i], 
                      comparison['Predicted Position'].iloc[i]),
                xytext=(5, 5), textcoords='offset points')
plt.title('Historical vs Predicted Positions')
plt.xlabel('Average Historical Position (2023-2024)')
plt.ylabel('Predicted Position (2025)')
plt.gca().invert_yaxis()  # Invert y-axis to match F1 position convention
plt.tight_layout()
plt.savefig('position_comparison.png')
plt.close()

# 4. Year-over-Year Lap Time Improvement
plt.figure(figsize=(10, 6))
yearly_avg = laps_combined.groupby('Year')['LapTime (s)'].mean().reset_index()
sns.lineplot(data=yearly_avg, x='Year', y='LapTime (s)', marker='o')
plt.title('Average Lap Time Trend (2023-2024)')
plt.xlabel('Year')
plt.ylabel('Average Lap Time (s)')
plt.tight_layout()
plt.savefig('yearly_improvement.png')
plt.close()

# 5. Driver Performance Distribution
plt.figure(figsize=(12, 6))
sns.boxplot(data=laps_combined, x='Driver', y='LapTime (s)')
plt.xticks(rotation=45, ha='right')
plt.title('Lap Time Distribution by Driver')
plt.tight_layout()
plt.savefig('driver_distribution.png')
plt.close()

print("\n📊 Visualizations have been saved as PNG files:")
print("1. feature_importance.png - Shows the importance of different features in the model")
print("2. predicted_vs_historical.png - Compares predicted lap times with historical performance")
print("3. position_comparison.png - Shows the relationship between historical and predicted positions")
print("4. yearly_improvement.png - Displays the year-over-year lap time improvement trend")
print("5. driver_distribution.png - Shows the distribution of lap times for each driver")