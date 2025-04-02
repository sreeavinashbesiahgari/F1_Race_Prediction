# F1 Japanese Grand Prix Lap Time Predictor

This project uses machine learning to predict lap times for the 2025 Japanese Grand Prix at Suzuka Circuit based on historical F1 data from 2023 and 2024.

## Overview

The predictor analyzes data from multiple F1 races with similar characteristics to Suzuka (high-speed circuits, technical sections, smooth surfaces) to build a machine learning model that forecasts lap times for the 2025 Japanese GP. The model considers factors such as qualifying performance, tire life, and year-over-year improvements.

## Features

- **Data Collection**: Automatically fetches and processes F1 race data using the FastF1 API
- **Machine Learning Model**: Uses GradientBoostingRegressor with optimized hyperparameters
- **Historical Analysis**: Compares predictions with actual results from 2023 and 2024 Japanese GPs
- **Visualizations**: Generates five different plots to visualize predictions and historical performance
- **Performance Metrics**: Calculates Mean Absolute Error (MAE) to evaluate prediction accuracy

## Visualizations

The project generates five visualization files:

1. **Feature Importance Plot** (`feature_importance.png`): Shows the relative importance of different features in the model
2. **Predicted vs Historical Lap Times** (`predicted_vs_historical.png`): Compares predicted lap times with historical performance
3. **Position Comparison Plot** (`position_comparison.png`): Shows the relationship between historical and predicted positions
4. **Yearly Improvement Trend** (`yearly_improvement.png`): Displays the trend of average lap times from 2023 to 2024
5. **Driver Performance Distribution** (`driver_distribution.png`): Shows the distribution of lap times for each driver

## Requirements

- Python 3.8+
- FastF1
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/f1-japanese.git
   cd f1-japanese
   ```

2. Install the required packages:
   ```
   pip install fastf1 pandas numpy scikit-learn matplotlib seaborn
   ```

## Usage

Run the prediction script:
```
python prediction.py
```

The script will:
1. Load and process F1 race data
2. Train the machine learning model
3. Generate predictions for the 2025 Japanese GP
4. Analyze historical performance
5. Create visualizations

## Model Details

The model uses a GradientBoostingRegressor with the following optimized parameters:
- Learning rate: 0.05
- Max depth: 3
- Min samples split: 5
- Number of estimators: 100
- Subsample: 0.8

The most important features in the model are:
- Year (58.8% importance)
- Tyre Life (38.4% importance)
- Qualifying Time (2.8% importance)

## Results

The model achieves a Mean Absolute Error (MAE) of 5.08 seconds, which is reasonable given the complexity of F1 race conditions.

## Future Improvements

- Include weather data as a feature
- Add more historical races for training
- Implement driver-specific performance factors
- Consider track evolution during the race
- Add tire degradation modeling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastF1 library for providing access to F1 timing data
- Formula 1 for the race data
- The F1 community for insights and discussions
