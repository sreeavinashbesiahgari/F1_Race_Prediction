# F1 Japanese Grand Prix Prediction

This project predicts the winner of the 2025 Japanese Grand Prix using machine learning techniques and historical Formula 1 data.

## Overview

The prediction model uses:
- Historical race data from similar circuits
- Qualifying times
- Track conditions
- Driver performance metrics

## Features

- Uses FastF1 API to fetch real Formula 1 data
- Implements Gradient Boosting Regressor for predictions
- Considers multiple factors including:
  - Qualifying performance
  - Track temperature
  - Tyre life
  - Historical performance at similar circuits

## Data Sources

The model uses data from:
- 2023 Japanese GP
- 2024 Japanese GP
- Similar high-speed circuits:
  - British GP (Silverstone)
  - Dutch GP (Zandvoort)
  - Italian GP (Monza)
  - Saudi Arabian GP (Jeddah)
  - Bahrain GP

## Qualifying Data

The prediction model uses the following hardcoded qualifying times for the 2025 Japanese GP:

| Driver | Qualifying Time (s) |
|--------|---------------------|
| Max Verstappen | 90.153 |
| Charles Leclerc | 90.254 |
| Lando Norris | 90.355 |
| Carlos Sainz | 90.456 |
| Lewis Hamilton | 90.557 |
| George Russell | 90.658 |
| Oscar Piastri | 90.759 |
| Fernando Alonso | 90.860 |
| Lance Stroll | 90.961 |
| Pierre Gasly | 91.062 |
| Yuki Tsunoda | 91.163 |
| Alexander Albon | 91.264 |

These times are based on:
- Historical performance at Suzuka
- Recent form
- Expected car development
- Track characteristics

## Requirements

- Python 3.x
- Required packages:
  - fastf1
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the prediction script:
```bash
python prediction.py
```

The script will:
1. Load and process historical race data
2. Train the prediction model
3. Output the predicted winner for the 2025 Japanese GP

## Output

The script provides:
- Predicted winner for the 2025 Japanese GP
- Model performance metrics
- Feature importance analysis
- Historical race analysis

## Model Performance

The model's accuracy is measured using Mean Absolute Error (MAE) in seconds. Feature importance analysis helps understand which factors most influence the predictions.

## Visualizations

The script generates several visualizations:
- Feature importance plot
- Predicted vs historical lap times
- Position comparison
- Year-over-year improvement trends
- Driver performance distributions

## Notes

- The prediction is based on historical data and current form
- Real-world factors like weather, mechanical issues, and race incidents are not accounted for
- Results should be interpreted as probabilities rather than certainties