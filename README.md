# Hierarchical Time Series Forecasting for FMCG Items

## Project Overview

This project aims to perform hierarchical time series forecasting for FMCG (Fast-Moving Consumer Goods) items at various levels such as country, state, division, district, zone, and route. The project utilizes multiple forecasting models, including ARIMA, Prophet, XGBoost, and LightGBM, and performs hierarchical reconciliation using various methods.

## Folder Structure

```
hierarchical_time_series/
├── config/
│ └── config.yaml
├── data/
│ ├── raw/
│ │ └── fmcg_data.csv
│ └── processed/
├── src/
│ ├── init.py
│ ├── preprocess.py
│ ├── feature_engineering.py
│ ├── model.py
│ ├── reconciliation.py
│ ├── evaluation.py
│ ├── utils.py
│ └── hyperparameter_tuning.py
├── notebooks/
│ └── exploratory_data_analysis.ipynb
├── main.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.7+
- Install required packages:
  ```sh
  pip install -r requirements.txt
  ```
  
### Configuration
Update the config/config.yaml file with appropriate settings.

### Running the Project
- Ensure your data is available in the data/raw/ directory.
- Run the main script:
```
python main.py
```

## Project Components

### Data Preprocessing
Data loading and preprocessing are handled in src/preprocess.py.

### Feature Engineering
Feature engineering is performed in src/feature_engineering.py to create various lag, rolling, and date features.

### Model Training
Models are defined and trained in src/model.py, including ARIMA, Prophet, XGBoost, and LightGBM.

### Hyperparameter Tuning
Hyperparameter tuning for XGBoost and LightGBM is performed using Optuna in src/hyperparameter_tuning.py.

### Forecasting
Forecasts are generated in src/model.py and reconciled in src/reconciliation.py.

### Evaluation
Model evaluation is done in src/evaluation.py.

### Visualization
Forecasts are visualized using src/utils.py.