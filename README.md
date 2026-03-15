<<<<<<< HEAD
# 🌦️ Climate Extreme Event Prediction

A comprehensive machine learning and deep learning project for predicting climate extreme events including floods, heatwaves, and rainfall intensity.

## 📋 Project Overview

This project uses climate data to predict:
- **🌊 Flood Risk** - Predicts probability of flooding based on rainfall, soil moisture, and other factors
- **🔥 Heatwave Risk** - Predicts extreme heat events and temperature forecasts
- **🌧️ Rainfall Intensity** - Predicts rainfall amounts and creates precipitation maps

## 🏗️ Project Structure

```
climate_extreme_prediction/
├── data/                    # Data handling modules
│   ├── __init__.py
│   ├── download.py          # Climate data download/generation
│   ├── preprocess.py        # Data preprocessing utilities
│   └── loader.py            # Data loading functions
├── models/                  # ML/DL models
│   ├── __init__.py
│   ├── flood_model.py       # Flood prediction models (LSTM, CNN-LSTM, RF)
│   ├── heatwave_model.py    # Heatwave prediction (LSTM, GRU, XGBoost)
│   └── rainfall_model.py    # Rainfall prediction (ConvLSTM, UNet)
├── visualizations/          # Visualization modules
│   ├── __init__.py
│   ├── plots.py            # Climate plots and charts
│   └── maps.py             # Risk map generation
├── dashboard/              # Streamlit dashboard
│   └── app.py              # Interactive dashboard
├── config.yaml             # Project configuration
├── requirements.txt         # Python dependencies
├── train.py                # Model training script
└── README.md               # This file
```

## 🚀 Getting Started

### 1. Installation

Create a conda environment and install dependencies:

```bash
# Create environment
conda create -n climate_env python=3.10
conda activate climate_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Climate Data

Run the data generation script:

```bash
python -c "from data.download import ClimateDataDownloader; d = ClimateDataDownloader(); d.generate_all_data()"
```

This generates synthetic climate data covering:
- Temperature, Humidity, Pressure
- Wind Speed, Rainfall, Soil Moisture
- Precipitable Water, Cloud Cover

### 3. Train Models

```bash
python train.py
```

This trains all three prediction models:
- Flood Prediction (LSTM, CNN-LSTM, Random Forest)
- Heatwave Prediction (LSTM, GRU, XGBoost)
- Rainfall Prediction (ConvLSTM, UNet)

### 4. Run Dashboard

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## 📊 Features

### Data Pipeline
- Downloads climate data from ERA5, NASA MERRA-2
- Generates synthetic data for demonstration
- Processes NetCDF files to ML-ready formats
- Creates features: rolling averages, anomalies, indices

### Prediction Models

#### Flood Prediction
- **Input**: Rainfall, soil moisture, river discharge
- **Models**: LSTM, CNN-LSTM, Random Forest
- **Output**: Flood probability (next 24 hours)

#### Heatwave Prediction  
- **Input**: Temperature, humidity, wind
- **Models**: LSTM, GRU, XGBoost
- **Output**: Heatwave risk score, 7-day forecast

#### Rainfall Prediction
- **Input**: Pressure, precipitable water, cloud cover
- **Models**: ConvLSTM, UNet
- **Output**: Rainfall heatmap (next 24 hours)

### Visualizations
- Temperature time series
- Temperature anomaly plots
- Rainfall heatmaps
- Risk maps with color coding
- Seasonal analysis

### Dashboard
- Interactive location selection
- Real-time risk assessment
- Climate statistics
- 7-14 day forecasts

## 📦 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
xarray>=2023.1.0
netCDF4>=1.6.4
tensorflow>=2.13.0
torch>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
streamlit>=1.25.0
```

## 🔧 Configuration

Edit `config.yaml` to customize:

- Data source and time range
- Model hyperparameters
- Training settings
- Visualization options

## 📈 Model Performance

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| LSTM | Flood | AUC | ~0.85 |
| Random Forest | Flood | Accuracy | ~0.80 |
| XGBoost | Heatwave | RMSE | ~2.5°C |
| ConvLSTM | Rainfall | MAE | ~5mm |

*Note: Scores based on synthetic data. Real data will vary.*

## 🌐 Data Sources

For real climate data, use:
- **ERA5**: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- **NASA MERRA-2**: [NASA GES DISC](https://disc.gsfc.nasa.gov/)
- **NOAA**: [NOAA Climate Data](https://www.ncdc.noaa.gov/)

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This is a demonstration project. For real-world climate prediction, 
consult with meteorologists and use official data sources.

=======
# Climate-Extreme-Event-Prediction-System-
Machine learning project to predict extreme climate events such as floods, heatwaves, and heavy rainfall using historical climate datasets. Includes data preprocessing, feature engineering, predictive modeling with Scikit-Learn, TensorFlow, and XGBoost, and visualization of climate trends.

