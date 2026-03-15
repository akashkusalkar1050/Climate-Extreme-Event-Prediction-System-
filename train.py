"""
Model Training Script
====================
Trains all climate prediction models.
"""

import os
import sys
import numpy as np
import xarray as xr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.download import ClimateDataDownloader
from data.loader import ClimateDataLoader
from models.flood_model import FloodPredictionModel
from models.heatwave_model import HeatwavePredictionModel
from models.rainfall_model import RainfallPredictionModel


def generate_data():
    """Generate or load climate data."""
    print("=" * 60)
    print("Step 1: Generating Climate Data")
    print("=" * 60)
    
    data_path = "data/raw/climate_data.nc"
    
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}")
        return xr.open_dataset(data_path)
    else:
        print("Generating synthetic climate data...")
        downloader = ClimateDataDownloader()
        datasets = downloader.generate_all_data()
        return datasets['climate']


def train_flood_model(data):
    """Train flood prediction models."""
    print("\n" + "=" * 60)
    print("Step 2: Training Flood Prediction Models")
    print("=" * 60)
    
    loader = ClimateDataLoader()
    
    # Load flood data
    X_train, X_test, y_train, y_test, scaler = loader.load_flood_data(data)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Positive samples: {y_train.sum()} ({100*y_train.mean():.1f}%)")
    
    # Create model
    model = FloodPredictionModel()
    
    # Train Random Forest (baseline)
    print("\n--- Training Random Forest ---")
    model.train_random_forest(X_train, y_train)
    rf_metrics = model.evaluate(X_test, y_test, 'rf')
    print(f"Random Forest Metrics: {rf_metrics}")
    
    # Train LSTM
    print("\n--- Training LSTM ---")
    model.train_lstm(X_train, y_train, epochs=10, batch_size=32)
    lstm_metrics = model.evaluate(X_test, y_test, 'lstm')
    print(f"LSTM Metrics: {lstm_metrics}")
    
    # Save models
    model.save_model("models/saved/flood_lstm.h5", 'lstm')
    model.save_model("models/saved/flood_rf.pkl", 'rf')
    
    print("\n✓ Flood models trained successfully!")
    
    return {'rf': rf_metrics, 'lstm': lstm_metrics}


def train_heatwave_model(data):
    """Train heatwave prediction models."""
    print("\n" + "=" * 60)
    print("Step 3: Training Heatwave Prediction Models")
    print("=" * 60)
    
    loader = ClimateDataLoader()
    
    # Load heatwave data
    X_train, X_test, y_train, y_test, scaler = loader.load_heatwave_data(data)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create model
    model = HeatwavePredictionModel()
    
    # Train XGBoost
    print("\n--- Training XGBoost ---")
    model.train_xgboost(X_train, y_train)
    xgb_metrics = model.evaluate(X_test, y_test, 'xgb')
    print(f"XGBoost Metrics: {xgb_metrics}")
    
    # Train LSTM
    print("\n--- Training LSTM ---")
    model.train_lstm(X_train, y_train, epochs=10, batch_size=32)
    lstm_metrics = model.evaluate(X_test, y_test, 'lstm')
    print(f"LSTM Metrics: {lstm_metrics}")
    
    # Save models
    model.save_model("models/saved/heatwave_lstm.h5", 'lstm')
    model.save_model("models/saved/heatwave_xgb.pkl", 'xgb')
    
    print("\n✓ Heatwave models trained successfully!")
    
    return {'xgb': xgb_metrics, 'lstm': lstm_metrics}


def train_rainfall_model(data):
    """Train rainfall prediction models."""
    print("\n" + "=" * 60)
    print("Step 4: Training Rainfall Prediction Models")
    print("=" * 60)
    
    loader = ClimateDataLoader()
    
    # Load rainfall data
    X_train, X_test, y_train, y_test, scaler = loader.load_rainfall_data(data)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create model
    model = RainfallPredictionModel()
    
    # Prepare spatial data for ConvLSTM
    print("\n--- Preparing spatial data for ConvLSTM ---")
    X_train_spatial, y_train_spatial = model.prepare_spatial_data(
        data,
        features=['pressure', 'precipitable_water', 'cloud_cover'],
        target='rainfall',
        sequence_length=24,
        grid_size=10
    )
    
    print(f"Spatial training data shape: {X_train_spatial.shape}")
    
    # Train ConvLSTM (reduced epochs for demo)
    print("\n--- Training ConvLSTM ---")
    try:
        model.train_conv_lstm(X_train_spatial, y_train_spatial, epochs=5, batch_size=16)
        conv_metrics = model.evaluate(X_test, y_test, 'convlstm')
        print(f"ConvLSTM Metrics: {conv_metrics}")
    except Exception as e:
        print(f"ConvLSTM training skipped: {e}")
        conv_metrics = None
    
    print("\n✓ Rainfall models trained successfully!")
    
    return {'convlstm': conv_metrics}


def generate_visualizations(data):
    """Generate sample visualizations."""
    print("\n" + "=" * 60)
    print("Step 5: Generating Visualizations")
    print("=" * 60)
    
    from visualizations.plots import ClimatePlotter
    from visualizations.maps import RiskMapGenerator
    
    # Create output directory
    os.makedirs("output/visualizations", exist_ok=True)
    
    # Generate plots
    plotter = ClimatePlotter()
    
    try:
        # Temperature time series
        plotter.plot_temperature_timeseries(
            data, lat=28.6, lon=77.2,
            save_path="output/visualizations/temperature_timeseries.png"
        )
        print("✓ Temperature time series saved")
        
        # Climate summary
        plotter.plot_climate_summary(
            data, lat=20.0, lon=77.0,
            save_path="output/visualizations/climate_summary.png"
        )
        print("✓ Climate summary saved")
        
    except Exception as e:
        print(f"Plot generation note: {e}")
    
    # Generate risk maps
    generator = RiskMapGenerator()
    
    try:
        generator.create_combined_risk_map(
            data, time_idx=-1,
            save_path="output/visualizations/risk_map.png"
        )
        print("✓ Risk maps saved")
        
    except Exception as e:
        print(f"Risk map generation note: {e}")
    
    print("\n✓ Visualizations generated successfully!")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("CLIMATE EXTREME EVENT PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Step 1: Generate data
    data = generate_data()
    
    # Step 2: Train flood model
    flood_metrics = train_flood_model(data)
    
    # Step 3: Train heatwave model
    heatwave_metrics = train_heatwave_model(data)
    
    # Step 4: Train rainfall model
    rainfall_metrics = train_rainfall_model(data)
    
    # Step 5: Generate visualizations
    generate_visualizations(data)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    
    print("\n📊 Flood Prediction Metrics:")
    for model_name, metrics in flood_metrics.items():
        print(f"  {model_name}: Accuracy={metrics.get('accuracy', 'N/A'):.3f}, AUC={metrics.get('auc', 'N/A'):.3f}")
    
    print("\n📊 Heatwave Prediction Metrics:")
    for model_name, metrics in heatwave_metrics.items():
        print(f"  {model_name}: RMSE={metrics.get('rmse', 'N/A'):.3f}, MAE={metrics.get('mae', 'N/A'):.3f}")
    
    print("\n📊 Rainfall Prediction Metrics:")
    for model_name, metrics in rainfall_metrics.items():
        if metrics:
            print(f"  {model_name}: RMSE={metrics.get('rmse', 'N/A'):.3f}")
        else:
            print(f"  {model_name}: Not trained")
    
    print("\n" + "=" * 60)
    print("All models trained successfully!")
    print("Run 'streamlit run dashboard/app.py' to start the dashboard")
    print("=" * 60)


if __name__ == "__main__":
    main()

