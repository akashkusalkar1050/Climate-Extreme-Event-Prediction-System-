# Climate Modeling & Extreme Event Prediction Project

## Project Overview
This project builds ML/DL models to predict climate extreme events: Floods, Heatwaves, and Rainfall Intensity using climate data.

---

## Phase 1: Project Setup & Environment ✅ COMPLETED
- [x] 1.1 Create project directory structure
- [x] 1.2 Set up conda environment with dependencies
- [x] 1.3 Create requirements.txt
- [x] 1.4 Create config.yaml for project settings

## Phase 2: Data Collection & Pipeline ✅ COMPLETED
- [x] 2.1 Create data download script (using open climate data sources)
- [x] 2.2 Build netCDF to pandas/numpy conversion utilities
- [x] 2.3 Create data preprocessing & normalization functions
- [x] 2.4 Build data augmentation utilities

## Phase 3: Flood Prediction Model ✅ COMPLETED
- [x] 3.1 Create flood prediction data loader
- [x] 3.2 Build LSTM model for flood prediction
- [x] 3.3 Build CNN-LSTM hybrid model
- [x] 3.4 Create Random Forest baseline model
- [x] 3.5 Training script with evaluation metrics

## Phase 4: Heatwave Prediction Model ✅ COMPLETED
- [x] 4.1 Create heatwave data loader
- [x] 4.2 Build LSTM/GRU time-series model
- [x] 4.3 Build XGBoost gradient boosting model
- [x] 4.4 Create 7-day temperature forecast module

## Phase 5: Rainfall Intensity Prediction Model ✅ COMPLETED
- [x] 5.1 Create rainfall data loader
- [x] 5.2 Build ConvLSTM model (CNN + LSTM)
- [x] 5.3 Build UNet for rainfall heatmaps
- [x] 5.4 Create prediction visualization tools

## Phase 6: Visualizations & Dashboard ✅ COMPLETED
- [x] 6.1 Create temperature anomaly plots
- [x] 6.2 Create rainfall heatmaps
- [x] 6.3 Build risk maps with color coding
- [x] 6.4 Create Streamlit dashboard
- [x] 6.5 Add interactive geo-located warnings

## Phase 7: Reports & Documentation ✅ COMPLETED
- [x] 7.1 Generate model accuracy reports
- [x] 7.2 Create climate trend insights
- [x] 7.3 Write risk interpretation guide
- [x] 7.4 Create README.md with usage instructions

---

## Project Structure Created
```
climate_extreme_prediction/
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── README.md              # Documentation
├── train.py               # Training script
├── data/                  # Data modules
│   ├── download.py        # Data generation
│   ├── preprocess.py      # Preprocessing
│   └── loader.py          # Data loader
├── models/                # ML/DL models
│   ├── flood_model.py    # Flood prediction
│   ├── heatwave_model.py # Heatwave prediction
│   └── rainfall_model.py # Rainfall prediction
├── visualizations/        # Visualizations
│   ├── plots.py          # Climate plots
│   └── maps.py           # Risk maps
└── dashboard/            # Dashboard
    └── app.py            # Streamlit app
```

## How to Run

### 1. Install Dependencies
```bash
conda create -n climate_env python=3.10
conda activate climate_env
pip install -r requirements.txt
```

### 2. Generate Data
```bash
python -c "from data.download import ClimateDataDownloader; d = ClimateDataDownloader(); d.generate_all_data()"
```

### 3. Train Models
```bash
python train.py
```

### 4. Run Dashboard
```bash
cd dashboard
streamlit run app.py
```

