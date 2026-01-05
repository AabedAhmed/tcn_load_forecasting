# Short-Term Load Forecasting Using Temporal Convolutional Networks

## Project Overview
This project implements TCN and LSTM models for next-hour electricity load forecasting using the PJME dataset.

## Requirements
- Python 3.8+
- PyTorch 1.12+
- pandas, numpy, matplotlib, scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
PJME Hourly Energy Consumption (2002-2018)
- Source: [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- 145,000+ hourly measurements

## Usage

### Step 1: Data Loading and EDA
```bash
# Run in Google Colab or Jupyter
notebooks/step1_data_loading.ipynb
```

### Step 2: Data Preprocessing
```bash
notebooks/step2_preprocessing.ipynb
```

### Step 3: Model Training
```bash
notebooks/step3_model_training.ipynb
```

## Results
- Best TCN MAPE: 2.65%
- Best LSTM MAPE: 3.52%
- Target achieved: MAPE < 5% ✓

## Model Configurations
**TCN:**
- Filters: [32, 64, 128]
- Residual blocks: [2, 3, 4]
- Kernel sizes: [2, 3, 4]
- Lookback windows: [24, 48, 168] hours

**LSTM:**
- Layers: [1, 2, 3]
- Hidden units: [32, 64, 128]
- Lookback windows: [24, 48, 168] hours

## Authors
- Abed Ahmed
- Bouderbala Mohamed Islem

## Citation
If you use this code, please cite:
```
@article{yourname2024tcn,
  title={Short-Term Load Forecasting Using Temporal Convolutional Networks},
  author={Ahmed, Abed and Islem, Bouderbala Mohamed},
  journal={University of Guelma},
  year={2026}
}
```
