# Short-Term Load Forecasting with Temporal Convolutional Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predict next-hour electricity load for smart grids using Temporal Convolutional Networks (TCN) and compare performance with LSTM baseline.

## 🎯 Objective

Develop a deep learning model to forecast hourly electricity consumption with **MAPE < 5%**, comparing TCN architecture against traditional LSTM approach.

## 📊 Dataset

- **Primary**: PJME Hourly Energy Consumption (2002-2018)
- **Alternative**: UCI Individual Household Electric Power Consumption
- **Source**: [Kaggle PJME Dataset](https://www.kaggle.com/robikscube/hourly-energy-consumption)

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/tcn-load-forecasting.git
cd tcn-load-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data
```bash
python scripts/download_data.py --dataset pjme
```

### Train Models
```bash
# Train LSTM baseline
python scripts/train_lstm.py --config configs/lstm_config.yaml

# Train TCN model
python scripts/train_tcn.py --config configs/tcn_config.yaml
```

### Evaluate
```bash
python scripts/evaluate.py --model-path results/models/tcn_best.pth
```

## 📈 Results

| Model | MAPE (%) | MAE (MW) | RMSE (MW) | Training Time |
|-------|----------|----------|-----------|---------------|
| LSTM  | TBD      | TBD      | TBD       | TBD           |
| TCN   | TBD      | TBD      | TBD       | TBD           |

*Results will be updated after model training*

## 🏗️ Architecture

### LSTM Baseline
- 2 LSTM layers (128 units each)
- Dropout (0.2)
- Dense output layer
- Lookback: 168 hours (1 week)

### TCN Model
- 6 residual blocks
- Channel sizes: [32, 64, 128, 128, 256, 256]
- Kernel size: 3
- Dilation factors: [1, 2, 4, 8, 16, 32]
- Dropout: 0.2

## 📁 Project Structure
├── data/              # Data storage
├── notebooks/         # Jupyter notebooks for exploration
├── src/              # Source code
├── scripts/          # Executable scripts
├── configs/          # Configuration files
├── results/          # Model outputs and visualizations
└── tests/            # Unit tests
## 🔧 Configuration

Edit `configs/tcn_config.yaml` to modify hyperparameters:
```yaml
model:
  num_channels: [32, 64, 128, 128, 256, 256]
  kernel_size: 3
  dropout: 0.2

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  
data:
  lookback_window: 168
  forecast_horizon: 1
```

## 📊 Features

- ✅ Temporal feature engineering (hour, day, week patterns)
- ✅ Multiple evaluation metrics (MAPE, MAE, RMSE)
- ✅ Visualization tools for predictions
- ✅ Model comparison framework
- ✅ Configurable architecture via YAML
- ✅ Checkpoint saving and loading

## 🧪 Running Tests
```bash
pytest tests/
```

## 📝 Documentation

- [Methodology](docs/methodology.md)
- [Results Analysis](docs/results.md)
- [API Reference](docs/api_reference.md)

## 📚 References

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.
- [PyTorch TCN Implementation](https://github.com/locuslab/TCN)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Your Name - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- PJME dataset from Kaggle
- PyTorch TCN library
- Anthropic Claude for project guidance
