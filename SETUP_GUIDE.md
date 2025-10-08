# AI Day Trading Bot - Setup Guide

## 🚀 Quick Start

### 1. Clone and Setup
```bash
cd AI-day-trading-bot
python setup.py
```

### 2. Configure API Keys
Edit the `.env` file with your API keys:
```bash
# API Keys (at least one required)
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here

# Trading Configuration
INITIAL_CAPITAL=10000
MAX_POSITION_SIZE=0.05
STOP_LOSS_PERCENTAGE=0.02
```

### 3. Get API Keys (Free Options)

#### Alpha Vantage (Recommended - Free)
1. Visit: https://www.alphavantage.co/support/#api-key
2. Sign up for free API key
3. Limit: 5 calls per minute, 500 calls per day

#### Yahoo Finance (Built-in - No Key Required)
- Used as fallback data source
- No registration required
- Rate limited but reliable

#### Polygon.io (Optional - Free Tier)
1. Visit: https://polygon.io/
2. Sign up for free tier
3. Limit: 5 calls per minute

### 4. Test Installation
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Test data fetching
python scripts/fetch_data_example.py

# Train models (requires data)
python scripts/train_models_example.py

# Run dashboard
streamlit run dashboard.py
```

## 📋 System Requirements

### Python
- Python 3.8 or higher
- Virtual environment recommended

### System Dependencies

#### For TA-Lib (Technical Analysis)

**Windows:**
```bash
# Download TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Install the .whl file for your Python version
pip install TA_Lib‑0.4.32‑cp310‑cp310‑win_amd64.whl
```

**Mac:**
```bash
brew install ta-lib
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ta-lib-dev
```

### Memory Requirements
- Minimum: 4GB RAM
- Recommended: 8GB+ RAM for multiple models
- Storage: 2GB+ for data and models

## 🏗️ Project Structure

```
AI-day-trading-bot/
├── config/
│   └── config.py              # Configuration settings
├── src/
│   ├── data_sources/          # Data fetching from APIs
│   ├── ml_models/             # Machine learning models
│   ├── utils/                 # Technical indicators & preprocessing
│   ├── risk_management/       # Portfolio & risk management
│   ├── signals/               # Trading signal generation
│   └── backtesting/           # Strategy testing
├── data/
│   ├── raw/                   # Raw market data
│   ├── processed/             # Processed features
│   └── historical/            # Historical data archive
├── models/                    # Trained ML models
├── logs/                      # Application logs
├── plots/                     # Analysis visualizations
├── scripts/                   # Example and utility scripts
├── dashboard.py               # Streamlit dashboard
├── run_bot.py                # Main bot execution
└── requirements.txt           # Python dependencies
```

## 🔧 Configuration Options

### Trading Parameters
```python
# Position sizing
MAX_POSITION_SIZE=0.05        # 5% max per position
INITIAL_CAPITAL=10000         # Starting capital

# Risk management
STOP_LOSS_PERCENTAGE=0.02     # 2% stop loss
TAKE_PROFIT_PERCENTAGE=0.05   # 5% take profit

# Model parameters
FEATURE_WINDOW_DAYS=60        # Days of history for features
PREDICTION_HORIZON_DAYS=5     # Days ahead to predict
MODEL_RETRAIN_DAYS=30         # Retrain frequency
```

### Symbols Configuration
```python
# Default symbols (modify in config.py)
TRADING_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "NFLX", "CRM", "ADBE"
]
```

## 🤖 Usage Examples

### Basic Data Fetching
```python
from src.data_sources.data_manager import DataManager

# Initialize with API keys
data_manager = DataManager(
    alpha_vantage_key="your_key",
    polygon_key="your_key"
)

# Fetch historical data
data = data_manager.get_historical_data("AAPL", period="1y")
print(f"Fetched {len(data)} records")
```

### Model Training
```python
from src.ml_models.model_trainer import ModelTrainer

# Train multiple models
trainer = ModelTrainer()
results = trainer.train_multiple_models(
    trainer.get_default_model_configs(),
    X_train, y_train, X_val, y_val
)

# Compare performance
comparison = trainer.compare_models(X_test, y_test)
print(comparison)
```

### Command Line Usage
```bash
# Fetch data only
python run_bot.py --mode fetch --symbols AAPL MSFT GOOGL

# Train models
python run_bot.py --mode train --symbols AAPL

# Generate signals
python run_bot.py --mode signal --symbols AAPL MSFT

# Full pipeline
python run_bot.py --mode full --symbols AAPL MSFT --period 2y
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

1. **Market Overview**: Real-time prices and market status
2. **Trading Signals**: AI-generated buy/sell recommendations
3. **Portfolio Management**: Position tracking and P&L
4. **Performance Analytics**: Strategy backtesting results
5. **Model Configuration**: Parameter tuning interface

### Running the Dashboard
```bash
streamlit run dashboard.py
```
Access at: http://localhost:8501

## 🔬 Model Types

### XGBoost
- **Best for**: Tabular data with mixed feature types
- **Pros**: Fast, interpretable, robust
- **Use case**: Default choice for most scenarios

### LightGBM
- **Best for**: Large datasets, faster training
- **Pros**: Memory efficient, fast
- **Use case**: When speed is priority

### LSTM (Neural Network)
- **Best for**: Sequential time series patterns
- **Pros**: Captures complex temporal relationships
- **Use case**: High-frequency trading, complex patterns

### Ensemble
- **Best for**: Combining multiple model strengths
- **Pros**: More robust, better generalization
- **Use case**: Production deployment

## 🛡️ Risk Management

### Position Sizing
- Kelly Criterion implementation
- Volatility-based sizing
- Maximum position limits

### Stop Loss/Take Profit
- ATR-based dynamic stops
- Trailing stop losses
- Risk-reward optimization

### Portfolio Management
- Correlation analysis
- Sector diversification
- Maximum drawdown controls

## 📈 Backtesting

### Features
- Walk-forward analysis
- Out-of-sample testing
- Performance metrics calculation
- Drawdown analysis

### Metrics Tracked
- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio

## 🚨 Important Disclaimers

### ⚠️ Risk Warning
- **This is for educational purposes only**
- **Never risk money you cannot afford to lose**
- **Past performance does not guarantee future results**
- **Always test with paper trading first**

### Legal Considerations
- Check local regulations for algorithmic trading
- Ensure compliance with tax obligations
- Consider professional financial advice

### Technical Limitations
- Models may fail during market regime changes
- API rate limits may affect real-time performance
- No guarantee of profitability

## 🔧 Troubleshooting

### Common Issues

#### "No module named 'talib'"
```bash
# Install TA-Lib system dependency first
# Then reinstall Python package:
pip uninstall TA-Lib
pip install TA-Lib
```

#### "API key not working"
- Verify key is correct in .env file
- Check API quotas and limits
- Ensure no extra spaces in .env file

#### "No data fetched"
- Check internet connection
- Verify API keys are active
- Try different symbols
- Check market hours

#### "Memory errors during training"
- Reduce feature window size
- Use fewer symbols
- Increase system RAM
- Use LightGBM instead of XGBoost

### Getting Help

1. Check logs in `logs/` directory
2. Review error messages carefully
3. Verify all dependencies are installed
4. Test with simple examples first

### Performance Optimization

#### For Large Datasets
- Use data sampling
- Implement incremental learning
- Use feature selection
- Parallel processing

#### For Real-time Trading
- Pre-load models
- Cache frequently used data
- Use async operations
- Monitor API rate limits

## 🚀 Next Steps

After successful setup:

1. **Paper Trading**: Test strategies without real money
2. **Live Data Integration**: Connect to real-time feeds
3. **Broker Integration**: Implement actual trade execution
4. **Risk Monitoring**: Set up real-time alerts
5. **Performance Tracking**: Monitor live performance

## 📚 Additional Resources

- [Python for Finance](https://github.com/PacktPublishing/Python-for-Finance-Second-Edition)
- [Quantitative Trading with Python](https://www.quantstart.com/)
- [Machine Learning for Trading](https://www.coursera.org/learn/machine-learning-trading)
- [Financial Data APIs Comparison](https://rapidapi.com/blog/best-stock-api/)

---

**Happy Trading! 🚀📈**

*Remember: The best strategy is the one you understand and can stick to consistently.*