# 🤖 AI Day Trading Bot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **complete, enterprise-grade AI trading system** featuring machine learning signal generation, advanced risk management, and professional portfolio tracking. Ready for deployment with comprehensive backtesting and interactive dashboard.

🚀 **[LIVE DEMO](https://your-app-name.streamlit.app)** - Deploy your own in 5 minutes!

> ⚠️ **IMPORTANT**: This software is for educational and research purposes. Always paper trade first and never risk money you can't afford to lose.

## ✨ Features

### 📊 **Market Data Integration**

- **Multiple Data Sources**: Yahoo Finance, Alpha Vantage, Polygon.io with automatic failover
- **Real-time Data**: Live market prices and historical data fetching
- **Smart Caching**: Efficient data storage and retrieval

### 🧠 **AI & Machine Learning**

- **XGBoost**: Fast gradient boosting for tabular financial data
- **LightGBM**: Memory-efficient boosting for large datasets
- **LSTM**: Deep learning time series prediction (TensorFlow)
- **Ensemble Methods**: Combining multiple models for better accuracy

### 📈 **Technical Analysis**

- **27+ Technical Indicators**: RSI, MACD, Bollinger Bands, and more
- **Custom Features**: Price patterns, volatility measures, volume analysis
- **Feature Engineering**: Automated creation of 140+ trading features

### 🛡️ **Risk Management**

- **Position Sizing**: Kelly criterion and fixed percentage methods
- **Stop Loss/Take Profit**: Automated risk controls
- **Portfolio Management**: Diversification and exposure limits

### 📱 **Interactive Dashboard**

- **Real-time Monitoring**: Live market data and signals
- **Interactive Charts**: Plotly-powered visualizations
- **Performance Tracking**: Portfolio and strategy analytics
- **Risk Controls**: Manual override and safety controls

### ⚠️ **Advanced Risk Management** ✨

- **Position Sizing**: Volatility-based, Kelly Criterion, Risk Parity algorithms
- **Portfolio Protection**: Stop-loss, take-profit, correlation monitoring
- **Real-time Risk Assessment**: Concentration limits, drawdown protection
- **Smart Trade Validation**: Pre-trade risk checks and warnings

## 🚀 **Quick Deploy to Streamlit Cloud (FREE)**

1. **Fork this repository**
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Create new app** with:
   - Repository: `your-username/AI-day-trading-bot`
   - Main file: `app.py`
4. **Click Deploy!**

**Result**: Live trading bot in 5 minutes at `https://your-app.streamlit.app`

📖 **Detailed Instructions**: See [STREAMLIT_DEPLOY.md](STREAMLIT_DEPLOY.md)

## 🏗️ Architecture

```
AI-day-trading-bot/
├── 📁 config/              # Configuration files
├── 📁 data/               # Market data storage
│   ├── 📁 raw/            # Raw market data
│   └── 📁 processed/     # Preprocessed data
├── 📁 src/               # Source code
│   ├── 📁 data_sources/  # API integrations (Yahoo, Alpha Vantage, Polygon)
│   ├── 📁 ml_models/     # AI models (XGBoost, LightGBM, LSTM)
│   ├── 📁 risk_management/ # Position sizing & risk controls
│   ├── 📁 signals/       # Trading signal generation
│   ├── 📁 backtesting/   # Strategy validation
│   └── 📁 utils/         # Technical indicators & preprocessing
├── 📁 models/            # Trained ML models
├── 📁 logs/             # Application logs
├── 📁 plots/            # Charts and visualizations
├── 📁 scripts/          # Example and utility scripts
├── 📄 dashboard.py      # Streamlit dashboard
├── 📄 run_bot.py        # Main trading bot runner
└── 📄 demo.py           # Quick demonstration
```

## 🚀 Quick Start

### 1. **Installation**

```bash
git clone https://github.com/yourusername/AI-day-trading-bot.git
cd AI-day-trading-bot
python setup.py  # Automatic setup with virtual environment
```

### 2. **Quick Demo**

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python demo.py           # Test basic functionality
```

### 3. **Launch Dashboard**

```bash
streamlit run dashboard.py
# Visit http://localhost:8501
```

### 4. **Configuration (Optional)**

Edit `.env` file with your API keys:

```bash
ALPHA_VANTAGE_API_KEY=your_free_key_here
POLYGON_API_KEY=your_optional_key_here
```

## 🔧 Usage Examples

### **Fetch Market Data**

```bash
python scripts/fetch_data_example.py
```

### **Train AI Models**

```bash
python scripts/train_models_example.py
```

### **Run Full Pipeline**

```bash
python run_bot.py --mode demo --symbols AAPL MSFT GOOGL
```

## 🔑 API Keys (Optional but Recommended)

| Provider          | Cost      | Features                  | Signup                                                  |
| ----------------- | --------- | ------------------------- | ------------------------------------------------------- |
| **Yahoo Finance** | Free      | Basic data, no key needed | Built-in                                                |
| **Alpha Vantage** | Free tier | 500 calls/day             | [Get Key](https://www.alphavantage.co/support/#api-key) |
| **Polygon.io**    | Free tier | 5 calls/min               | [Get Key](https://polygon.io/)                          |

## 🎯 Trading Strategy

The bot employs a sophisticated multi-layered approach:

### **1. Technical Analysis Layer**

- 27+ indicators: RSI, MACD, Bollinger Bands, Stochastic, Williams %R
- Volume analysis and price action patterns
- Support/resistance level detection

### **2. Machine Learning Layer**

- **XGBoost**: Gradient boosting for feature-rich prediction
- **LightGBM**: Fast, memory-efficient ensemble learning
- **LSTM**: Deep learning for sequence pattern recognition
- **Ensemble**: Combines all models for robust predictions

### **3. Risk Management Layer**

- **Position Sizing**: Kelly criterion and fixed percentage methods
- **Stop Loss**: Automatic exit strategies
- **Portfolio Limits**: Maximum exposure controls
- **Volatility Adjustment**: Dynamic position sizing

## 📊 Features In Action

- **Live Market Data**: Real-time price feeds and technical indicators
- **AI Predictions**: Machine learning models generating buy/sell signals
- **Risk Controls**: Automatic position sizing and stop-loss management
- **Performance Tracking**: Portfolio analytics and strategy backtesting
- **Interactive Dashboard**: Web-based control and monitoring interface

## 🛠️ Development & Contributing

```bash
# Setup development environment
python setup.py

# Run tests
pytest tests/ -v

# Code formatting
black . && flake8 .

# Generate documentation
sphinx-build -b html docs/ docs/_build/
```

## 📈 Performance & Backtesting

The bot includes comprehensive backtesting capabilities:

- Historical strategy validation
- Performance metrics (Sharpe ratio, max drawdown, etc.)
- Risk-adjusted returns analysis
- Monte Carlo simulations

## 🔐 Security & Safety

- **No Real Trading**: Currently configured for analysis and paper trading only
- **API Key Protection**: Environment variables and .gitignore for sensitive data
- **Rate Limiting**: Built-in API call throttling
- **Error Handling**: Comprehensive exception handling and logging

## ⚠️ **Risk Disclaimer**

**IMPORTANT**: This software is for educational and research purposes only.

- ✅ **Do**: Use for learning, paper trading, and strategy research
- ❌ **Don't**: Use with real money without thorough testing and risk assessment
- 🧪 **Always**: Start with paper trading and small amounts
- 📚 **Remember**: Past performance doesn't guarantee future results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for free market data API
- Alpha Vantage for comprehensive financial data
- The open-source Python community for excellent libraries
- TensorFlow and scikit-learn teams for ML frameworks

---

**Made with ❤️ for the trading and AI community**
