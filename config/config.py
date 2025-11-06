import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Detect environment - Render uses /app as the root directory
if os.path.exists("/app"):
    # We're on Render with /app path
    BASE_DIR = "/app"
    print("Running in Render environment: BASE_DIR set to /app")
else:
    # Local development
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Running locally: BASE_DIR set to {BASE_DIR}")

# Directory paths
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Ensure the directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Create subdirectories
os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "historical"), exist_ok=True)
os.makedirs(os.path.join(PLOTS_DIR, "analysis"), exist_ok=True)
os.makedirs(os.path.join(PLOTS_DIR, "performance"), exist_ok=True)
os.makedirs(os.path.join(PLOTS_DIR, "signals"), exist_ok=True)

# API Configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# Trading Configuration
INITIAL_CAPITAL = 100000.0
MAX_POSITION_SIZE = 0.20  # 20% of portfolio per position
STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.15  # 15% take profit

# Environment Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Market Hours Configuration
MARKET_OPEN_HOUR = int(os.getenv("MARKET_OPEN_HOUR", 9))
MARKET_OPEN_MINUTE = int(os.getenv("MARKET_OPEN_MINUTE", 30))
MARKET_CLOSE_HOUR = int(os.getenv("MARKET_CLOSE_HOUR", 16))
MARKET_CLOSE_MINUTE = int(os.getenv("MARKET_CLOSE_MINUTE", 0))

# Model Configuration
MODEL_RETRAIN_DAYS = int(os.getenv("MODEL_RETRAIN_DAYS", 30))
FEATURE_WINDOW_DAYS = int(os.getenv("FEATURE_WINDOW_DAYS", 60))
PREDICTION_HORIZON_DAYS = int(os.getenv("PREDICTION_HORIZON_DAYS", 5))

# Symbols to trade (S&P 500 top performers by default)
DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "META",
    "NFLX",
    "SHOP",
    "SQ",
]

TRADING_SYMBOLS = os.getenv("TRADING_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",")
