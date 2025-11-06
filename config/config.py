import os
import sys
import tempfile

# Try to load environment variables, but don't fail if dotenv is missing
try:
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()
    print("✅ Environment variables loaded from .env")
except (ImportError, Exception) as e:
    print(f"⚠️ Could not load dotenv: {e}. Using system environment variables only.")

# Detect environment - CHECK STREAMLIT FIRST
if os.path.exists("/mount/src"):
    # We're on Streamlit Cloud - use temp directory for writable files
    BASE_DIR = "/mount/src/ai-day-trading-bot"
    WRITABLE_DIR = tempfile.gettempdir()
    print(
        f"Running in Streamlit Cloud environment: BASE_DIR={BASE_DIR}, WRITABLE_DIR={WRITABLE_DIR}"
    )
elif os.path.exists("/app"):
    # We're on Render with /app path
    BASE_DIR = "/app"
    WRITABLE_DIR = BASE_DIR
    print("Running in Render environment: BASE_DIR set to /app")
elif hasattr(sys, "_MEIPASS"):
    # Running as a PyInstaller bundle
    BASE_DIR = sys._MEIPASS
    WRITABLE_DIR = tempfile.gettempdir()
    print(f"Running as executable: BASE_DIR={BASE_DIR}, WRITABLE_DIR={WRITABLE_DIR}")
else:
    # Local development
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    WRITABLE_DIR = BASE_DIR
    print(f"Running locally: BASE_DIR set to {BASE_DIR}")

# Directory paths - use WRITABLE_DIR for logs, models, data, plots
LOGS_DIR = os.path.join(WRITABLE_DIR, "logs")
MODELS_DIR = os.path.join(WRITABLE_DIR, "models")
DATA_DIR = os.path.join(WRITABLE_DIR, "data")
PLOTS_DIR = os.path.join(WRITABLE_DIR, "plots")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Ensure the directories exist (with error handling for read-only filesystems)
for directory in [LOGS_DIR, MODELS_DIR, DATA_DIR, PLOTS_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
    except (PermissionError, OSError) as e:
        print(f"⚠️ Could not create directory {directory}: {e}")

# Create subdirectories
subdirs = [
    os.path.join(DATA_DIR, "raw"),
    os.path.join(DATA_DIR, "processed"),
    os.path.join(DATA_DIR, "historical"),
    os.path.join(PLOTS_DIR, "analysis"),
    os.path.join(PLOTS_DIR, "performance"),
    os.path.join(PLOTS_DIR, "signals"),
]

for subdir in subdirs:
    try:
        os.makedirs(subdir, exist_ok=True)
    except (PermissionError, OSError) as e:
        print(f"⚠️ Could not create subdirectory {subdir}: {e}")

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

# Risk management settings (alternative names for compatibility)
STOP_LOSS_PCT = STOP_LOSS_PERCENTAGE
TAKE_PROFIT_PCT = TAKE_PROFIT_PERCENTAGE

# Trading engine settings
MAX_POSITIONS = 5
SIGNAL_THRESHOLD = 0.6
