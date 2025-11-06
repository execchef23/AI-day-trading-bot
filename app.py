"""
AI Day Trading Bot - Deployment Ready Dashboard

Streamlit dashboard optimized for cloud deployment with graceful dependency handling.
"""

import logging
import os
import sys
import warnings
from datetime import datetime

# Suppress warnings for cleaner deployment logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Streamlit must be imported first for deployment
try:
    import streamlit as st
except ImportError:
    print("❌ Streamlit not found. Install with: pip install streamlit")
    sys.exit(1)

# Core libraries with fallbacks
try:
    from collections import defaultdict

    import numpy as np
    import pandas as pd
except ImportError as e:
    st.error(f"❌ Core libraries missing: {e}")
    st.stop()

# Plotting libraries with fallbacks
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available - charts disabled")

# Setup logging first
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import our modules with error handling
MODULES_LOADED = {
    "data_manager": False,
    "signals": False,
    "risk_management": False,
    "small_account": False,
}

try:
    from src.data_sources import DataManager

    MODULES_LOADED["data_manager"] = True
except ImportError as e:
    st.sidebar.warning(f"Data sources not available: {e}")

try:
    from src.signals.signal_manager import SignalManager

    MODULES_LOADED["signals"] = True
except ImportError as e:
    st.sidebar.warning(f"Signal generation not available: {e}")

try:
    from src.risk_management.portfolio_manager import PortfolioManager

    MODULES_LOADED["risk_management"] = True
except ImportError:
    logger.warning("⚠️ Risk management module not available")

# Small Account Trading System
try:
    from src.small_account import (
        AccountTier,
        GrowthCalculator,
        GrowthScenario,
        GrowthStrategy,
        SmallAccountPositionSizer,
        SmallAccountStrategies,
        StrategyType,
    )

    MODULES_LOADED["small_account"] = True
    logger.info("✅ Small Account Trading System loaded")
except ImportError:
    logger.warning("⚠️ Small Account Trading System not available")

# Trading Engine
TRADING_ENGINE_LOADED = False
try:
    from src.trading_engine.live_trading_engine import (
        TradingConfig,
        TradingState,
        get_trading_engine,
    )

    TRADING_ENGINE_LOADED = True
    logger.info("✅ Live Trading Engine loaded")
except ImportError:
    logger.warning("⚠️ Live Trading Engine not available")

# Real-Time Monitoring System
MONITORING_SYSTEM_LOADED = False
try:
    from src.monitoring import AlertLevel, AlertType, RealTimeMonitor, get_monitor

    MONITORING_SYSTEM_LOADED = True
    logger.info("✅ Real-Time Monitoring System loaded")
except ImportError:
    logger.warning("⚠️ Real-Time Monitoring System not available")

# Stock Growth Screener System
SCREENER_LOADED = False
try:
    from src.screening.stock_screener import (
        GrowthCategory,
        ScreeningCriteria,
        StockGrowthScreener,
        get_screener,
    )

    SCREENER_LOADED = True
    logger.info("✅ Stock Growth Screener loaded")
except ImportError:
    logger.warning("⚠️ Stock Growth Screener not available")

# Configuration
INITIAL_CAPITAL = 100000
TRADING_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

# Configuration with fallbacks
try:
    from config.config import INITIAL_CAPITAL, TRADING_SYMBOLS
except ImportError:
    TRADING_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    INITIAL_CAPITAL = 100000.0

# Page configuration
st.set_page_config(
    page_title="AI Day Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = not all(MODULES_LOADED.values())
if "portfolio_value" not in st.session_state:
    st.session_state.portfolio_value = INITIAL_CAPITAL
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "trading_active" not in st.session_state:
    st.session_state.trading_active = False
if "total_trades" not in st.session_state:
    st.session_state.total_trades = 0
if "winning_trades" not in st.session_state:
    st.session_state.winning_trades = 0
if "daily_pnl" not in st.session_state:
    st.session_state.daily_pnl = 0.0
if "last_signal_time" not in st.session_state:
    st.session_state.last_signal_time = None
if "trading_engine" not in st.session_state and TRADING_ENGINE_LOADED:
    st.session_state.trading_engine = get_trading_engine()
if "engine_status" not in st.session_state:
    st.session_state.engine_status = {"state": "stopped", "enabled": False}

# Initialize monitoring system
if "monitor" not in st.session_state and MONITORING_SYSTEM_LOADED:
    st.session_state.monitor = get_monitor()
    # Connect monitoring to trading engine
    if "trading_engine" in st.session_state:
        st.session_state.monitor.set_trading_components(
            trading_engine=st.session_state.trading_engine
        )
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False

# Initialize stock screener
if "screener" not in st.session_state and SCREENER_LOADED:
    st.session_state.screener = get_screener()
    # Connect screener to data manager and signal manager if available
    if MODULES_LOADED["data_manager"]:
        try:
            from src.data_sources.data_manager import DataManager

            data_manager = DataManager()
            st.session_state.screener.set_components(data_manager=data_manager)
        except Exception as e:
            logger.warning(f"Could not connect screener to data manager: {e}")
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "scan_results" not in st.session_state:
    st.session_state.scan_results = {}

# Initialize small account system
if "small_account_balance" not in st.session_state:
    st.session_state.small_account_balance = 500.0
if "selected_strategy" not in st.session_state:
    st.session_state.selected_strategy = (
        StrategyType.SWING_TRADING if MODULES_LOADED["small_account"] else None
    )
if "growth_scenario" not in st.session_state:
    st.session_state.growth_scenario = (
        GrowthScenario.MODERATE if MODULES_LOADED["small_account"] else None
    )
if "position_sizer" not in st.session_state and MODULES_LOADED["small_account"]:
    st.session_state.position_sizer = SmallAccountPositionSizer()
if "growth_calculator" not in st.session_state and MODULES_LOADED["small_account"]:
    st.session_state.growth_calculator = GrowthCalculator()


def create_demo_data():
    """Create demo data for deployment showcase"""

    # Demo portfolio data
    demo_positions = {
        "AAPL": {
            "quantity": 50,
            "entry_price": 150.00,
            "current_price": 155.25,
            "pnl": 262.50,
        },
        "MSFT": {
            "quantity": 30,
            "entry_price": 300.00,
            "current_price": 305.75,
            "pnl": 172.50,
        },
        "GOOGL": {
            "quantity": 10,
            "entry_price": 2500.00,
            "current_price": 2485.30,
            "pnl": -147.00,
        },
    }

    # Demo market data
    dates = pd.date_range(start="2024-09-01", end="2024-10-09", freq="D")
    np.random.seed(42)  # Consistent demo data

    demo_data = {}
    for symbol in TRADING_SYMBOLS[:3]:
        base_price = {"AAPL": 150, "MSFT": 300, "GOOGL": 2500}.get(symbol, 100)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        demo_data[symbol] = pd.DataFrame(
            {
                "Date": dates,
                "Close": prices,
                "Open": [p * 0.995 for p in prices],
                "High": [p * 1.01 for p in prices],
                "Low": [p * 0.99 for p in prices],
                "Volume": np.random.randint(1000000, 5000000, len(dates)),
            }
        ).set_index("Date")

    return demo_positions, demo_data


def generate_demo_data_for_symbol(symbol: str, time_period: str):
    """Generate demo data for any symbol"""

    # Map time periods to number of days
    period_days = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}

    days = period_days.get(time_period, 30)
    dates = pd.date_range(
        start=f"2024-{10 - days // 30:02d}-01", periods=days, freq="D"
    )

    # Base prices for different symbols
    base_prices = {
        "AAPL": 150,
        "MSFT": 300,
        "GOOGL": 2500,
        "AMZN": 120,
        "TSLA": 200,
        "NVDA": 400,
        "META": 250,
        "NFLX": 180,
    }

    base_price = base_prices.get(symbol, 100)

    # Generate realistic price movements
    np.random.seed(hash(symbol) % 1000)  # Consistent per symbol
    returns = np.random.normal(0.001, 0.02, len(dates))

    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    return pd.DataFrame(
        {
            "Open": [p * (0.99 + np.random.random() * 0.02) for p in prices],
            "High": [p * (1.005 + np.random.random() * 0.015) for p in prices],
            "Low": [p * (0.985 + np.random.random() * 0.01) for p in prices],
            "Close": prices,
            "Volume": np.random.randint(500000, 10000000, len(dates)),
        },
        index=dates,
    )


def create_line_chart(data, symbol, title):
    """Create a line chart for price data"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name=f"{symbol} Close",
            line=dict(color="#1f77b4", width=2),
            hovertemplate=f"{symbol}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=True,
        hovermode="x",
    )

    return fig


def create_candlestick_chart(data, symbol, title):
    """Create a candlestick chart"""
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=symbol,
            increasing_line_color="green",
            decreasing_line_color="red",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        xaxis_rangeslider_visible=False,
    )

    return fig


def create_volume_chart(data, symbol, title):
    """Create a volume chart"""
    fig = go.Figure()

    colors = [
        "green" if close >= open else "red"
        for close, open in zip(data["Close"], data["Open"])
    ]

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker_color=colors,
            hovertemplate=f"{symbol}<br>Date: %{{x}}<br>Volume: %{{y:,.0f}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title.replace("Volume", "Volume Chart"),
        xaxis_title="Date",
        yaxis_title="Volume",
        height=500,
        showlegend=True,
    )

    return fig


def display_price_statistics(data, symbol, data_source):
    """Display current price statistics"""

    current_price = data["Close"].iloc[-1]
    prev_price = data["Close"].iloc[-2] if len(data) > 1 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0

    # Calculate statistics
    high_52w = data["High"].max()
    low_52w = data["Low"].min()
    avg_volume = data["Volume"].mean()

    st.subheader(f"📊 {symbol} Statistics ({data_source} Data)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"{change:+.2f} ({change_pct:+.1f}%)",
        )

    with col2:
        st.metric(label="52W High", value=f"${high_52w:.2f}")

    with col3:
        st.metric(label="52W Low", value=f"${low_52w:.2f}")

    with col4:
        st.metric(label="Avg Volume", value=f"{avg_volume:,.0f}")


def display_technical_indicators(data, symbol):
    """Display technical indicators"""

    st.subheader("🔍 Technical Indicators")

    try:
        # Calculate simple moving averages
        sma_20 = data["Close"].rolling(window=20).mean()
        sma_50 = data["Close"].rolling(window=50).mean()

        # Calculate RSI (simple version)
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Display current values
        col1, col2, col3 = st.columns(3)

        with col1:
            if not sma_20.empty:
                st.metric("SMA 20", f"${sma_20.iloc[-1]:.2f}")

        with col2:
            if not sma_50.empty:
                st.metric("SMA 50", f"${sma_50.iloc[-1]:.2f}")

        with col3:
            if not rsi.empty:
                rsi_val = rsi.iloc[-1]
                rsi_signal = (
                    "Overbought"
                    if rsi_val > 70
                    else "Oversold"
                    if rsi_val < 30
                    else "Neutral"
                )
                st.metric("RSI (14)", f"{rsi_val:.1f}", rsi_signal)

        # Plot technical indicators
        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data["Close"], name="Close", line=dict(color="blue")
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=sma_20, name="SMA 20", line=dict(color="orange")
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=sma_50, name="SMA 50", line=dict(color="red")
                )
            )

            fig.update_layout(
                title=f"{symbol} - Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")

    return demo_positions, demo_data


def display_system_status():
    """Display system status and controls"""

    # Trading controls header
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🤖 Trading Bot Control Center")

    with col2:
        # Main trading toggle - integrate with live engine if available
        if TRADING_ENGINE_LOADED and "trading_engine" in st.session_state:
            engine = st.session_state.trading_engine
            engine_status = engine.get_status()

            if engine_status["state"] == "running":
                if st.button(
                    "🛑 Stop Live Engine",
                    type="primary",
                    help="Stop live trading engine",
                ):
                    result = engine.stop_trading()
                    if result["success"]:
                        st.session_state.trading_active = False
                        st.success("🛑 Live trading engine stopped")
                        st.rerun()
            else:
                if st.button(
                    "🚀 Start Live Engine",
                    type="primary",
                    help="Start live trading engine",
                ):
                    result = engine.start_trading()
                    if result["success"]:
                        st.session_state.trading_active = True
                        st.success("🚀 Live trading engine started")
                        st.rerun()
        else:
            # Fallback to basic demo controls
            if st.session_state.trading_active:
                if st.button(
                    "🛑 Stop Trading", type="primary", help="Stop automated trading"
                ):
                    st.session_state.trading_active = False
                    st.success("🛑 Trading bot stopped")
                    st.rerun()
            else:
                if st.button(
                    "▶️ Start Trading", type="primary", help="Start automated trading"
                ):
                    st.session_state.trading_active = True
                    st.success("▶️ Trading bot started")
                    st.rerun()

    # Status metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        status_color = "🟢" if st.session_state.trading_active else "🔴"
        status_text = "Active" if st.session_state.trading_active else "Stopped"
        st.metric("Bot Status", f"{status_color} {status_text}")

    with col2:
        st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.2f}")

    with col3:
        available_cash = INITIAL_CAPITAL * 0.7  # Assume 70% available for new positions
        st.metric("Available Cash", f"${available_cash:,.0f}")

    with col4:
        st.metric("Active Positions", len(st.session_state.positions))

    with col5:
        # System readiness indicator
        if all(MODULES_LOADED.values()):
            st.metric("System", "🟢 Live")
        else:
            st.metric("System", "🟡 Demo")


def display_portfolio_overview():
    """Display portfolio overview with live trading features"""

    st.header("� Portfolio Overview")

    # Trading performance summary
    col1, col2, col3, col4 = st.columns(4)

    if st.session_state.demo_mode:
        # Demo portfolio data
        demo_positions, _ = create_demo_data()

        total_value = sum(
            pos["current_price"] * pos["quantity"] for pos in demo_positions.values()
        )
        total_pnl = sum(pos["pnl"] for pos in demo_positions.values())
        pnl_pct = (
            (total_pnl / (total_value - total_pnl)) * 100
            if total_value > total_pnl
            else 0
        )

        with col1:
            st.metric("Portfolio Value", f"${total_value:,.2f}")

        with col2:
            delta_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric(
                "Total P&L",
                f"${total_pnl:,.2f}",
                delta=f"{pnl_pct:+.1f}%",
                delta_color=delta_color,
            )

        with col3:
            daily_change = st.session_state.daily_pnl
            st.metric(
                "Today's P&L",
                f"${daily_change:+,.2f}",
                delta=f"{(daily_change / total_value) * 100:+.1f}%"
                if total_value > 0
                else "0%",
            )

        with col4:
            win_rate = (
                st.session_state.winning_trades / max(st.session_state.total_trades, 1)
            ) * 100
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                delta=f"{st.session_state.total_trades} trades",
            )

        # Trading activity controls
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Refresh Positions", help="Update portfolio positions"):
                st.success("✅ Positions refreshed")

        with col2:
            if st.button("📊 Generate Signals", help="Run ML models for new signals"):
                with st.spinner("Analyzing market data..."):
                    import time

                    time.sleep(2)  # Simulate processing
                    st.session_state.last_signal_time = datetime.now()
                    st.success("📈 New signals generated!")

        with col3:
            if st.button(
                "💰 Rebalance Portfolio", help="Auto-rebalance based on strategy"
            ):
                with st.spinner("Rebalancing portfolio..."):
                    import time

                    time.sleep(1.5)
                    st.success("⚖️ Portfolio rebalanced")

        # Current positions with enhanced details
        st.subheader("📋 Current Positions")

        positions_data = []
        for symbol, pos in demo_positions.items():
            pnl_pct = ((pos["current_price"] / pos["entry_price"]) - 1) * 100
            positions_data.append(
                {
                    "Symbol": symbol,
                    "Quantity": pos["quantity"],
                    "Entry Price": f"${pos['entry_price']:.2f}",
                    "Current Price": f"${pos['current_price']:.2f}",
                    "Market Value": f"${pos['current_price'] * pos['quantity']:,.2f}",
                    "P&L": f"${pos['pnl']:+,.2f}",
                    "Return %": f"{pnl_pct:+.1f}%",
                    "Status": "🟢 Profitable"
                    if pos["pnl"] > 0
                    else "🔴 Loss"
                    if pos["pnl"] < 0
                    else "➡️ Flat",
                }
            )

        st.dataframe(positions_data, use_container_width=True)

        # Position management tools
        st.subheader("🛠️ Position Management")

        selected_symbol = st.selectbox(
            "Select position to manage:", list(demo_positions.keys())
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(f"📈 Add to {selected_symbol}", help="Increase position size"):
                st.info(f"📈 Order placed: BUY {selected_symbol}")

        with col2:
            if st.button(f"📉 Reduce {selected_symbol}", help="Decrease position size"):
                st.info(f"📉 Order placed: SELL {selected_symbol}")

        with col3:
            if st.button(f"❌ Close {selected_symbol}", help="Close entire position"):
                st.warning(f"❌ Position closed: {selected_symbol}")

    else:
        # Live trading mode placeholder
        with col1:
            st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.2f}")
        with col2:
            st.metric("Daily P&L", f"${st.session_state.daily_pnl:+,.2f}")
        with col3:
            st.metric("Active Positions", len(st.session_state.positions))
        with col4:
            st.metric(
                "Trading Status",
                "🟢 Live" if st.session_state.trading_active else "🔴 Stopped",
            )

        st.info("🔌 Connect to live portfolio data for full functionality")


def display_market_data():
    """Display market data and charts"""

    st.header("📈 Market Data")

    # Improved symbol selection with more symbols
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_symbol = st.selectbox(
            "Select Symbol",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
        )

    with col2:
        time_period = st.selectbox(
            "Time Period", ["1 Month", "3 Months", "6 Months", "1 Year"]
        )

    with col3:
        chart_type = st.selectbox("Chart Type", ["Line Chart", "Candlestick", "Volume"])

    # Fetch real-time data button
    if st.button("🔄 Fetch Live Data", help="Click to get real-time market data"):
        with st.spinner(f"Fetching real-time data for {selected_symbol}..."):
            try:
                import yfinance as yf

                # Map time periods
                period_map = {
                    "1 Month": "1mo",
                    "3 Months": "3mo",
                    "6 Months": "6mo",
                    "1 Year": "1y",
                }

                # Fetch real data
                ticker = yf.Ticker(selected_symbol)
                real_data = ticker.history(period=period_map[time_period])

                if not real_data.empty:
                    st.session_state.live_data = real_data
                    st.session_state.current_symbol = selected_symbol
                    st.success(
                        f"✅ Fetched {len(real_data)} days of data for {selected_symbol}"
                    )
                else:
                    st.error("❌ No data found for this symbol")

            except Exception as e:
                st.error(f"❌ Error fetching data: {e}")

    # Display data (live if available, otherwise demo)
    data_to_show = None
    data_source = "Demo"

    if hasattr(st.session_state, "live_data") and hasattr(
        st.session_state, "current_symbol"
    ):
        if st.session_state.current_symbol == selected_symbol:
            data_to_show = st.session_state.live_data
            data_source = "Live"

    # Fall back to demo data if no live data
    if data_to_show is None:
        if st.session_state.demo_mode:
            _, demo_data = create_demo_data()
            if selected_symbol in demo_data:
                data_to_show = demo_data[selected_symbol]
            else:
                # Generate demo data for any symbol
                st.info(f"Generating demo data for {selected_symbol}...")
                data_to_show = generate_demo_data_for_symbol(
                    selected_symbol, time_period
                )

    # Create charts if we have data
    if data_to_show is not None and PLOTLY_AVAILABLE:
        # Chart title with data source indicator
        chart_title = f"{selected_symbol} - {chart_type} ({data_source} Data)"

        # Create different chart types
        if chart_type == "Line Chart":
            fig = create_line_chart(data_to_show, selected_symbol, chart_title)
        elif chart_type == "Candlestick":
            fig = create_candlestick_chart(data_to_show, selected_symbol, chart_title)
        elif chart_type == "Volume":
            fig = create_volume_chart(data_to_show, selected_symbol, chart_title)

        st.plotly_chart(fig, use_container_width=True)

        # Show current price and statistics
        display_price_statistics(data_to_show, selected_symbol, data_source)

        # Recent data table
        st.subheader("Recent Data")
        recent_data = data_to_show.tail(10).round(2)
        st.dataframe(recent_data, use_container_width=True)

        # Technical indicators
        if st.checkbox("Show Technical Indicators", key="show_tech_indicators"):
            display_technical_indicators(data_to_show, selected_symbol)

    elif not PLOTLY_AVAILABLE:
        st.warning("Charts require plotly installation")
    else:
        st.info("No data available for the selected symbol")


def display_risk_management():
    """Display enhanced risk management with live controls"""

    st.header("⚠️ Risk Management Center")

    # Risk control panel
    st.subheader("🎛️ Risk Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        max_position = st.slider(
            "Max Position Size (%)",
            1,
            25,
            10,
            1,
            help="Maximum percentage of portfolio per position",
        )

    with col2:
        stop_loss = st.slider(
            "Stop Loss (%)", 1, 20, 5, 1, help="Automatic stop loss percentage"
        )

    with col3:
        take_profit = st.slider(
            "Take Profit (%)", 5, 50, 15, 5, help="Automatic take profit percentage"
        )

    # Risk monitoring toggles
    col1, col2, col3 = st.columns(3)

    with col1:
        correlation_check = st.checkbox(
            "📊 Correlation Monitoring",
            value=True,
            help="Monitor portfolio correlation risk",
        )

    with col2:
        volatility_control = st.checkbox(
            "📈 Volatility Control",
            value=True,
            help="Adjust position sizes based on volatility",
        )

    with col3:
        emergency_stop = st.checkbox(
            "🚨 Emergency Stop Active",
            value=False,
            help="Halt all trading on large losses",
        )

    # Live risk metrics
    st.subheader("📊 Current Risk Assessment")

    col1, col2, col3, col4 = st.columns(4)

    if st.session_state.demo_mode:
        # Enhanced demo risk metrics
        import random

        random.seed(42)  # Consistent demo data

        # Calculate dynamic risk metrics
        portfolio_risk = random.uniform(1.5, 4.5)
        risk_color = (
            "🟢" if portfolio_risk < 3 else "🟡" if portfolio_risk < 4 else "🔴"
        )

        volatility = random.uniform(0.15, 0.35)
        vol_color = "🟢" if volatility < 0.25 else "🟡" if volatility < 0.3 else "🔴"

        correlation = random.uniform(0.2, 0.8)
        corr_color = "🟢" if correlation < 0.5 else "🟡" if correlation < 0.7 else "🔴"

        drawdown = random.uniform(0.5, 8.5)
        dd_color = "🟢" if drawdown < 5 else "🟡" if drawdown < 10 else "🔴"

        with col1:
            st.metric(
                "Portfolio Risk",
                f"{risk_color} {portfolio_risk:.1f}%",
                delta=f"Target: <3%",
                help="Overall portfolio risk level",
            )

        with col2:
            st.metric(
                "Volatility",
                f"{vol_color} {volatility:.1%}",
                delta=f"30-day avg",
                help="Portfolio volatility measure",
            )

        with col3:
            st.metric(
                "Correlation Risk",
                f"{corr_color} {correlation:.2f}",
                delta=f"Diversification score",
                help="Asset correlation analysis",
            )

        with col4:
            st.metric(
                "Max Drawdown",
                f"{dd_color} {drawdown:.1f}%",
                delta=f"Limit: 15%",
                help="Maximum portfolio decline",
            )

        # Risk breakdown by position
        st.subheader("🔍 Position Risk Analysis")

        demo_positions, _ = create_demo_data()

        risk_data = []
        for symbol, pos in demo_positions.items():
            position_value = pos["current_price"] * pos["quantity"]
            portfolio_pct = (position_value / st.session_state.portfolio_value) * 100

            # Simulate risk metrics per position
            beta = random.uniform(0.7, 1.8)
            var_95 = random.uniform(2.5, 8.0)

            risk_level = (
                "🟢 Low"
                if portfolio_pct < 15
                else "🟡 Medium"
                if portfolio_pct < 20
                else "🔴 High"
            )

            risk_data.append(
                {
                    "Symbol": symbol,
                    "Position Size": f"{portfolio_pct:.1f}%",
                    "Beta": f"{beta:.2f}",
                    "VaR (95%)": f"{var_95:.1f}%",
                    "Risk Level": risk_level,
                    "Action": "Monitor" if portfolio_pct < 20 else "Reduce",
                }
            )

        st.dataframe(risk_data, use_container_width=True)

        # Risk alerts and recommendations
        st.subheader("🚨 Risk Alerts & Recommendations")

        alerts = []
        if portfolio_risk > 4:
            alerts.append("⚠️ Portfolio risk above target - consider position reduction")
        if correlation > 0.7:
            alerts.append("⚠️ High correlation detected - diversify holdings")
        if drawdown > 10:
            alerts.append("🚨 Drawdown exceeds 10% - review strategy")
        if any(
            float(pos.split("%")[0]) > 20
            for pos in [r["Position Size"] for r in risk_data]
        ):
            alerts.append("⚠️ Position concentration risk - rebalance portfolio")

        if not alerts:
            st.success("✅ All risk metrics within acceptable ranges")
        else:
            for alert in alerts:
                st.warning(alert)

        # Risk management actions
        st.subheader("🛠️ Risk Management Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "📊 Run Risk Analysis", help="Comprehensive portfolio risk assessment"
            ):
                with st.spinner("Analyzing portfolio risk..."):
                    import time

                    time.sleep(2)
                    st.success("✅ Risk analysis complete - see alerts above")

        with col2:
            if st.button(
                "⚖️ Auto-Rebalance", help="Automatically rebalance based on risk rules"
            ):
                with st.spinner("Rebalancing portfolio..."):
                    import time

                    time.sleep(2)
                    st.success("✅ Portfolio rebalanced within risk limits")

        with col3:
            if st.button("🚨 Emergency Hedge", help="Apply emergency hedging strategy"):
                with st.spinner("Applying hedge positions..."):
                    import time

                    time.sleep(1.5)
                    st.success("✅ Emergency hedge positions applied")

        # Risk settings persistence
        if st.button("💾 Save Risk Settings", help="Save current risk parameters"):
            st.info(
                f"✅ Risk settings saved - Max Position: {max_position}%, Stop Loss: {stop_loss}%, Take Profit: {take_profit}%"
            )

    else:
        # Live mode placeholders
        with col1:
            st.metric("Portfolio Risk", "🟡 2.8%")
        with col2:
            st.metric("Volatility", "🟢 0.18%")
        with col3:
            st.metric("Correlation", "🟢 0.45")
        with col4:
            st.metric("Drawdown", "🟢 1.2%")

        st.info("🔌 Connect to live risk management system")


def display_trading_signals():
    """Display enhanced trading signals with ML integration"""

    st.header("📡 AI Trading Signals")

    # Signal generation controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "🧠 Generate New Signals", help="Run ML models to generate fresh signals"
        ):
            with st.spinner("🤖 AI analyzing market patterns..."):
                import time

                time.sleep(3)  # Simulate ML processing
                st.session_state.last_signal_time = datetime.now()
                st.success("✅ Fresh signals generated!")
                st.rerun()

    with col2:
        auto_signals = st.checkbox(
            "🔄 Auto-Generate",
            value=st.session_state.trading_active,
            help="Automatically generate signals every 15 minutes",
        )
        if auto_signals and not st.session_state.trading_active:
            st.info("Enable trading bot to activate auto-signals")

    with col3:
        signal_threshold = st.slider(
            "Signal Threshold",
            0.1,
            1.0,
            0.6,
            0.1,
            help="Minimum confidence for signal alerts",
        )

    # Last signal generation time
    if st.session_state.last_signal_time:
        time_since = datetime.now() - st.session_state.last_signal_time
        minutes_ago = int(time_since.total_seconds() / 60)
        st.caption(f"🕒 Last updated: {minutes_ago} minutes ago")
    else:
        st.caption("🕒 No signals generated yet")

    # Enhanced demo signals with ML-style data
    if st.session_state.demo_mode:
        st.subheader("🎯 Current Signals")

        # Simulate dynamic signals based on current time
        import random

        random.seed(int(datetime.now().hour))  # Consistent but changing signals

        enhanced_signals = [
            {
                "Symbol": "AAPL",
                "Signal": "BUY",
                "Confidence": 0.78,
                "Price": "$155.25",
                "Target": "$165.00",
                "StopLoss": "$148.00",
                "ML_Score": 8.5,
                "Technical": "Bullish",
                "Volume": "High",
                "Reason": "Strong momentum + earnings beat",
            },
            {
                "Symbol": "TSLA",
                "Signal": "SELL",
                "Confidence": 0.82,
                "Price": "$242.50",
                "Target": "$225.00",
                "StopLoss": "$255.00",
                "ML_Score": 9.1,
                "Technical": "Bearish",
                "Volume": "Very High",
                "Reason": "Overbought conditions + resistance",
            },
            {
                "Symbol": "MSFT",
                "Signal": "HOLD",
                "Confidence": 0.55,
                "Price": "$305.75",
                "Target": "$310.00",
                "StopLoss": "$295.00",
                "ML_Score": 6.2,
                "Technical": "Neutral",
                "Volume": "Medium",
                "Reason": "Mixed signals, await direction",
            },
            {
                "Symbol": "GOOGL",
                "Signal": "BUY",
                "Confidence": 0.71,
                "Price": "$2485.30",
                "Target": "$2650.00",
                "StopLoss": "$2350.00",
                "ML_Score": 7.8,
                "Technical": "Bullish",
                "Volume": "High",
                "Reason": "AI momentum + cloud growth",
            },
        ]

        # Filter by threshold
        filtered_signals = [
            s for s in enhanced_signals if s["Confidence"] >= signal_threshold
        ]

        for signal in filtered_signals:
            # Dynamic styling based on signal type
            if signal["Signal"] == "BUY":
                bg_color = "#d4edda"  # Light green
                border_color = "#28a745"
            elif signal["Signal"] == "SELL":
                bg_color = "#f8d7da"  # Light red
                border_color = "#dc3545"
            else:
                bg_color = "#e2e3e5"  # Light gray
                border_color = "#6c757d"

            # The content for the card will be built as a string
            card_content = f"""
            <div style='padding: 15px; border-left: 4px solid {border_color};
                       background-color: {bg_color}; border-radius: 5px; margin: 10px 0;'>
            """

            with st.container():
                # Use markdown to render the card's opening div
                st.markdown(
                    f"""
                    <div style='padding: 15px; border-left: 4px solid {border_color};
                               background-color: {bg_color}; border-radius: 5px; margin: 10px 0;'>
                    """,
                    unsafe_allow_html=True,
                )

                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    st.markdown(f"**{signal['Symbol']} - {signal['Signal']}**")
                    st.markdown(f"*{signal['Reason']}*")

                with col2:
                    confidence_color = (
                        "green"
                        if signal["Confidence"] > 0.7
                        else "orange"
                        if signal["Confidence"] > 0.5
                        else "red"
                    )
                    st.markdown(
                        f"**Confidence:** <span style='color: {confidence_color}'>{signal['Confidence']:.1%}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**ML Score:** {signal['ML_Score']}/10")

                with col3:
                    st.markdown(f"**Current:** {signal['Price']}")
                    st.markdown(f"**Target:** {signal['Target']}")
                    st.markdown(f"**Stop Loss:** {signal['StopLoss']}")

                with col4:
                    st.markdown(f"**Technical:** {signal['Technical']}")
                    st.markdown(f"**Volume:** {signal['Volume']}")

                    # Action button
                    if signal["Confidence"] >= 0.7:
                        if st.button(
                            f"Execute {signal['Signal']}",
                            key=f"exec_{signal['Symbol']}",
                            help=f"Place {signal['Signal']} order for {signal['Symbol']}",
                        ):
                            st.success(
                                f" {signal['Signal']} order placed for {signal['Symbol']}"
                            )

                # Close the div
                st.markdown("</div>", unsafe_allow_html=True)

        # Signal summary statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            buy_signals = len([s for s in filtered_signals if s["Signal"] == "BUY"])
            st.metric("🟢 BUY Signals", buy_signals)

        with col2:
            sell_signals = len([s for s in filtered_signals if s["Signal"] == "SELL"])
            st.metric("� SELL Signals", sell_signals)

        with col3:
            hold_signals = len([s for s in filtered_signals if s["Signal"] == "HOLD"])
            st.metric("🟡 HOLD Signals", hold_signals)

        with col4:
            avg_confidence = (
                sum(s["Confidence"] for s in filtered_signals) / len(filtered_signals)
                if filtered_signals
                else 0
            )
            st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")

    else:
        st.info("🔌 Connect ML models for live signal generation")


def create_performance_chart():
    """Create interactive performance visualization"""
    if not PLOTLY_AVAILABLE:
        return None

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Generate sample performance data
    dates = pd.date_range(start="2024-01-01", end="2024-10-11", freq="D")
    np.random.seed(42)

    # Simulate portfolio performance
    returns = np.random.normal(0.0008, 0.02, len(dates))  # Slightly positive bias
    portfolio_values = [100000]  # Starting value

    for return_rate in returns[1:]:
        portfolio_values.append(portfolio_values[-1] * (1 + return_rate))

    # Benchmark performance (S&P 500 simulation)
    benchmark_returns = np.random.normal(0.0005, 0.015, len(dates))
    benchmark_values = [100000]

    for return_rate in benchmark_returns[1:]:
        benchmark_values.append(benchmark_values[-1] * (1 + return_rate))

    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Portfolio Performance vs S&P 500", "Daily Returns"),
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
    )

    # Portfolio value line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_values,
            name="AI Trading Bot",
            line=dict(color="#00D4AA", width=3),
            hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Benchmark line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=benchmark_values,
            name="S&P 500 Benchmark",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Daily returns bar chart
    daily_returns = [
        (portfolio_values[i] / portfolio_values[i - 1] - 1) * 100
        for i in range(1, len(portfolio_values))
    ]

    colors = ["green" if ret > 0 else "red" for ret in daily_returns]

    fig.add_trace(
        go.Bar(
            x=dates[1:],
            y=daily_returns,
            name="Daily Returns (%)",
            marker_color=colors,
            opacity=0.7,
            hovertemplate="<b>Daily Return</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    return fig


def display_trading_engine():
    """Display Live Trading Engine controls and status"""

    st.header("🤖 Live Trading Engine")

    if not TRADING_ENGINE_LOADED:
        st.error("❌ Trading Engine not available - missing dependencies")
        st.info("Install required components to enable live trading functionality")
        return

    # Get current engine status
    if "trading_engine" in st.session_state:
        engine = st.session_state.trading_engine
        status = engine.get_status()
        st.session_state.engine_status = status
    else:
        status = st.session_state.engine_status

    # Engine Control Panel
    st.subheader("🎛️ Engine Control Panel")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Start/Stop Controls
        if status["state"] in ["stopped", "error"]:
            if st.button(
                "🚀 Start Trading Engine",
                type="primary",
                help="Start the automated trading engine",
            ):
                if "trading_engine" in st.session_state:
                    with st.spinner("Starting trading engine..."):
                        result = st.session_state.trading_engine.start_trading()
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                            st.session_state.trading_active = True
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")

        elif status["state"] == "running":
            if st.button(
                "🛑 Stop Trading Engine",
                type="secondary",
                help="Stop the automated trading engine",
            ):
                if "trading_engine" in st.session_state:
                    with st.spinner("Stopping trading engine..."):
                        result = st.session_state.trading_engine.stop_trading()
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                            st.session_state.trading_active = False
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")

        elif status["state"] == "paused":
            if st.button("▶️ Resume Trading", help="Resume paused trading"):
                if "trading_engine" in st.session_state:
                    result = st.session_state.trading_engine.resume_trading()
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.rerun()

    with col2:
        # Pause/Resume for running engine
        if status["state"] == "running":
            if st.button("⏸️ Pause Trading", help="Temporarily pause trading"):
                if "trading_engine" in st.session_state:
                    result = st.session_state.trading_engine.pause_trading()
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.rerun()

    with col3:
        # Refresh Status
        if st.button("🔄 Refresh Status", help="Update engine status"):
            st.rerun()

    # Engine Status Display
    st.subheader("📊 Engine Status")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        state_colors = {
            "stopped": "🔴",
            "starting": "🟡",
            "running": "🟢",
            "paused": "🟡",
            "error": "❌",
        }
        state_color = state_colors.get(status["state"], "⚪")
        st.metric("Engine State", f"{state_color} {status['state'].title()}")

    with col2:
        uptime_str = f"{status.get('uptime_hours', 0):.1f}h"
        st.metric("Uptime", uptime_str)

    with col3:
        st.metric("Active Positions", status.get("positions_count", 0))

    with col4:
        win_rate = status.get("win_rate", 0)
        st.metric("Win Rate", f"{win_rate:.1f}%")

    with col5:
        pnl = status.get("total_pnl", 0)
        pnl_color = "normal" if pnl >= 0 else "inverse"
        st.metric("Total P&L", f"${pnl:+,.2f}", delta_color=pnl_color)

    # Live Positions (if engine is available)
    if "trading_engine" in st.session_state and status["state"] != "stopped":
        st.subheader("📋 Live Positions")

        try:
            positions = st.session_state.trading_engine.get_positions()

            if positions:
                positions_df = []
                for pos in positions:
                    positions_df.append(
                        {
                            "Symbol": pos["symbol"],
                            "Quantity": pos["quantity"],
                            "Entry Price": f"${pos['entry_price']:.2f}",
                            "Current Price": f"${pos['current_price']:.2f}",
                            "P&L": f"${pos['pnl']:+,.2f}",
                            "P&L %": f"{pos['pnl_pct']:+.1f}%",
                            "Entry Time": pos["entry_time"][:19],  # Remove microseconds
                            "Stop Loss": f"${pos['stop_loss']:.2f}",
                            "Take Profit": f"${pos['take_profit']:.2f}",
                        }
                    )

                st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("No active positions")

        except Exception as e:
            st.warning(f"⚠️ Could not fetch positions: {e}")

    # Recent Trading Activity
    if "trading_engine" in st.session_state:
        st.subheader("📡 Recent Signals")

        try:
            recent_signals = st.session_state.trading_engine.get_recent_signals(
                limit=10
            )

            if recent_signals:
                signals_df = []
                for signal in recent_signals[-5:]:  # Show last 5
                    signals_df.append(
                        {
                            "Time": signal["timestamp"][:19],
                            "Symbol": signal["symbol"],
                            "Signal": signal["signal"],
                            "Confidence": f"{signal['confidence']:.1%}",
                            "Price": f"${signal['price']:.2f}",
                        }
                    )

                st.dataframe(signals_df, use_container_width=True)
            else:
                st.info("No recent signals")

        except Exception as e:
            st.warning(f"⚠️ Could not fetch signals: {e}")

    # Trading Configuration
    st.subheader("⚙️ Trading Configuration")

    with st.expander("Engine Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            symbols = st.multiselect(
                "Trading Symbols",
                options=TRADING_SYMBOLS,
                default=TRADING_SYMBOLS[:5],
                help="Select symbols for automated trading",
            )

            max_positions = st.slider(
                "Max Positions",
                1,
                10,
                5,
                help="Maximum number of simultaneous positions",
            )

            position_size = st.slider(
                "Position Size (%)",
                5,
                50,
                20,
                help="Percentage of portfolio per position",
            )

        with col2:
            stop_loss = st.slider(
                "Stop Loss (%)", 1, 15, 5, help="Automatic stop loss percentage"
            )

            take_profit = st.slider(
                "Take Profit (%)", 5, 50, 15, help="Automatic take profit percentage"
            )

            signal_threshold = st.slider(
                "Signal Threshold",
                0.1,
                1.0,
                0.6,
                0.1,
                help="Minimum confidence for signal execution",
            )

        # Update configuration button
        if st.button(
            "💾 Update Configuration", help="Apply new settings to trading engine"
        ):
            if "trading_engine" in st.session_state:
                new_config = {
                    "symbols": symbols,
                    "max_positions": max_positions,
                    "position_size_pct": position_size / 100,
                    "stop_loss_pct": stop_loss / 100,
                    "take_profit_pct": take_profit / 100,
                    "signal_threshold": signal_threshold,
                }

                result = st.session_state.trading_engine.update_config(new_config)
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"❌ {result['message']}")
            else:
                st.warning("Trading engine not available")

    # Error Display
    if status.get("errors_count", 0) > 0:
        st.subheader("⚠️ Errors & Warnings")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Error Count", status["errors_count"])
        with col2:
            if status.get("last_error"):
                st.error(f"Last Error: {status['last_error']}")


def display_real_time_monitoring():
    """Display real-time monitoring dashboard"""

    st.header("🔍 Real-Time Monitoring System")

    if not MONITORING_SYSTEM_LOADED:
        st.error("❌ Real-Time Monitoring System not available")
        st.info(
            "The monitoring system requires additional dependencies and proper imports."
        )
        return

    if "monitor" not in st.session_state:
        st.error("❌ Monitor not initialized")
        return

    monitor = st.session_state.monitor

    # Monitoring Control Panel
    st.subheader("🎛️ Control Panel")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Start/Stop Monitoring
        if st.button(
            "🚀 Start Monitoring",
            disabled=st.session_state.monitoring_active,
            help="Start real-time monitoring",
        ):
            try:
                result = monitor.start_monitoring()
                if result["success"]:
                    st.session_state.monitoring_active = True
                    st.success(f"✅ {result['message']}")
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
            except Exception as e:
                st.error(f"❌ Failed to start monitoring: {e}")

    with col2:
        if st.button(
            "⏹️ Stop Monitoring",
            disabled=not st.session_state.monitoring_active,
            help="Stop real-time monitoring",
        ):
            try:
                result = monitor.stop_monitoring()
                if result["success"]:
                    st.session_state.monitoring_active = False
                    st.success(f"✅ {result['message']}")
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
            except Exception as e:
                st.error(f"❌ Failed to stop monitoring: {e}")

    with col3:
        # Refresh Dashboard
        if st.button("🔄 Refresh Dashboard", help="Refresh monitoring data"):
            st.rerun()

    # Monitoring Status
    st.subheader("📊 System Status")

    try:
        status = monitor.get_monitoring_status()
        dashboard_data = monitor.get_dashboard_data()

        # Status Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status_icon = "🟢 Running" if status["is_running"] else "🔴 Stopped"
            st.metric("Monitor Status", status_icon)

        with col2:
            active_alerts = status.get("active_alerts_count", 0)
            alert_color = "🔴" if active_alerts > 0 else "🟢"
            st.metric("Active Alerts", f"{alert_color} {active_alerts}")

        with col3:
            total_alerts = status.get("total_alerts_count", 0)
            st.metric("Total Alerts", total_alerts)

        with col4:
            current_metrics = status.get("current_metrics", {})
            portfolio_value = current_metrics.get("total_value", 0)
            st.metric("Portfolio Value", f"${portfolio_value:,.2f}")

        # Active Alerts Section
        st.subheader("🚨 Active Alerts")

        active_alerts = dashboard_data.get("active_alerts", [])

        if active_alerts:
            for alert in active_alerts[-5:]:  # Show last 5 active alerts
                level = alert["level"]
                alert_type = alert["alert_type"]
                title = alert["title"]
                message = alert["message"]
                timestamp = alert["timestamp"]

                # Alert styling based on level
                if level == "critical":
                    alert_class = "🔴"
                elif level == "warning":
                    alert_class = "🟡"
                elif level == "info":
                    alert_class = "🔵"
                else:
                    alert_class = "⚪"

                with st.expander(
                    f"{alert_class} {title}",
                    expanded=level in ["critical", "emergency"],
                ):
                    st.write(f"**Level:** {level.upper()}")
                    st.write(f"**Type:** {alert_type}")
                    st.write(f"**Time:** {timestamp}")
                    st.write(f"**Message:** {message}")

                    # Alert actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✓ Acknowledge", key=f"ack_{alert['id']}"):
                            monitor.acknowledge_alert(alert["id"])
                            st.success("Alert acknowledged")
                            st.rerun()
                    with col2:
                        if st.button(f"✅ Resolve", key=f"resolve_{alert['id']}"):
                            monitor.resolve_alert(alert["id"])
                            st.success("Alert resolved")
                            st.rerun()
        else:
            st.success("🟢 No active alerts - all systems operating normally")

        # Performance Metrics
        st.subheader("📈 Real-Time Performance")

        if current_metrics:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                unrealized_pnl = current_metrics.get("unrealized_pnl", 0)
                pnl_color = "green" if unrealized_pnl >= 0 else "red"
                st.markdown(
                    f"**Unrealized P&L:** <span style='color: {pnl_color}'>${unrealized_pnl:+,.2f}</span>",
                    unsafe_allow_html=True,
                )

            with col2:
                win_rate = current_metrics.get("win_rate", 0)
                st.metric("Win Rate", f"{win_rate:.1%}")

            with col3:
                total_trades = current_metrics.get("total_trades", 0)
                st.metric("Total Trades", total_trades)

            with col4:
                position_count = current_metrics.get("position_count", 0)
                st.metric("Active Positions", position_count)

        # Performance Trends Chart
        if PLOTLY_AVAILABLE:
            st.subheader("📊 Performance Trends")

            trends = dashboard_data.get("performance_trends", {})

            if trends and trends.get("timestamps"):
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=2,
                        cols=2,
                        subplot_titles=(
                            "Portfolio Value",
                            "P&L",
                            "Position Count",
                            "Drawdown",
                        ),
                        vertical_spacing=0.08,
                    )

                    timestamps = trends["timestamps"]

                    # Portfolio value
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=trends.get("portfolio_values", []),
                            name="Portfolio Value",
                            line=dict(color="blue"),
                        ),
                        row=1,
                        col=1,
                    )

                    # P&L
                    pnl_values = trends.get("pnl_values", [])
                    colors = ["green" if pnl >= 0 else "red" for pnl in pnl_values]
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=pnl_values,
                            name="P&L",
                            line=dict(color="green"),
                            fill="tozeroy",
                        ),
                        row=1,
                        col=2,
                    )

                    # Position count
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=trends.get("position_counts", []),
                            name="Position Count",
                            line=dict(color="orange"),
                        ),
                        row=2,
                        col=1,
                    )

                    # Drawdown (if available)
                    if trends.get("drawdown_values"):
                        fig.add_trace(
                            go.Scatter(
                                x=timestamps,
                                y=trends.get("drawdown_values", []),
                                name="Drawdown",
                                line=dict(color="red"),
                                fill="tozeroy",
                            ),
                            row=2,
                            col=2,
                        )

                    fig.update_layout(
                        title="Real-Time Performance Monitoring",
                        height=600,
                        showlegend=False,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Failed to create performance chart: {e}")
            else:
                st.info(
                    "📊 Performance trend data will appear here as monitoring collects data"
                )

        # Recent Alerts History
        st.subheader("📜 Recent Alert History")

        recent_alerts = dashboard_data.get("recent_alerts", [])

        if recent_alerts:
            # Create a table of recent alerts
            alert_data = []
            for alert in recent_alerts[-10:]:  # Last 10 alerts
                alert_data.append(
                    {
                        "Time": alert["timestamp"].split("T")[1][:8],  # Just time part
                        "Level": alert["level"].upper(),
                        "Type": alert["alert_type"],
                        "Title": alert["title"],
                        "Status": "✅ Resolved" if alert["resolved"] else "🟡 Active",
                    }
                )

            df_alerts = pd.DataFrame(alert_data)
            st.dataframe(df_alerts, use_container_width=True, hide_index=True)
        else:
            st.info(
                "📜 Alert history will appear here as the monitoring system generates alerts"
            )

        # System Health Check
        st.subheader("🏥 System Health")

        health_col1, health_col2 = st.columns(2)

        with health_col1:
            st.write("**Trading Engine Connection:**")
            if "trading_engine" in st.session_state:
                engine_status = st.session_state.trading_engine.get_status()
                if engine_status.get("is_running"):
                    st.success("🟢 Connected and Running")
                else:
                    st.warning("🟡 Connected but Stopped")
            else:
                st.error("🔴 Not Connected")

        with health_col2:
            st.write("**Monitoring System:**")
            if status["is_running"]:
                st.success("🟢 Active and Monitoring")
            else:
                st.warning("🟡 Stopped")

        # Configuration Info
        with st.expander("⚙️ Monitoring Configuration", expanded=False):
            config_info = status.get("config", {})

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Alert Thresholds:**")
                st.write(
                    f"• P&L Warning: {config_info.get('pnl_warning_threshold', 0.05):.1%}"
                )
                st.write(
                    f"• P&L Critical: {config_info.get('pnl_critical_threshold', 0.10):.1%}"
                )
                st.write(
                    f"• Position Size Warning: {config_info.get('position_size_warning', 0.25):.1%}"
                )

            with col2:
                st.write("**Notification Settings:**")
                email_enabled = config_info.get("email_alerts_enabled", False)
                st.write(
                    f"• Email Alerts: {'✅ Enabled' if email_enabled else '❌ Disabled'}"
                )
                st.write("• Database Storage: ✅ Enabled")
                st.write("• Real-time Updates: ✅ Enabled")

    except Exception as e:
        st.error(f"❌ Error loading monitoring data: {e}")
        logger.error(f"Monitoring dashboard error: {e}")


def display_stock_screener():
    """Display stock growth screener dashboard"""

    st.header("🔎 Stock Growth Screener")
    st.markdown(
        "**Identify high-growth opportunities using advanced technical analysis, momentum indicators, and ML predictions**"
    )

    if not SCREENER_LOADED:
        st.error("❌ Stock Growth Screener not available")
        st.info(
            "The screener system requires additional dependencies and proper imports."
        )
        return

    if "screener" not in st.session_state:
        st.error("❌ Screener not initialized")
        return

    screener = st.session_state.screener

    # Control Panel
    st.subheader("🎛️ Screening Control Panel")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Run Full Scan
        if st.button(
            "🚀 Run Full Scan",
            help="Scan all stocks for growth opportunities",
            type="primary",
        ):
            with st.spinner("🔍 Scanning stocks for growth opportunities..."):
                try:
                    results = screener.run_full_scan()
                    st.session_state.scan_results = results
                    st.session_state.last_scan_time = datetime.now()
                    st.success(f"✅ Scan complete! Found {len(results)} opportunities")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Scan failed: {e}")

    with col2:
        # Quick Refresh
        if st.button("🔄 Refresh Results", help="Refresh screening results"):
            st.rerun()

    with col3:
        # Configuration
        if st.button("⚙️ Settings", help="Screener configuration"):
            st.session_state.show_screener_config = not st.session_state.get(
                "show_screener_config", False
            )
            st.rerun()

    # Configuration Panel (if enabled)
    if st.session_state.get("show_screener_config", False):
        with st.expander("⚙️ Screening Configuration", expanded=True):
            st.markdown("**Screening Thresholds:**")

            col1, col2 = st.columns(2)
            with col1:
                st.slider(
                    "Minimum Momentum Score",
                    0.0,
                    1.0,
                    0.6,
                    0.1,
                    help="Minimum momentum requirement",
                )
                st.slider(
                    "Volume Surge Threshold",
                    1.0,
                    5.0,
                    1.5,
                    0.1,
                    help="Volume increase multiplier",
                )
                st.slider(
                    "Technical Score Minimum",
                    0.0,
                    1.0,
                    0.5,
                    0.1,
                    help="Technical analysis threshold",
                )

            with col2:
                st.slider(
                    "Max Volatility",
                    0.1,
                    1.0,
                    0.6,
                    0.1,
                    help="Maximum acceptable volatility",
                )
                st.slider(
                    "RSI Recovery Level",
                    20.0,
                    50.0,
                    35.0,
                    5.0,
                    help="RSI oversold recovery level",
                )
                st.number_input(
                    "Min Daily Volume ($)",
                    100000,
                    10000000,
                    1000000,
                    help="Minimum liquidity",
                )

    # Scan Status and Summary
    st.subheader("📊 Screening Status")

    if st.session_state.get("last_scan_time"):
        scan_time = st.session_state.last_scan_time
        time_ago = datetime.now() - scan_time

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Last Scan", f"{time_ago.seconds // 60} min ago")

        with col2:
            scan_results = st.session_state.get("scan_results", {})
            st.metric("Stocks Scanned", len(scan_results))

        with col3:
            if scan_results:
                top_opportunities = [r for r in scan_results.values() if r.score >= 0.7]
                st.metric("High Potential", len(top_opportunities))

        with col4:
            if scan_results:
                explosive_growth = [
                    r
                    for r in scan_results.values()
                    if r.growth_category.value == "explosive_growth"
                ]
                st.metric("Explosive Growth", len(explosive_growth))
    else:
        st.info("📊 No recent scans available. Run a full scan to get started!")

    # Display Results
    if st.session_state.get("scan_results"):
        results = st.session_state.scan_results

        # Top Opportunities Section
        st.subheader("🎯 Top Growth Opportunities")

        # Filter controls
        col1, col2, col3 = st.columns(3)

        with col1:
            min_score = st.slider("Minimum Score", 0.0, 1.0, 0.6, 0.1)

        with col2:
            category_filter = st.selectbox(
                "Growth Category",
                [
                    "All",
                    "Explosive Growth",
                    "High Growth",
                    "Moderate Growth",
                    "Stable Growth",
                ],
            )

        with col3:
            max_results = st.number_input("Max Results", 5, 50, 20)

        # Filter results
        filtered_results = []
        for result in results.values():
            if result.score >= min_score:
                if category_filter == "All":
                    filtered_results.append(result)
                elif (
                    category_filter.lower().replace(" ", "_")
                    == result.growth_category.value
                ):
                    filtered_results.append(result)

        # Sort by score
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        top_results = filtered_results[:max_results]

        if top_results:
            # Results Table
            table_data = []
            for result in top_results:
                # Growth category emoji
                category_emojis = {
                    "explosive_growth": "🚀",
                    "high_growth": "📈",
                    "moderate_growth": "📊",
                    "stable_growth": "💹",
                    "low_growth": "📉",
                }
                category_emoji = category_emojis.get(result.growth_category.value, "📊")

                # Calculate potential return
                potential_return = ""
                if result.target_price and result.current_price > 0:
                    return_pct = (
                        result.target_price - result.current_price
                    ) / result.current_price
                    potential_return = f"{return_pct:.1%}"

                table_data.append(
                    {
                        "Symbol": result.symbol,
                        "Score": f"{result.score:.3f}",
                        "Category": f"{category_emoji} {result.growth_category.value.replace('_', ' ').title()}",
                        "Price": f"${result.current_price:.2f}",
                        "Target": f"${result.target_price:.2f}"
                        if result.target_price
                        else "-",
                        "Potential": potential_return,
                        "RSI": f"{result.rsi:.1f}" if result.rsi else "-",
                        "MACD": result.macd_signal or "-",
                        "Volume Ratio": f"{result.volume_ratio:.1f}x"
                        if result.volume_ratio
                        else "-",
                    }
                )

            df_results = pd.DataFrame(table_data)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Detailed Analysis for Top Pick
            if top_results:
                st.subheader("🏆 Top Pick Analysis")

                top_pick = top_results[0]

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"### {top_pick.symbol}")
                    st.markdown(
                        f"**Score:** {top_pick.score:.3f} | **Category:** {top_pick.growth_category.value.replace('_', ' ').title()}"
                    )

                    # Key metrics
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    with metrics_col1:
                        st.metric("Current Price", f"${top_pick.current_price:.2f}")
                        if top_pick.rsi:
                            st.metric("RSI", f"{top_pick.rsi:.1f}")

                    with metrics_col2:
                        if top_pick.target_price:
                            st.metric("Target Price", f"${top_pick.target_price:.2f}")
                        if top_pick.price_change_1w:
                            st.metric("1W Change", f"{top_pick.price_change_1w:.1%}")

                    with metrics_col3:
                        if top_pick.target_price:
                            upside = (
                                top_pick.target_price - top_pick.current_price
                            ) / top_pick.current_price
                            st.metric("Upside Potential", f"{upside:.1%}")
                        if top_pick.volume_ratio:
                            st.metric("Volume Surge", f"{top_pick.volume_ratio:.1f}x")

                with col2:
                    # Component Scores
                    st.markdown("**Component Scores:**")

                    score_data = {
                        "Technical": top_pick.technical_score,
                        "Momentum": top_pick.momentum_score,
                        "Volume": top_pick.volume_score,
                        "ML Prediction": top_pick.ml_prediction,
                    }

                    for component, score in score_data.items():
                        color = (
                            "green"
                            if score > 0.6
                            else "orange"
                            if score > 0.3
                            else "red"
                        )
                        st.markdown(
                            f"• **{component}:** <span style='color: {color}'>{score:.3f}</span>",
                            unsafe_allow_html=True,
                        )

                # Criteria Met
                if top_pick.criteria_met:
                    st.markdown("**✅ Criteria Met:**")
                    criteria_text = ", ".join(
                        [
                            c.value.replace("_", " ").title()
                            for c in top_pick.criteria_met
                        ]
                    )
                    st.success(criteria_text)

                # Risk Factors
                if top_pick.risk_factors:
                    st.markdown("**⚠️ Risk Factors:**")
                    for risk in top_pick.risk_factors:
                        st.warning(f"• {risk}")

        else:
            st.info(
                f"📊 No results match the current filters (minimum score: {min_score:.1f})"
            )

        # Category Breakdown Chart
        if PLOTLY_AVAILABLE and results:
            st.subheader("📊 Growth Category Distribution")

            category_counts = defaultdict(int)
            for result in results.values():
                category_counts[result.growth_category.value] = (
                    category_counts.get(result.growth_category.value, 0) + 1
                )

            try:
                import plotly.graph_objects as go

                labels = [
                    cat.replace("_", " ").title() for cat in category_counts.keys()
                ]
                values = list(category_counts.values())

                colors = ["#ff4444", "#ff8800", "#ffcc00", "#88cc00", "#00cc88"]

                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=labels, values=values, marker_colors=colors, hole=0.3
                        )
                    ]
                )

                fig.update_layout(title="Growth Stock Distribution", height=400)

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Failed to create distribution chart: {e}")

        # Performance Statistics
        st.subheader("📈 Screening Statistics")

        if results:
            scores = [r.score for r in results.values()]

            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

            with stats_col1:
                st.metric("Average Score", f"{np.mean(scores):.3f}")

            with stats_col2:
                st.metric("Median Score", f"{np.median(scores):.3f}")

            with stats_col3:
                st.metric("Highest Score", f"{np.max(scores):.3f}")

            with stats_col4:
                high_potential = len([s for s in scores if s >= 0.7])
                st.metric("High Potential (>0.7)", high_potential)

    else:
        st.info(
            "🔍 No screening results available yet. Click 'Run Full Scan' to get started!"
        )

        # Sample stocks preview
        st.subheader("📋 Sample Screening Universe")
        st.markdown("The screener will analyze these stocks and many more:")

        sample_stocks = [
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
            "ROKU",
            "ZOOM",
            "SNOW",
            "PLTR",
        ]

        cols = st.columns(7)
        for i, symbol in enumerate(sample_stocks[:14]):
            with cols[i % 7]:
                st.code(symbol, language=None)


def display_performance_analytics():
    """Display performance analytics dashboard"""

    st.header("📈 Performance Analytics")

    # Performance summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_return = 12.5  # Demo value
        st.metric("Total Return", f"{total_return:.1f}%", delta="vs benchmark +3.2%")

    with col2:
        sharpe_ratio = 1.47
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", delta="Risk-adjusted return")

    with col3:
        max_dd = -5.8
        st.metric("Max Drawdown", f"{max_dd:.1f}%", delta="Peak to trough")

    with col4:
        win_rate = 68.5
        st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"142/208 trades")

    # Performance chart
    if PLOTLY_AVAILABLE:
        perf_chart = create_performance_chart()
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
    else:
        st.warning("📊 Performance charts require plotly installation")

    # Strategy breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Strategy Performance")
        strategy_data = {
            "Strategy": ["Momentum", "Mean Reversion", "ML Ensemble", "Risk Parity"],
            "Allocation": ["35%", "25%", "30%", "10%"],
            "Return": ["15.2%", "8.7%", "18.5%", "6.1%"],
            "Trades": [45, 32, 38, 12],
        }
        st.dataframe(strategy_data, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🎯 Monthly Performance")
        monthly_data = {
            "Month": ["Aug 2024", "Sep 2024", "Oct 2024"],
            "Return": ["2.3%", "1.8%", "3.1%"],
            "Benchmark": ["1.5%", "0.9%", "2.2%"],
            "Alpha": ["+0.8%", "+0.9%", "+0.9%"],
        }
        st.dataframe(monthly_data, use_container_width=True, hide_index=True)


def main():
    """Main dashboard application"""

    # Header with real-time status
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🤖 AI Day Trading Bot")
        st.markdown("**Intelligent Trading with Risk Management**")

    with col2:
        # Live clock and market status
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"🕒 **{current_time}**")

        # Simulate market hours (9:30 AM - 4:00 PM ET)
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 16:
            st.markdown("🟢 **Market Open**")
        else:
            st.markdown("🔴 **Market Closed**")

    # Deployment banner
    if st.session_state.demo_mode:
        st.info("🎮 **Demo Mode**: Showcasing capabilities with simulated data")

    # System status
    display_system_status()

    # Enhanced Navigation
    with st.sidebar:
        st.markdown("---")
        # Navigation options based on available features
        nav_options = [
            "📊 Dashboard",
            "💼 Portfolio",
            "📈 Market Data",
            "⚠️ Risk Management",
            "📡 Trading Signals",
            "🏆 Performance",
        ]

        # Add Small Account option if available
        if MODULES_LOADED["small_account"]:
            nav_options.insert(1, "💰 Small Account")  # Insert after Dashboard

        if TRADING_ENGINE_LOADED:
            nav_options.insert(
                -1, "🤖 Live Trading Engine"
            )  # Insert before Performance

        if MONITORING_SYSTEM_LOADED:
            nav_options.insert(
                -1, "🔍 Real-Time Monitoring"
            )  # Insert before Performance

        if SCREENER_LOADED:
            nav_options.insert(-1, "🔎 Stock Screener")  # Insert before Performance

        page = st.radio("🧭 Navigation:", nav_options)

        # Additional sidebar info
        st.markdown("---")
        st.markdown("### ⚡ Quick Stats")

        if st.session_state.demo_mode:
            demo_positions, _ = create_demo_data()
            total_pnl = sum(pos["pnl"] for pos in demo_positions.values())
            pnl_color = "green" if total_pnl >= 0 else "red"

            st.markdown(
                f"**P&L Today:** <span style='color: {pnl_color}'>${total_pnl:+,.2f}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Active Trades:** {len(demo_positions)}")
            st.markdown(
                f"**Bot Status:** {'🟢 Active' if st.session_state.trading_active else '🔴 Stopped'}"
            )

        # Quick actions
        st.markdown("---")
        st.markdown("### 🚀 Quick Actions")

        if st.button("🔄 Refresh All", help="Refresh all dashboard data"):
            st.success("✅ Dashboard refreshed!")
            st.rerun()

    # Main content routing
    if page == "📊 Dashboard":
        # Dashboard overview combining key metrics
        col1, col2 = st.columns([2, 1])

        with col1:
            display_portfolio_overview()

        with col2:
            st.subheader("🎯 Today's Highlights")

            # Mini trading signals
            st.markdown("**🔥 Hot Signals:**")
            st.success("📈 AAPL - Strong BUY (0.78)")
            st.error("📉 TSLA - SELL (0.82)")
            st.info("➡️ MSFT - HOLD (0.55)")

            # Mini risk summary
            st.markdown("**⚠️ Risk Status:**")
            st.markdown("🟢 Portfolio Risk: 2.8%")
            st.markdown("🟢 Correlation: Low")
            st.markdown("🟡 Volatility: Medium")

            # Quick market update
            st.markdown("**📊 Market Snapshot:**")
            st.markdown("• S&P 500: +0.8%")
            st.markdown("• VIX: 18.5 (-2.1%)")
            st.markdown("• 10Y Treasury: 4.35%")

    elif page == "💰 Small Account":
        display_small_account_dashboard()
    elif page == "💼 Portfolio":
        display_portfolio_overview()
    elif page == "📈 Market Data":
        display_market_data()
    elif page == "⚠️ Risk Management":
        display_risk_management()
    elif page == "📡 Trading Signals":
        display_trading_signals()
    elif page == "🤖 Live Trading Engine":
        display_trading_engine()
    elif page == "🔍 Real-Time Monitoring":
        display_real_time_monitoring()
    elif page == "🔎 Stock Screener":
        display_stock_screener()
    elif page == "🏆 Performance":
        display_performance_analytics()

    # Real-time notifications (if trading is active)
    if st.session_state.trading_active and page != "📊 Dashboard":
        with st.container():
            if st.session_state.demo_mode:
                # Simulate live notifications
                notification_placeholder = st.empty()

                import random

                notifications = [
                    "📈 AAPL position +2.1% - consider profit taking",
                    "⚠️ TSLA approaching stop loss level",
                    "🔔 New BUY signal generated for MSFT",
                    "📊 Portfolio rebalancing recommended",
                    "✅ Risk levels within targets",
                ]

                random.seed(int(datetime.now().minute))
                current_notification = random.choice(notifications)

                notification_placeholder.info(
                    f"🔔 **Live Update:** {current_notification}"
                )

    # Enhanced Footer
    st.markdown("---")

    # Performance summary bar
    if st.session_state.demo_mode:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown("**🎯 Success Rate**")
            st.markdown("68.5% Win Rate")

        with col2:
            st.markdown("**�️ Total P&L**")
            st.markdown("+$12,340.56")

        with col3:
            st.markdown("**📊 Sharpe Ratio**")
            st.markdown("1.47")

        with col4:
            st.markdown("**⚡ Uptime**")
            st.markdown("99.8%")

        with col5:
            st.markdown("**🔄 Last Update**")
            st.markdown("Just now")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**🔧 Technology Stack:**")
        st.markdown("• Python & Streamlit")
        st.markdown("• XGBoost & LightGBM")
        st.markdown("• Real-time APIs")

    with col2:
        st.markdown("**�️ Risk Controls:**")
        st.markdown("• Automated Stop-Loss")
        st.markdown("• Position Sizing")
        st.markdown("• Correlation Monitoring")

    with col3:
        st.markdown("**📡 Data Sources:**")
        st.markdown("• Yahoo Finance")
        st.markdown("• Technical Indicators")
        st.markdown("• ML Feature Engineering")

    with col4:
        st.markdown("**🚀 Current Status:**")
        status_text = "Demo Mode" if st.session_state.demo_mode else "Live Trading"
        bot_status = "Active" if st.session_state.trading_active else "Stopped"
        st.markdown(f"• Mode: {status_text}")
        st.markdown(f"• Bot: {bot_status}")
        st.markdown(f"• Version: 2.1.0")
