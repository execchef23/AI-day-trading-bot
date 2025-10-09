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

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import our modules with error handling
MODULES_LOADED = {"data_manager": False, "signals": False, "risk_management": False}

try:
    from src.data_sources import DataManager

    MODULES_LOADED["data_manager"] = True
except ImportError as e:
    st.sidebar.warning(f"Data sources not available: {e}")

try:
    from src.signals import SignalManager

    MODULES_LOADED["signals"] = True
except ImportError as e:
    st.sidebar.warning(f"Signal generation not available: {e}")

try:
    from src.risk_management import (
        PortfolioManager,
        PositionSizingMethod,
        RiskParameters,
    )

    MODULES_LOADED["risk_management"] = True
except ImportError as e:
    st.sidebar.warning(f"Risk management not available: {e}")

# Configuration with fallbacks
try:
    from config.config import INITIAL_CAPITAL, TRADING_SYMBOLS
except ImportError:
    TRADING_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    INITIAL_CAPITAL = 100000.0

# Configure logging for deployment
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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


def display_system_status():
    """Display system status and deployment info"""

    st.sidebar.markdown("### 🚀 Deployment Status")

    # Module status
    for module, loaded in MODULES_LOADED.items():
        status = "✅" if loaded else "❌"
        st.sidebar.write(f"{status} {module.replace('_', ' ').title()}")

    # Deployment mode
    if st.session_state.demo_mode:
        st.sidebar.warning("🎮 Demo Mode Active")
        st.sidebar.info("Some features use simulated data for demonstration")
    else:
        st.sidebar.success("🔴 Live Mode Active")

    # Quick stats
    st.sidebar.markdown("---")
    st.sidebar.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.2f}")
    st.sidebar.metric("Active Positions", len(st.session_state.positions))


def display_portfolio_overview():
    """Display portfolio overview with demo/live data"""

    st.header("📊 Portfolio Overview")

    if st.session_state.demo_mode:
        demo_positions, _ = create_demo_data()

        # Calculate totals
        total_value = sum(
            pos["quantity"] * pos["current_price"] for pos in demo_positions.values()
        )
        cash_balance = st.session_state.portfolio_value - total_value
        total_pnl = sum(pos["pnl"] for pos in demo_positions.values())
        total_return = (
            st.session_state.portfolio_value + total_pnl - INITIAL_CAPITAL
        ) / INITIAL_CAPITAL

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Value",
                f"${st.session_state.portfolio_value + total_pnl:,.2f}",
                f"{total_return:+.2%}",
            )

        with col2:
            st.metric("Cash Balance", f"${cash_balance:,.2f}")

        with col3:
            st.metric("Unrealized P&L", f"${total_pnl:,.2f}")

        with col4:
            st.metric("Active Positions", len(demo_positions))

        # Position details
        st.subheader("Current Positions")

        positions_data = []
        for symbol, pos in demo_positions.items():
            positions_data.append(
                {
                    "Symbol": symbol,
                    "Quantity": pos["quantity"],
                    "Entry Price": f"${pos['entry_price']:.2f}",
                    "Current Price": f"${pos['current_price']:.2f}",
                    "Position Value": f"${pos['quantity'] * pos['current_price']:,.2f}",
                    "Unrealized P&L": f"${pos['pnl']:,.2f}",
                }
            )

        st.dataframe(pd.DataFrame(positions_data), use_container_width=True)

    else:
        st.info("Connect your portfolio manager for live data")


def display_market_data():
    """Display market data and charts"""

    st.header("📈 Market Data")

    # Symbol selection
    selected_symbol = st.selectbox("Select Symbol", TRADING_SYMBOLS[:3])

    if st.session_state.demo_mode:
        _, demo_data = create_demo_data()

        if selected_symbol in demo_data and PLOTLY_AVAILABLE:
            data = demo_data[selected_symbol]

            # Create price chart
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name=f"{selected_symbol} Price",
                    line=dict(color="#1f77b4", width=2),
                )
            )

            fig.update_layout(
                title=f"{selected_symbol} Price Chart (Demo Data)",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Recent data
            st.subheader("Recent Data")
            recent_data = data.tail(7).round(2)
            st.dataframe(recent_data, use_container_width=True)

        elif not PLOTLY_AVAILABLE:
            st.warning("Charts require plotly installation")

    else:
        st.info("Connect data manager for live market data")


def display_risk_management():
    """Display risk management info"""

    st.header("⚠️ Risk Management")

    # Risk parameters display
    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Risk Management Features:**
        - Position sizing algorithms
        - Stop-loss automation
        - Portfolio diversification
        - Correlation analysis
        - Drawdown protection
        """)

    with col2:
        st.success("""
        **Current Settings:**
        - Max position size: 10%
        - Max portfolio risk: 2%
        - Stop loss: 5%
        - Take profit: 10%
        - Max drawdown: 15%
        """)

    if st.session_state.demo_mode:
        # Demo risk metrics
        st.subheader("Portfolio Risk Metrics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Level", "🟢 LOW")
        with col2:
            st.metric("Diversification", "0.85")
        with col3:
            st.metric("Max Drawdown", "3.2%")

    else:
        st.info("Risk management active with live portfolio")


def display_trading_signals():
    """Display trading signals info"""

    st.header("📡 Trading Signals")

    if st.session_state.demo_mode:
        # Demo signals
        st.subheader("Latest Signals")

        demo_signals = [
            {"Symbol": "AAPL", "Signal": "BUY", "Strength": 0.75, "Price": "$155.25"},
            {"Symbol": "MSFT", "Signal": "HOLD", "Strength": 0.45, "Price": "$305.75"},
            {
                "Symbol": "GOOGL",
                "Signal": "SELL",
                "Strength": 0.68,
                "Price": "$2485.30",
            },
        ]

        for signal in demo_signals:
            with st.expander(f"{signal['Symbol']} - {signal['Signal']}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Signal Strength", f"{signal['Strength']:.2f}")
                with col2:
                    st.metric("Current Price", signal["Price"])

                if signal["Signal"] == "BUY":
                    st.success("📈 Strong buy signal detected")
                elif signal["Signal"] == "SELL":
                    st.error("📉 Sell signal detected")
                else:
                    st.info("➡️ Hold current position")

    else:
        st.info("Signal generation requires ML models")


def main():
    """Main dashboard application"""

    # Header
    st.title("🤖 AI Day Trading Bot")
    st.markdown("**Intelligent Trading with Risk Management**")

    # Deployment banner
    if st.session_state.demo_mode:
        st.info("🎮 **Demo Mode**: Showcasing capabilities with simulated data")

    # System status
    display_system_status()

    # Navigation
    with st.sidebar:
        st.markdown("---")
        page = st.radio(
            "Navigation:",
            ["Portfolio", "Market Data", "Risk Management", "Trading Signals"],
        )

    # Main content
    if page == "Portfolio":
        display_portfolio_overview()
    elif page == "Market Data":
        display_market_data()
    elif page == "Risk Management":
        display_risk_management()
    elif page == "Trading Signals":
        display_trading_signals()

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔧 Built with:**")
        st.markdown("• Streamlit • Python • ML")

    with col2:
        st.markdown("**📊 Features:**")
        st.markdown("• Risk Management • ML Signals")

    with col3:
        st.markdown("**🚀 Status:**")
        status = "Demo" if st.session_state.demo_mode else "Live"
        st.markdown(f"• {status} Mode Active")


if __name__ == "__main__":
    main()
