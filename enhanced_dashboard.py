"""
Enhanced AI Day Trading Bot Dashboard with Risk Management

Streamlit dashboard integrating ML signals, risk management, and portfolio tracking.
"""

import logging
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# plotly and numpy imports removed for cleaner code

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import our modules
try:
    from config.config import INITIAL_CAPITAL, LOG_LEVEL, TRADING_SYMBOLS
    from src.data_sources import DataManager
    from src.risk_management import (
        PortfolioManager,
        PositionSizingMethod,
        RiskManager,
        RiskParameters,
    )
    from src.signals import MLTechnicalSignalGenerator, SignalManager
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Day Trading Bot - Risk Management",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "portfolio_manager" not in st.session_state:
    st.session_state.portfolio_manager = None
if "data_manager" not in st.session_state:
    st.session_state.data_manager = None
if "signal_manager" not in st.session_state:
    st.session_state.signal_manager = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False


def initialize_managers():
    """Initialize data, signal, and portfolio managers"""

    if st.session_state.initialized:
        return

    try:
        # Initialize data manager
        st.session_state.data_manager = DataManager()

        # Initialize signal manager (it initializes with default generators)
        st.session_state.signal_manager = SignalManager()

        # Initialize portfolio manager with risk management
        risk_params = RiskParameters(
            max_portfolio_risk=0.02,  # 2% max risk per trade
            max_position_size=0.1,  # 10% max position size
            max_correlation=0.7,  # Max 70% correlation
            stop_loss_multiplier=2.0,
            max_drawdown_limit=0.15,  # 15% max drawdown
        )

        st.session_state.portfolio_manager = PortfolioManager(
            initial_cash=INITIAL_CAPITAL,
            risk_params=risk_params,
            data_manager=st.session_state.data_manager,
            signal_manager=st.session_state.signal_manager,
        )

        st.session_state.initialized = True
        st.success("✅ System initialized successfully!")

    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        return False

    return True


def display_portfolio_overview():
    """Display portfolio overview and metrics"""

    st.header("📊 Portfolio Overview")

    if not st.session_state.portfolio_manager:
        st.warning("Portfolio manager not initialized")
        return

    portfolio = st.session_state.portfolio_manager

    # Get portfolio summary
    try:
        summary = portfolio.get_portfolio_summary()

        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Value",
                value=f"${summary['total_value']:,.2f}",
                delta=f"{summary['total_return']:+.2%}",
            )

        with col2:
            st.metric(
                label="Cash Balance",
                value=f"${summary['cash_balance']:,.2f}",
                delta=None,
            )

        with col3:
            st.metric(
                label="Unrealized P&L",
                value=f"${summary['unrealized_pnl']:,.2f}",
                delta=None,
            )

        with col4:
            st.metric(
                label="Current Positions",
                value=summary["current_positions"],
                delta=None,
            )

        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                label="Max Drawdown", value=f"{summary['max_drawdown']:.2%}", delta=None
            )

        with col6:
            st.metric(label="Total Trades", value=summary["total_trades"], delta=None)

        with col7:
            st.metric(label="Win Rate", value=f"{summary['win_rate']:.1%}", delta=None)

        with col8:
            st.metric(
                label="Total Fees", value=f"${summary['total_fees']:.2f}", delta=None
            )

        # Position details
        if summary["positions"]:
            st.subheader("Current Positions")
            positions_df = pd.DataFrame(summary["positions"])

            # Format the dataframe for display
            positions_df["Position Value"] = positions_df["position_value"].apply(
                lambda x: f"${x:,.2f}"
            )
            positions_df["Entry Price"] = positions_df["entry_price"].apply(
                lambda x: f"${x:.2f}"
            )
            positions_df["Current Price"] = positions_df["current_price"].apply(
                lambda x: f"${x:.2f}"
            )
            positions_df["Unrealized P&L"] = positions_df["unrealized_pnl"].apply(
                lambda x: f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"
            )

            display_cols = [
                "symbol",
                "quantity",
                "Entry Price",
                "Current Price",
                "Position Value",
                "Unrealized P&L",
            ]
            st.dataframe(
                positions_df[display_cols].rename(
                    columns={"symbol": "Symbol", "quantity": "Quantity"}
                ),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Error displaying portfolio overview: {e}")


def display_risk_metrics():
    """Display risk management metrics and controls"""

    st.header("⚠️ Risk Management")

    if not st.session_state.portfolio_manager:
        st.warning("Portfolio manager not initialized")
        return

    portfolio = st.session_state.portfolio_manager
    risk_manager = portfolio.risk_manager

    # Risk parameters configuration
    st.subheader("Risk Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Current Risk Settings:**
        - Max Portfolio Risk: {risk_manager.risk_params.max_portfolio_risk:.1%}
        - Max Position Size: {risk_manager.risk_params.max_position_size:.1%}
        - Max Correlation: {risk_manager.risk_params.max_correlation:.1%}
        - Max Drawdown Limit: {risk_manager.risk_params.max_drawdown_limit:.1%}
        """)

    with col2:
        # Position sizing method selection
        sizing_method = st.selectbox(
            "Position Sizing Method",
            options=[method.value for method in PositionSizingMethod],
            index=2,  # Default to volatility-based
        )

    # Portfolio risk summary
    try:
        if portfolio.positions:
            # Get current prices
            current_prices = {}
            for symbol in portfolio.positions.keys():
                try:
                    data = st.session_state.data_manager.get_market_data(
                        symbol, period="1d", limit=1
                    )
                    if not data.empty:
                        current_prices[symbol] = data["Close"].iloc[-1]
                except Exception:
                    continue

            # Get portfolio risk summary
            positions = {
                symbol: pos.quantity for symbol, pos in portfolio.positions.items()
            }
            portfolio_value = portfolio.get_current_value(current_prices)

            risk_summary = risk_manager.get_portfolio_risk_summary(
                positions=positions,
                current_prices=current_prices,
                portfolio_value=portfolio_value,
            )

            st.subheader("Portfolio Risk Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Equity Exposure",
                    f"{risk_summary['total_equity_exposure']:.1%}",
                    delta=None,
                )

            with col2:
                st.metric(
                    "Diversification Score",
                    f"{risk_summary['risk_metrics']['diversification_score']:.2f}",
                    delta=None,
                )

            with col3:
                risk_level = risk_summary["risk_metrics"]["risk_level"]
                color = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴"}.get(
                    risk_level, "⚪"
                )
                st.metric("Risk Level", f"{color} {risk_level}", delta=None)

            # Risk warnings
            if risk_summary["warnings"]:
                st.warning("⚠️ Risk Warnings:")
                for warning in risk_summary["warnings"]:
                    st.write(f"• {warning}")

        else:
            st.info("No current positions to analyze")

    except Exception as e:
        st.error(f"Error calculating risk metrics: {e}")


def display_trading_signals():
    """Display current trading signals and recommendations"""

    st.header("📡 Trading Signals & Recommendations")

    if not st.session_state.portfolio_manager or not st.session_state.signal_manager:
        st.warning("System not fully initialized")
        return

    # Symbol selection
    symbols = st.multiselect(
        "Select symbols to analyze:",
        options=TRADING_SYMBOLS,
        default=TRADING_SYMBOLS[:3],  # Default to first 3 symbols
    )

    if not symbols:
        st.info("Please select at least one symbol to analyze")
        return

    # Generate signals and recommendations
    if st.button("🔄 Refresh Signals", type="primary"):
        with st.spinner("Generating trading signals..."):
            try:
                recommendations = st.session_state.portfolio_manager.process_signals(
                    symbols
                )

                if recommendations:
                    st.subheader("Trading Recommendations")

                    for symbol, rec in recommendations.items():
                        with st.expander(
                            f"{symbol} - {rec['action']} Signal", expanded=True
                        ):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Signal Strength",
                                    f"{rec['signal_strength']:.2f}",
                                    delta=None,
                                )

                            with col2:
                                st.metric(
                                    "Current Price",
                                    f"${rec['current_price']:.2f}",
                                    delta=None,
                                )

                            with col3:
                                st.metric(
                                    "Suggested Quantity",
                                    rec["suggested_quantity"],
                                    delta=None,
                                )

                            # Risk assessment
                            risk_assessment = rec.get("risk_assessment", {})
                            if risk_assessment:
                                recommendation = risk_assessment.get(
                                    "recommendation", "UNKNOWN"
                                )
                                color = {
                                    "PROCEED": "🟢",
                                    "CAUTION": "🟡",
                                    "REJECT": "🔴",
                                }.get(recommendation, "⚪")

                                st.write(
                                    f"**Risk Assessment:** {color} {recommendation}"
                                )

                                if risk_assessment.get("warnings"):
                                    st.warning("Risk Warnings:")
                                    for warning in risk_assessment["warnings"]:
                                        st.write(f"• {warning}")

                            # Trade details
                            if rec["action"] == "BUY":
                                st.write(f"""
                                **Trade Details:**
                                - Action: **{rec["action"]}**
                                - Quantity: **{rec["suggested_quantity"]} shares**
                                - Estimated Cost: **${rec["suggested_quantity"] * rec["current_price"]:,.2f}**
                                - Stop Loss: **${rec.get("stop_loss", 0):.2f}**
                                - Take Profit: **${rec.get("take_profit", 0):.2f}**
                                """)

                            elif rec["action"] == "SELL":
                                current_pos = rec.get("current_position")
                                if current_pos:
                                    unrealized_pnl = rec.get("unrealized_pnl", 0)
                                    pnl_color = "🟢" if unrealized_pnl >= 0 else "🔴"

                                    st.write(f"""
                                    **Position Details:**
                                    - Action: **{rec["action"]}**
                                    - Current Quantity: **{current_pos.quantity} shares**
                                    - Entry Price: **${current_pos.entry_price:.2f}**
                                    - Unrealized P&L: **{pnl_color} ${unrealized_pnl:,.2f}**
                                    """)

                            # Execute trade button
                            if st.button(
                                f"Execute {rec['action']} for {symbol}",
                                key=f"execute_{symbol}",
                            ):
                                st.info(
                                    "Trade execution would happen here (currently in demo mode)"
                                )

                else:
                    st.info("No strong trading signals detected at this time")

            except Exception as e:
                st.error(f"Error generating signals: {e}")


def display_market_data():
    """Display market data and charts"""

    st.header("📈 Market Data & Charts")

    if not st.session_state.data_manager:
        st.warning("Data manager not initialized")
        return

    # Symbol selection for charting
    chart_symbol = st.selectbox(
        "Select symbol for detailed chart:", options=TRADING_SYMBOLS, index=0
    )

    # Time period selection
    period = st.selectbox(
        "Time Period:",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        index=4,  # Default to 6 months
    )

    try:
        # Get market data
        data = st.session_state.data_manager.get_market_data(
            chart_symbol, period=period, limit=252
        )

        if not data.empty:
            # Create candlestick chart
            fig = go.Figure()

            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name=chart_symbol,
                )
            )

            fig.update_layout(
                title=f"{chart_symbol} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Volume chart
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"))

            vol_fig.update_layout(
                title=f"{chart_symbol} Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300,
            )

            st.plotly_chart(vol_fig, use_container_width=True)

            # Recent data table
            st.subheader("Recent Data")
            recent_data = data.tail(10).copy()
            recent_data = recent_data.round(2)
            st.dataframe(recent_data, use_container_width=True)

        else:
            st.error(f"No data available for {chart_symbol}")

    except Exception as e:
        st.error(f"Error loading market data: {e}")


def main():
    """Main dashboard application"""

    # Title and header
    st.title("🤖 AI Day Trading Bot - Enhanced Dashboard")
    st.markdown("**Integrated ML Signals & Risk Management**")

    # Initialize managers
    if not st.session_state.initialized:
        st.info("Initializing trading system...")
        if not initialize_managers():
            st.stop()

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Choose a page:",
            ["Portfolio Overview", "Risk Management", "Trading Signals", "Market Data"],
        )

        st.markdown("---")

        # System status
        st.subheader("System Status")
        if st.session_state.initialized:
            st.success("🟢 System Online")
        else:
            st.error("🔴 System Offline")

        # Quick stats
        if st.session_state.portfolio_manager:
            portfolio = st.session_state.portfolio_manager
            current_value = portfolio.get_current_value()
            st.metric("Portfolio Value", f"${current_value:,.2f}")
            st.metric("Positions", len(portfolio.positions))

    # Main content based on selected page
    if page == "Portfolio Overview":
        display_portfolio_overview()
    elif page == "Risk Management":
        display_risk_metrics()
    elif page == "Trading Signals":
        display_trading_signals()
    elif page == "Market Data":
        display_market_data()

    # Footer
    st.markdown("---")
    st.markdown("*AI Day Trading Bot - Enhanced with Risk Management v2.0*")


if __name__ == "__main__":
    main()
