"""
AI Day Trading Bot Dashboard

Main Streamlit dashboard for monitoring trading signals, portfolio performance,
and controlling the trading bot.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import configuration
from config.config import (
    BASE_DIR, LOGS_DIR, LOG_LEVEL, TRADING_SYMBOLS, MAX_POSITION_SIZE,
    STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE, INITIAL_CAPITAL,
    FEATURE_WINDOW_DAYS, PREDICTION_HORIZON_DAYS, MODEL_RETRAIN_DAYS,
    ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY, QUANDL_API_KEY, ENVIRONMENT
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Day Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main dashboard application"""
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 AI Trading Bot")
        st.markdown("---")
        
        # Bot Status
        st.subheader("Bot Status")
        if st.button("🟢 Start Trading"):
            st.success("Trading bot started!")
        if st.button("🔴 Stop Trading"):
            st.warning("Trading bot stopped!")
        
        # Configuration
        st.subheader("Configuration")
        selected_symbols = st.multiselect(
            "Trading Symbols",
            options=TRADING_SYMBOLS,
            default=TRADING_SYMBOLS[:5]
        )
        
        max_position = st.slider(
            "Max Position Size (%)",
            min_value=1,
            max_value=20,
            value=int(MAX_POSITION_SIZE * 100)
        )
        
        stop_loss = st.slider(
            "Stop Loss (%)",
            min_value=1,
            max_value=10,
            value=int(STOP_LOSS_PERCENTAGE * 100)
        )
    
    # Main content
    st.title("📈 AI Day Trading Dashboard")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview",
        "🎯 Trading Signals", 
        "💰 Portfolio",
        "📈 Performance",
        "⚙️ Settings"
    ])
    
    with tab1:
        market_overview_tab()
    
    with tab2:
        trading_signals_tab()
    
    with tab3:
        portfolio_tab()
    
    with tab4:
        performance_tab()
    
    with tab5:
        settings_tab()

def market_overview_tab():
    """Market overview and watchlist"""
    st.header("Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("S&P 500", "4,200.50", "1.2%")
    with col2:
        st.metric("NASDAQ", "13,100.25", "0.8%")
    with col3:
        st.metric("VIX", "18.45", "-2.1%")
    with col4:
        st.metric("Market Status", "OPEN", "")
    
    # Placeholder for market data
    st.subheader("Top Movers")
    
    # Sample data - replace with real data
    sample_data = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'Price': [150.25, 280.50, 2650.75, 3200.25, 850.50],
        'Change': [2.5, -1.8, 15.25, -25.50, 35.75],
        'Change %': [1.69, -0.64, 0.58, -0.79, 4.38],
        'Volume': ['45.2M', '32.1M', '1.2M', '2.8M', '28.5M']
    })
    
    st.dataframe(sample_data, use_container_width=True)

def trading_signals_tab():
    """Current trading signals and recommendations"""
    st.header("Trading Signals")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Active Signals")
        
        # Sample signals - replace with real data
        signals_data = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'TSLA'],
            'Signal': ['BUY', 'SELL', 'BUY'],
            'Confidence': [0.85, 0.72, 0.91],
            'Target Price': [155.00, 275.00, 890.00],
            'Stop Loss': [145.00, 285.00, 820.00],
            'Timestamp': ['10:30 AM', '11:15 AM', '02:45 PM']
        })
        
        # Color code signals
        def highlight_signal(val):
            if val == 'BUY':
                return 'background-color: #90EE90'  # Light green
            elif val == 'SELL':
                return 'background-color: #FFB6C1'  # Light red
            return ''
        
        styled_signals = signals_data.style.applymap(
            highlight_signal, subset=['Signal']
        )
        st.dataframe(styled_signals, use_container_width=True)
    
    with col2:
        st.subheader("Signal Strength")
        
        # Signal confidence chart
        fig = px.bar(
            signals_data, 
            x='Symbol', 
            y='Confidence',
            title="Signal Confidence by Symbol",
            color='Confidence',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

def portfolio_tab():
    """Portfolio holdings and allocation"""
    st.header("Portfolio Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Value", "$12,450.75", "2.3%")
    with col2:
        st.metric("Today's P&L", "+$287.50", "+2.37%")
    with col3:
        st.metric("Available Cash", "$2,549.25", "")
    
    # Portfolio allocation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Holdings")
        
        # Sample portfolio data
        portfolio_data = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'Cash'],
            'Shares': [50, 25, 5, 1],
            'Avg Cost': [145.50, 275.25, 2600.00, 2549.25],
            'Current Price': [150.25, 280.50, 2650.75, 2549.25],
            'Market Value': [7512.50, 7012.50, 13253.75, 2549.25],
            'P&L': [237.50, 131.25, 253.75, 0.00],
            'P&L %': [3.26, 1.90, 1.95, 0.00]
        })
        
        st.dataframe(portfolio_data, use_container_width=True)
    
    with col2:
        st.subheader("Allocation")
        
        # Pie chart for allocation
        fig = px.pie(
            portfolio_data, 
            values='Market Value', 
            names='Symbol',
            title="Portfolio Allocation"
        )
        st.plotly_chart(fig, use_container_width=True)

def performance_tab():
    """Performance metrics and charts"""
    st.header("Performance Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", "24.5%", "")
    with col2:
        st.metric("Sharpe Ratio", "1.85", "")
    with col3:
        st.metric("Max Drawdown", "-5.2%", "")
    with col4:
        st.metric("Win Rate", "68%", "")
    
    # Performance chart
    st.subheader("Equity Curve")
    
    # Sample performance data
    dates = pd.date_range(start='2024-01-01', end='2024-12-29', freq='D')
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    equity_curve = (1 + pd.Series(returns, index=dates)).cumprod() * INITIAL_CAPITAL
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity_curve,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def settings_tab():
    """Bot configuration and settings"""
    st.header("Bot Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trading Parameters")
        
        st.number_input("Initial Capital ($)", value=INITIAL_CAPITAL)
        st.slider("Max Position Size (%)", 1, 20, int(MAX_POSITION_SIZE * 100))
        st.slider("Stop Loss (%)", 1, 10, int(STOP_LOSS_PERCENTAGE * 100))
        st.slider("Take Profit (%)", 1, 20, int(TAKE_PROFIT_PERCENTAGE * 100))
        
        st.subheader("Model Settings")
        st.number_input("Feature Window (days)", value=FEATURE_WINDOW_DAYS)
        st.number_input("Prediction Horizon (days)", value=PREDICTION_HORIZON_DAYS)
        st.number_input("Model Retrain Frequency (days)", value=MODEL_RETRAIN_DAYS)
        
    with col2:
        st.subheader("API Configuration")
        
        api_status = {
            "Alpha Vantage": "🟢 Connected" if ALPHA_VANTAGE_API_KEY else "🔴 Not configured",
            "Polygon.io": "🟢 Connected" if POLYGON_API_KEY else "🔴 Not configured",
            "Quandl": "🟢 Connected" if QUANDL_API_KEY else "🔴 Not configured"
        }
        
        for api, status in api_status.items():
            st.write(f"**{api}**: {status}")
        
        st.subheader("System Info")
        st.write(f"**Environment**: {ENVIRONMENT}")
        st.write(f"**Log Level**: {LOG_LEVEL}")
        st.write(f"**Base Directory**: {BASE_DIR}")
        
        if st.button("Test API Connections"):
            st.info("Testing API connections...")
            # Add API testing logic here

if __name__ == "__main__":
    # Import numpy for sample data generation
    import numpy as np
    main()