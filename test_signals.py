#!/usr/bin/env python3
"""Test the new signal generation system"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    print("🎯 AI Day Trading Bot - Signal Generation Test")
    print("=" * 50)

    try:
        # Import components
        from src.data_sources.data_manager import DataManager
        from src.signals import SignalType
        from src.signals.signal_manager import SignalManager

        print("\n📡 Initializing components...")
        dm = DataManager()
        signal_manager = SignalManager()

        print("✅ Components initialized successfully")

        # Test symbols
        test_symbols = ["AAPL", "MSFT"]

        for symbol in test_symbols:
            print(f"\n🔍 Testing signal generation for {symbol}...")

            # Fetch data
            data = dm.get_historical_data(symbol, period="1mo")

            if data is not None and len(data) > 0:
                print(f"✅ Fetched {len(data)} data points")

                # Generate consensus signal
                signal = signal_manager.get_consensus_signal(data, symbol)

                if signal:
                    print(f"🎯 Generated Signal: {signal}")
                    print(f"   Type: {signal.signal_type.value}")
                    print(f"   Strength: {signal.strength.value}")
                    print(f"   Confidence: {signal.confidence:.2f}")
                    print(f"   Price: ${signal.price:.2f}")

                    # Show reasoning summary
                    if "technical_score" in signal.reasoning:
                        print(
                            f"   Technical Score: {signal.reasoning['technical_score']:.3f}"
                        )
                    if "ml_score" in signal.reasoning:
                        print(f"   ML Score: {signal.reasoning['ml_score']:.3f}")
                    if "combined_score" in signal.reasoning:
                        print(
                            f"   Combined Score: {signal.reasoning['combined_score']:.3f}"
                        )

                else:
                    print("   No significant signal generated")

                # Get all individual signals
                all_signals = signal_manager.generate_signals(data, symbol)
                print(
                    f"   Individual generators: {len([s for s in all_signals.values() if s is not None])}/{len(all_signals)} active"
                )

            else:
                print(f"❌ Failed to fetch data for {symbol}")

        # Show performance metrics
        print(f"\n📊 Signal Manager Performance:")
        for symbol in test_symbols:
            metrics = signal_manager.get_performance_metrics(symbol)
            if metrics:
                print(
                    f"   {symbol}: {metrics.get('total_signals', 0)} signals generated"
                )

        print("\n🎉 Signal generation test completed successfully!")

    except Exception as e:
        print(f"❌ Error during signal generation test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
