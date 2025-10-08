#!/usr/bin/env python3
"""Debug date issue"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    from src.data_sources.data_manager import DataManager
    
    dm = DataManager()
    data = dm.get_historical_data("AAPL", period="1mo")
    
    print(f"Data type: {type(data)}")
    print(f"Index type: {type(data.index)}")
    print(f"Index sample: {data.index[:3]}")
    print(f"Index sample types: {[type(x) for x in data.index[:3]]}")

if __name__ == "__main__":
    main()