import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class DataHandler:
    """
    Data handling and preprocessing utilities for SPT analysis.
    """
    
    @staticmethod
    def load_from_csv(price_file: str,
                     market_cap_file: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load and align price and market cap data from CSV files.
        
        Args:
            price_file: Path to price data CSV
            market_cap_file: Path to market cap data CSV
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Tuple of (price_data, market_cap_data) DataFrames
        """
        prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
        if market_cap_file:
            market_caps = pd.read_csv(market_cap_file, index_col=0, parse_dates=True)
        else:
            market_caps = None
            
        # Align dates
        if start_date:
            prices = prices.loc[start_date:]
            if market_caps is not None:
                market_caps = market_caps.loc[start_date:]
                
        if end_date:
            prices = prices.loc[:end_date]
            if market_caps is not None:
                market_caps = market_caps.loc[:end_date]
                
        return prices, market_caps
    
    @staticmethod
    def resample_data(prices: pd.DataFrame,
                     market_caps: Optional[pd.DataFrame] = None,
                     freq: str = 'B') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Resample data to specified frequency.
        
        Args:
            prices: Price data
            market_caps: Market cap data
            freq: Resampling frequency
            
        Returns:
            Tuple of resampled DataFrames
        """
        prices_resampled = prices.resample(freq).last().ffill()
        if market_caps is not None:
            market_caps_resampled = market_caps.resample(freq).last().ffill()
        else:
            market_caps_resampled = None
            
        return prices_resampled, market_caps_resampled
    
    @staticmethod
    def clean_financial_data(prices: pd.DataFrame,
                            market_caps: Optional[pd.DataFrame] = None,
                            min_price: float = 1.0,
                            min_mcap: float = 1e6) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Clean financial data by removing illiquid stocks and missing values.
        
        Args:
            prices: Price data
            market_caps: Market cap data
            min_price: Minimum price threshold
            min_mcap: Minimum market cap threshold
            
        Returns:
            Tuple of cleaned DataFrames
        """
        # Filter by price
        valid_prices = prices.columns[(prices > min_price).all()]
        prices_clean = prices[valid_prices]
        
        if market_caps is not None:
            # Align columns with prices
            market_caps_clean = market_caps[valid_prices]
            
            # Filter by market cap
            valid_mcaps = market_caps_clean.columns[(market_caps_clean > min_mcap).all()]
            prices_clean = prices_clean[valid_mcaps]
            market_caps_clean = market_caps_clean[valid_mcaps]
        else:
            market_caps_clean = None
            
        # Forward fill missing values
        prices_clean = prices_clean.ffill().bfill()
        if market_caps_clean is not None:
            market_caps_clean = market_caps_clean.ffill().bfill()
            
        return prices_clean, market_caps_clean
