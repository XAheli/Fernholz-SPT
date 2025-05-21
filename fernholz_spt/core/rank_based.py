import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union

from fernholz_spt.core.market_model import MarketModel

class RankBasedPortfolio:
    """
    Implementation of rank-based portfolios according to Fernholz's theory.
    
    Rank-based portfolios allocate capital based on the rank of stocks rather
    than their specific identity. This approach can capture structural properties
    of the market like the size effect.
    """
    
    def __init__(self, market_model: MarketModel):
        """
        Initialize with a market model.
        
        Args:
            market_model: Initialized MarketModel instance
        """
        self.market_model = market_model
        
    def top_m_portfolio(self, m: int, weighting: str = 'equal') -> pd.DataFrame:
        """
        Create a portfolio of the top m stocks by market capitalization.
        
        Args:
            m: Number of top stocks to include
            weighting: Weighting scheme ('equal', 'cap', or 'inverse_cap')
            
        Returns:
            DataFrame of portfolio weights
        """
        if m > self.market_model.n_stocks:
            raise ValueError(f"m must be <= number of stocks ({self.market_model.n_stocks})")
            
        weights = pd.DataFrame(0, index=self.market_model.market_weights.index, 
                              columns=self.market_model.stock_names)
        
        for date in weights.index:
            # Get top m stocks by market weight
            top_stocks = self.market_model.market_weights.loc[date].nlargest(m).index
            
            if weighting == 'equal':
                # Equal weighting
                weights.loc[date, top_stocks] = 1.0 / m
                
            elif weighting == 'cap':
                # Capitalization weighting
                top_weights = self.market_model.market_weights.loc[date, top_stocks]
                weights.loc[date, top_stocks] = top_weights / top_weights.sum()
                
            elif weighting == 'inverse_cap':
                # Inverse capitalization weighting
                top_weights = self.market_model.market_weights.loc[date, top_stocks]
                inv_weights = 1.0 / top_weights
                weights.loc[date, top_stocks] = inv_weights / inv_weights.sum()
                
            else:
                raise ValueError(f"Unknown weighting scheme: {weighting}")
            
        return weights
    
    def bottom_m_portfolio(self, m: int, weighting: str = 'equal') -> pd.DataFrame:
        """
        Create a portfolio of the bottom m stocks by market capitalization.
        
        Args:
            m: Number of bottom stocks to include
            weighting: Weighting scheme ('equal', 'cap', or 'inverse_cap')
            
        Returns:
            DataFrame of portfolio weights
        """
        if m > self.market_model.n_stocks:
            raise ValueError(f"m must be <= number of stocks ({self.market_model.n_stocks})")
            
        weights = pd.DataFrame(0, index=self.market_model.market_weights.index, 
                              columns=self.market_model.stock_names)
        
        for date in weights.index:
            # Get bottom m stocks by market weight
            bottom_stocks = self.market_model.market_weights.loc[date].nsmallest(m).index
            
            if weighting == 'equal':
                # Equal weighting
                weights.loc[date, bottom_stocks] = 1.0 / m
                
            elif weighting == 'cap':
                # Capitalization weighting
                bottom_weights = self.market_model.market_weights.loc[date, bottom_stocks]
                weights.loc[date, bottom_stocks] = bottom_weights / bottom_weights.sum()
                
            elif weighting == 'inverse_cap':
                # Inverse capitalization weighting
                bottom_weights = self.market_model.market_weights.loc[date, bottom_stocks]
                inv_weights = 1.0 / bottom_weights
                weights.loc[date, bottom_stocks] = inv_weights / inv_weights.sum()
                
            else:
                raise ValueError(f"Unknown weighting scheme: {weighting}")
            
        return weights
    
    def middle_m_portfolio(self, m: int, weighting: str = 'equal') -> pd.DataFrame:
        """
        Create a portfolio of m stocks in the middle of the market cap distribution.
        
        Args:
            m: Number of middle stocks to include
            weighting: Weighting scheme ('equal', 'cap', or 'inverse_cap')
            
        Returns:
            DataFrame of portfolio weights
        """
        if m > self.market_model.n_stocks:
            raise ValueError(f"m must be <= number of stocks ({self.market_model.n_stocks})")
            
        weights = pd.DataFrame(0, index=self.market_model.market_weights.index, 
                              columns=self.market_model.stock_names)
        
        for date in weights.index:
            # Sort stocks by market weight
            sorted_stocks = self.market_model.market_weights.loc[date].sort_values(ascending=False)
            
            # Determine middle range
            start_idx = (len(sorted_stocks) - m) // 2
            middle_stocks = sorted_stocks.index[start_idx:start_idx+m]
            
            if weighting == 'equal':
                # Equal weighting
                weights.loc[date, middle_stocks] = 1.0 / m
                
            elif weighting == 'cap':
                # Capitalization weighting
                middle_weights = self.market_model.market_weights.loc[date, middle_stocks]
                weights.loc[date, middle_stocks] = middle_weights / middle_weights.sum()
                
            elif weighting == 'inverse_cap':
                # Inverse capitalization weighting
                middle_weights = self.market_model.market_weights.loc[date, middle_stocks]
                inv_weights = 1.0 / middle_weights
                weights.loc[date, middle_stocks] = inv_weights / inv_weights.sum()
                
            else:
                raise ValueError(f"Unknown weighting scheme: {weighting}")
            
        return weights
    
    def decile_portfolio(self, decile: int, weighting: str = 'equal') -> pd.DataFrame:
        """
        Create a portfolio of stocks in a specific market cap decile.
        
        Args:
            decile: Decile to target (1 = largest, 10 = smallest)
            weighting: Weighting scheme ('equal', 'cap', or 'inverse_cap')
            
        Returns:
            DataFrame of portfolio weights
        """
        if decile < 1 or decile > 10:
            raise ValueError("Decile must be between 1 and 10")
            
        weights = pd.DataFrame(0, index=self.market_model.market_weights.index, 
                              columns=self.market_model.stock_names)
        
        for date in weights.index:
            # Sort stocks by market weight
            sorted_stocks = self.market_model.market_weights.loc[date].sort_values(ascending=False)
            
            # Determine decile range
            n_stocks = len(sorted_stocks)
            stocks_per_decile = n_stocks // 10
            
            start_idx = (decile - 1) * stocks_per_decile
            end_idx = min(n_stocks, start_idx + stocks_per_decile)
            
            decile_stocks = sorted_stocks.index[start_idx:end_idx]
            
            if weighting == 'equal':
                # Equal weighting
                weights.loc[date, decile_stocks] = 1.0 / len(decile_stocks)
                
            elif weighting == 'cap':
                # Capitalization weighting
                decile_weights = self.market_model.market_weights.loc[date, decile_stocks]
                weights.loc[date, decile_stocks] = decile_weights / decile_weights.sum()
                
            elif weighting == 'inverse_cap':
                # Inverse capitalization weighting
                decile_weights = self.market_model.market_weights.loc[date, decile_stocks]
                inv_weights = 1.0 / decile_weights
                weights.loc[date, decile_stocks] = inv_weights / inv_weights.sum()
                
            else:
                raise ValueError(f"Unknown weighting scheme: {weighting}")
            
        return weights
    
    def generic_rank_portfolio(self, 
                              rank_weight_function: Callable[[int, int], float],
                              normalize: bool = True) -> pd.DataFrame:
        """
        Create a portfolio with weights determined by a rank-based function.
        
        This allows for arbitrary weighting schemes based on rank.
        
        Args:
            rank_weight_function: Function that maps (rank, total_stocks) to weight
            normalize: Whether to normalize weights to sum to 1
            
        Returns:
            DataFrame of portfolio weights
        """
        weights = pd.DataFrame(0, index=self.market_model.market_weights.index, 
                              columns=self.market_model.stock_names)
        
        for date in weights.index:
            # Get stock ranks
            ranks = {}
            sorted_stocks = self.market_model.market_weights.loc[date].sort_values(ascending=False)
            
            for i, stock in enumerate(sorted_stocks.index):
                ranks[stock] = i + 1
                
            # Calculate weights based on rank
            for stock in self.market_model.stock_names:
                if stock in ranks:
                    weights.loc[date, stock] = rank_weight_function(ranks[stock], self.market_model.n_stocks)
                    
            # Normalize if requested
            if normalize and weights.loc[date].sum() > 0:
                weights.loc[date] = weights.loc[date] / weights.loc[date].sum()
                
        return weights
    
    def leaking_portfolio(self, alpha: float = 0.5) -> pd.DataFrame:
        """
        Create a leaking portfolio as described by Fernholz.
        
        The leaking portfolio overweights small stocks and underweights large stocks
        with a continuous function based on rank.
        
        Args:
            alpha: Parameter controlling the strength of the size effect (0 < alpha < 1)
            
        Returns:
            DataFrame of portfolio weights
        """
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
            
        def rank_function(rank, n_stocks):
            # Normalize rank to [0, 1]
            x = rank / n_stocks
            # Apply leaking function: f(x) = x^(-alpha)
            return x**(-alpha)
        
        return self.generic_rank_portfolio(rank_function)
    
    def diversity_weighted_rank_portfolio(self, p: float = 0.5) -> pd.DataFrame:
        """
        Create a diversity-weighted portfolio based on ranks.
        
        This applies the diversity weighting principle to ranks rather than
        directly to market weights.
        
        Args:
            p: Diversity parameter (0 < p < 1)
            
        Returns:
            DataFrame of portfolio weights
        """
        if p <= 0 or p >= 1:
            raise ValueError(f"p must be in (0, 1), got {p}")
            
        def rank_function(rank, n_stocks):
            # Get corresponding rank weight from ranked_weights
            return rank**(-p)
        
        return self.generic_rank_portfolio(rank_function)
    
    def calculate_turnover(self, portfolio_weights: pd.DataFrame) -> pd.Series:
        """
        Calculate the turnover of a portfolio over time.
        
        Turnover is an important measure of portfolio stability and trading costs.
        
        Args:
            portfolio_weights: DataFrame of portfolio weights
            
        Returns:
            Series of turnover values (half the sum of absolute weight changes)
        """
        turnover = pd.Series(index=portfolio_weights.index[1:])
        
        for i in range(1, len(portfolio_weights)):
            date = portfolio_weights.index[i]
            prev_date = portfolio_weights.index[i-1]
            
            # Calculate absolute changes in weights
            weight_changes = np.abs(portfolio_weights.loc[date] - portfolio_weights.loc[prev_date])
            
            # Turnover is half the sum of absolute changes
            turnover[date] = 0.5 * weight_changes.sum()
            
        return turnover
