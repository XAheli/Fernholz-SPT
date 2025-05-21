import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class MarketModel:
    """
    Market model class for Stochastic Portfolio Theory.
    
    This class handles market data, weight calculations, and covariance matrices.
    """
    
    def __init__(self, market_data: pd.DataFrame):
        """
        Initialize with market data.
        
        Args:
            market_data: DataFrame with stock prices (columns=stocks, index=dates)
        """
        self.market_data = market_data
        self.n_stocks = len(market_data.columns)
        
        # Calculate log returns
        self.log_returns = np.log(market_data / market_data.shift(1)).dropna()
        
        # Calculate market weights
        self.market_weights = self._calculate_market_weights()
        
        # Calculate covariance matrices
        self.cov_matrices = self._calculate_covariance_matrices()
    
    def _calculate_market_weights(self, window: int = 63) -> pd.DataFrame:
        """
        Calculate market capitalization weights.
        
        For simplicity, we're using price as a proxy for market cap.
        
        Args:
            window: Rolling window for calculating weights
            
        Returns:
            DataFrame of market weights
        """
        weights = pd.DataFrame(index=self.market_data.index, 
                             columns=self.market_data.columns)
        
        for date in self.market_data.index:
            total_cap = self.market_data.loc[date].sum()
            weights.loc[date] = self.market_data.loc[date] / total_cap
            
        return weights
    
    def _calculate_covariance_matrices(self, window: int = 63) -> Dict[pd.Timestamp, np.ndarray]:
        """
        Calculate rolling covariance matrices.
        
        Args:
            window: Rolling window for covariance calculation
            
        Returns:
            Dictionary of date -> covariance matrix
        """
        cov_matrices = {}
        
        for i in range(window, len(self.log_returns)):
            date = self.log_returns.index[i]
            start_idx = i - window
            end_idx = i
            
            # Get data window
            window_data = self.log_returns.iloc[start_idx:end_idx]
            
            # Calculate covariance matrix (annualized)
            cov_matrix = window_data.cov() * 252
            cov_matrices[date] = cov_matrix.values
            
        return cov_matrices


class RankBasedPortfolio:
    """
    Portfolio construction based on rank-based strategies.
    """
    
    def __init__(self, market_model: MarketModel):
        """
        Initialize with market model.
        
        Args:
            market_model: Initialized MarketModel instance
        """
        self.market_model = market_model
        
    def top_m_portfolio(self, m: int) -> pd.DataFrame:
        """
        Create a portfolio of the top m stocks by market cap.
        
        Args:
            m: Number of stocks to include
            
        Returns:
            DataFrame of portfolio weights
        """
        weights = pd.DataFrame(0, index=self.market_model.market_weights.index,
                              columns=self.market_model.market_weights.columns)
        
        for date in weights.index:
            sorted_weights = self.market_model.market_weights.loc[date].sort_values(ascending=False)
            top_stocks = sorted_weights.index[:m]
            
            # Equal weight the top m stocks
            weights.loc[date, top_stocks] = 1.0 / m
            
        return weights
    
    def bottom_m_portfolio(self, m: int) -> pd.DataFrame:
        """
        Create a portfolio of the bottom m stocks by market cap.
        
        Args:
            m: Number of stocks to include
            
        Returns:
            DataFrame of portfolio weights
        """
        weights = pd.DataFrame(0, index=self.market_model.market_weights.index,
                              columns=self.market_model.market_weights.columns)
        
        for date in weights.index:
            sorted_weights = self.market_model.market_weights.loc[date].sort_values(ascending=True)
            bottom_stocks = sorted_weights.index[:m]
            
            # Equal weight the bottom m stocks
            weights.loc[date, bottom_stocks] = 1.0 / m
            
        return weights


class PortfolioAnalyzer:
    """
    Performance analysis tools for portfolios constructed using SPT.
    
    This class implements various performance metrics and analysis methods
    specific to Stochastic Portfolio Theory.
    """
    
    def __init__(self, market_model: Optional[MarketModel] = None):
        """
        Initialize with optional market model.
        
        Args:
            market_model: Initialized MarketModel instance
        """
        self.market_model = market_model
        
    def calculate_relative_return(self,
                                portfolio_weights: pd.DataFrame,
                                benchmark_weights: pd.DataFrame) -> pd.Series:
        """
        Calculate relative return of portfolio versus benchmark.
        
        Args:
            portfolio_weights: DataFrame of portfolio weights
            benchmark_weights: DataFrame of benchmark weights (e.g., market portfolio)
            
        Returns:
            Series of relative returns
        """
        relative_returns = pd.Series(index=portfolio_weights.index[1:])
        
        for i in range(1, len(portfolio_weights)):
            date = portfolio_weights.index[i]
            prev_date = portfolio_weights.index[i-1]
            
            # Calculate log-relative return
            port_return = np.log(np.sum(portfolio_weights.loc[prev_date] * 
                                      self.market_model.market_weights.loc[date]))
            bench_return = np.log(np.sum(benchmark_weights.loc[prev_date] * 
                                       self.market_model.market_weights.loc[date]))
            
            relative_returns[date] = port_return - bench_return
            
        return relative_returns
    
    def calculate_excess_growth(self,
                              portfolio_weights: pd.DataFrame) -> pd.Series:
        """
        Calculate the excess growth rate (drift process) of a portfolio.
        
        Args:
            portfolio_weights: DataFrame of portfolio weights
            
        Returns:
            Series of excess growth rates
        """
        if self.market_model is None or self.market_model.cov_matrices is None:
            raise ValueError("Market model with covariance matrices required")
            
        excess_growth = pd.Series(index=portfolio_weights.index)
        
        for date in portfolio_weights.index:
            if date not in self.market_model.cov_matrices:
                continue
                
            weights = portfolio_weights.loc[date].values
            mu = self.market_model.market_weights.loc[date].values
            cov_matrix = self.market_model.cov_matrices[date]
            
            # Calculate relative weights
            relative_weights = np.zeros_like(weights)
            nonzero_idx = mu > 0
            relative_weights[nonzero_idx] = weights[nonzero_idx] / mu[nonzero_idx]
            
            # Fernholz's excess growth formula
            excess_growth[date] = 0.5 * np.sum(
                np.diag(cov_matrix) * (relative_weights - 1)**2
            )
            
        return excess_growth
    
    def analyze_turnover(self,
                       portfolio_weights: pd.DataFrame,
                       window: int = 21) -> pd.DataFrame:
        """
        Analyze portfolio turnover statistics.
        
        Args:
            portfolio_weights: DataFrame of portfolio weights
            window: Rolling window size for statistics
            
        Returns:
            DataFrame with turnover statistics
        """
        turnover = pd.Series(index=portfolio_weights.index[1:])
        
        for i in range(1, len(portfolio_weights)):
            date = portfolio_weights.index[i]
            prev_date = portfolio_weights.index[i-1]
            
            # Calculate absolute weight changes
            weight_changes = np.abs(
                portfolio_weights.loc[date] - portfolio_weights.loc[prev_date]
            )
            turnover[date] = 0.5 * weight_changes.sum()
            
        stats = pd.DataFrame(index=turnover.index)
        stats['Turnover'] = turnover
        stats['Rolling Mean'] = turnover.rolling(window).mean()
        stats['Rolling Std'] = turnover.rolling(window).std()
        stats['Expanding Max'] = turnover.expanding().max()
        
        return stats
    
    def calculate_diversification_ratio(self,
                                      portfolio_weights: pd.DataFrame,
                                      window: int = 63) -> pd.Series:
        """
        Calculate the diversification ratio of the portfolio.
        
        Diversification Ratio = Weighted Average Volatility / Portfolio Volatility
        
        Args:
            portfolio_weights: DataFrame of portfolio weights
            window: Rolling window for volatility calculation
            
        Returns:
            Series of diversification ratios
        """
        if self.market_model is None:
            raise ValueError("Market model required")
            
        diversification = pd.Series(index=portfolio_weights.index[window:])
        
        for i in range(window, len(portfolio_weights)):
            date = portfolio_weights.index[i]
            start_date = portfolio_weights.index[i-window]
            
            # Get weights and returns for the window
            weights = portfolio_weights.loc[start_date:date]
            returns = self.market_model.log_returns.loc[start_date:date]
            
            # Calculate individual volatilities
            volatilities = returns.std() * np.sqrt(252)
            weighted_vol = np.dot(weights.mean(), volatilities)
            
            # Calculate portfolio volatility
            port_returns = (weights.shift(1) * returns).sum(axis=1)
            port_vol = port_returns.std() * np.sqrt(252)
            
            diversification[date] = weighted_vol / port_vol if port_vol > 0 else 0
            
        return diversification

    def calculate_fernholz_metrics(self,
                                  portfolio_weights: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate key Fernholz metrics for portfolio analysis:
        1. Excess growth rate
        2. Portfolio entropy
        3. Relative return vs market
        4. Turnover statistics
        
        Args:
            portfolio_weights: DataFrame of portfolio weights
            
        Returns:
            DataFrame of calculated metrics
        """
        metrics = pd.DataFrame(index=portfolio_weights.index)
        
        # Excess growth
        metrics['Excess Growth'] = self.calculate_excess_growth(portfolio_weights)
        
        # Portfolio entropy
        entropy = []
        for date in portfolio_weights.index:
            weights = portfolio_weights.loc[date]
            valid_weights = weights[weights > 0]
            entropy.append(-np.sum(valid_weights * np.log(valid_weights)))
        metrics['Entropy'] = entropy
        
        # Relative return vs market
        market_weights = self.market_model.market_weights
        metrics['Relative Return'] = self.calculate_relative_return(
            portfolio_weights, market_weights
        ).cumsum()
        
        # Turnover
        turnover = self.analyze_turnover(portfolio_weights)['Turnover']
        metrics['Turnover'] = turnover.rolling(21).mean()
        
        return metrics.dropna()

    def _calculate_portfolio_returns(self, portfolio_weights: pd.DataFrame) -> pd.Series:
        """
        Calculate log returns of a portfolio.
        
        Args:
            portfolio_weights: DataFrame of portfolio weights
            
        Returns:
            Series of log returns
        """
        returns = pd.Series(index=portfolio_weights.index[1:])
        
        for i in range(1, len(portfolio_weights)):
            date = portfolio_weights.index[i]
            prev_date = portfolio_weights.index[i-1]
            
            # Get previous weights and current returns
            weights = portfolio_weights.loc[prev_date]
            
            if prev_date in self.market_model.log_returns.index and date in self.market_model.log_returns.index:
                asset_returns = self.market_model.log_returns.loc[date]
                # Calculate weighted return
                returns[date] = (weights * asset_returns).sum()
            
        return returns

    def calculate_size_effect(self,
                            top_frac: float = 0.2,
                            bottom_frac: float = 0.2) -> pd.DataFrame:
        """
        Calculate the size effect by comparing top and bottom quantiles.
        
        Args:
            top_frac: Fraction of stocks considered large
            bottom_frac: Fraction of stocks considered small
            
        Returns:
            DataFrame with size effect metrics
        """
        n_top = int(self.market_model.n_stocks * top_frac)
        n_bottom = int(self.market_model.n_stocks * bottom_frac)
        
        # Create portfolios
        rank_portfolio = RankBasedPortfolio(self.market_model)
        top_port = rank_portfolio.top_m_portfolio(n_top)
        bottom_port = rank_portfolio.bottom_m_portfolio(n_bottom)
        
        # Calculate returns
        top_returns = self._calculate_portfolio_returns(top_port)
        bottom_returns = self._calculate_portfolio_returns(bottom_port)
        
        # Calculate metrics
        results = pd.DataFrame({
            'Top Returns': top_returns,
            'Bottom Returns': bottom_returns,
            'Size Effect': bottom_returns - top_returns
        })
        
        return results.dropna()