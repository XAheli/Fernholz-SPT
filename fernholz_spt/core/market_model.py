import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats

class MarketModel:
    """
    Core market model based on Fernholz's Stochastic Portfolio Theory.
    
    This class implements the fundamental market dynamics described in Fernholz's
    "Stochastic Portfolio Theory" (2002), including market weight calculations,
    rank-based transformations, and diversity measures.
    """
    
    def __init__(self, 
                 stock_prices: pd.DataFrame, 
                 market_caps: Optional[pd.DataFrame] = None,
                 estimate_covariance: bool = True,
                 cov_window: int = 252,
                 cov_method: str = 'standard'):
        """
        Initialize the market model with historical stock data.
        
        Args:
            stock_prices: DataFrame with dates as index and stock tickers as columns
            market_caps: Optional DataFrame of market capitalizations. If None, prices will be used as proxy
            estimate_covariance: Whether to estimate the covariance matrix
            cov_window: Rolling window size for covariance estimation
            cov_method: Covariance estimation method ('standard', 'shrinkage', or 'exponential')
        """
        self.stock_prices = stock_prices
        self.n_stocks = stock_prices.shape[1]
        self.stock_names = stock_prices.columns
        self.dates = stock_prices.index
        
        # Calculate log returns
        self.log_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
        
        # Use market caps if provided, otherwise use prices as proxy
        if market_caps is not None:
            self.market_caps = market_caps
        else:
            self.market_caps = stock_prices.copy()
            
        # Calculate market weights and ranked weights
        self.market_weights = self._calculate_market_weights()
        self.ranked_weights = self._calculate_ranked_weights()
        self.rank_crossovers = self._calculate_rank_crossovers()
        
        # Calculate covariance matrix if requested
        if estimate_covariance:
            self.cov_matrices = self._estimate_covariance(window=cov_window, method=cov_method)
        else:
            self.cov_matrices = None
            
        # Calculate capitalization curves and diversity measures
        self.cap_curves = self._calculate_capitalization_curves()
        self.diversity = self._calculate_diversity()
        self.entropy = self._calculate_entropy()
            
    def _calculate_market_weights(self) -> pd.DataFrame:
        """
        Calculate market weights (capitalization proportions) for each stock.
        
        Market weights μ_i(t) represent the proportion of total market capitalization 
        for each stock i at time t. Mathematically:
        
        μ_i(t) = S_i(t) / Σ_j S_j(t)
        
        where S_i(t) is the market capitalization of stock i at time t.
        
        Returns:
            DataFrame of market weights with same structure as stock_prices
        """
        total_market_cap = self.market_caps.sum(axis=1)
        return self.market_caps.div(total_market_cap, axis=0)
    
    def _calculate_ranked_weights(self) -> pd.DataFrame:
        """
        Calculate market weights sorted by rank (largest to smallest).
        
        Ranked weights are a key concept in Fernholz's theory and form the basis
        for rank-based portfolio strategies. For each time t, stocks are sorted
        by market weight and assigned ranks from 1 to n.
        
        Returns:
            DataFrame with dates as index and ranks as columns
        """
        ranked_weights = pd.DataFrame(index=self.market_weights.index)
        rank_map = {}  # Maps each date to a dictionary of {stock: rank}
        
        for date in self.market_weights.index:
            # Sort weights in descending order
            sorted_stocks = self.market_weights.loc[date].sort_values(ascending=False)
            
            # Create rank mappings for this date
            rank_map[date] = {stock: i+1 for i, stock in enumerate(sorted_stocks.index)}
            
            # Create rank columns (rank_1 is the largest stock, etc.)
            for i, stock in enumerate(sorted_stocks.index):
                col_name = f'rank_{i+1}'
                ranked_weights.loc[date, col_name] = sorted_stocks.iloc[i]
                
        # Store the rank mappings
        self.rank_map = rank_map
            
        return ranked_weights
    
    def _calculate_rank_crossovers(self) -> pd.DataFrame:
        """
        Calculate the frequency of rank changes between consecutive time periods.
        
        Rank crossovers are important for understanding market stability and
        can affect the turnover of rank-based portfolios.
        
        Returns:
            DataFrame counting rank changes between consecutive periods
        """
        if not hasattr(self, 'rank_map'):
            raise ValueError("Rank map not calculated. Run _calculate_ranked_weights first.")
            
        dates = list(self.rank_map.keys())
        crossovers = pd.DataFrame(0, index=dates[1:], columns=['total_changes'])
        
        for i in range(1, len(dates)):
            curr_date = dates[i]
            prev_date = dates[i-1]
            
            # Count stocks that changed rank
            changed_stocks = 0
            for stock in self.stock_names:
                if stock in self.rank_map[curr_date] and stock in self.rank_map[prev_date]:
                    if self.rank_map[curr_date][stock] != self.rank_map[prev_date][stock]:
                        changed_stocks += 1
                        
            crossovers.loc[curr_date, 'total_changes'] = changed_stocks
            crossovers.loc[curr_date, 'change_rate'] = changed_stocks / self.n_stocks
            
        return crossovers
    
    def _estimate_covariance(self, window: int = 252, method: str = 'standard') -> Dict[pd.Timestamp, np.ndarray]:
        """
        Estimate the covariance matrix of log returns.
        
        Args:
            window: Rolling window size for estimation
            method: Estimation method ('standard', 'shrinkage', or 'exponential')
            
        Returns:
            Dictionary mapping dates to covariance matrices
        """
        cov_matrices = {}
        
        if method == 'standard':
            # Use rolling window standard covariance estimation
            for i in range(window, len(self.log_returns)):
                date = self.log_returns.index[i]
                window_returns = self.log_returns.iloc[i-window:i]
                cov_matrices[date] = window_returns.cov().values
                
        elif method == 'shrinkage':
            # Use Ledoit-Wolf shrinkage estimator
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            
            for i in range(window, len(self.log_returns)):
                date = self.log_returns.index[i]
                window_returns = self.log_returns.iloc[i-window:i].values
                lw.fit(window_returns)
                cov_matrices[date] = lw.covariance_
                
        elif method == 'exponential':
            # Use exponentially weighted covariance estimation
            decay = 0.94  # Standard EWMA decay factor
            
            for i in range(window, len(self.log_returns)):
                date = self.log_returns.index[i]
                window_returns = self.log_returns.iloc[i-window:i]
                
                # Calculate exponentially weighted covariance
                weights = np.array([(1-decay) * decay**j for j in range(window)])
                weights = weights[::-1] / weights.sum()  # Normalize
                
                weighted_cov = np.zeros((self.n_stocks, self.n_stocks))
                mean_returns = window_returns.mean().values
                
                for j in range(window):
                    dev = window_returns.iloc[j].values - mean_returns
                    weighted_cov += weights[j] * np.outer(dev, dev)
                    
                cov_matrices[date] = weighted_cov
        else:
            raise ValueError(f"Unknown covariance method: {method}")
            
        return cov_matrices
    
    def _calculate_capitalization_curves(self) -> pd.DataFrame:
        """
        Calculate the capitalization curves for each rank.
        
        Capitalization curves show how the market weight of each rank evolves over time.
        These are important for studying market structure and concentration.
        
        Returns:
            DataFrame of capitalization curves for each rank
        """
        cap_curves = pd.DataFrame(index=self.ranked_weights.index)
        
        for rank in range(1, self.n_stocks + 1):
            rank_col = f'rank_{rank}'
            if rank_col in self.ranked_weights.columns:
                cap_curves[f'cap_{rank}'] = self.ranked_weights[rank_col]
                
        return cap_curves
    
    def _calculate_diversity(self, p_values: List[float] = [0.5, 0.75, 0.9]) -> pd.DataFrame:
        """
        Calculate market diversity using the p-diversity measure.
        
        The p-diversity measure D_p(μ) is defined as:
        
        D_p(μ) = (Σ_i μ_i^p)^(1/p)
        
        where μ_i are the market weights and 0 < p < 1.
        
        Args:
            p_values: List of p-values for diversity calculation
            
        Returns:
            DataFrame of diversity measures for each p-value
        """
        diversity = pd.DataFrame(index=self.market_weights.index)
        
        for p in p_values:
            if p <= 0 or p >= 1:
                raise ValueError(f"p must be in (0, 1), got {p}")
                
            col_name = f'diversity_{p}'
            diversity_values = []
            
            for date in self.market_weights.index:
                weights = self.market_weights.loc[date].values
                diversity_values.append(np.sum(weights**p)**(1/p))
                
            diversity[col_name] = diversity_values
            
        return diversity
    
    def _calculate_entropy(self) -> pd.Series:
        """
        Calculate the entropy of market weights.
        
        The entropy H(μ) is defined as:
        
        H(μ) = -Σ_i μ_i * log(μ_i)
        
        Returns:
            Series of entropy values over time
        """
        entropy = pd.Series(index=self.market_weights.index)
        
        for date in self.market_weights.index:
            weights = self.market_weights.loc[date].values
            # Avoid log(0) by filtering out zeros
            valid_weights = weights[weights > 0]
            entropy[date] = -np.sum(valid_weights * np.log(valid_weights))
            
        return entropy
    
    def get_rank_at_date(self, date: pd.Timestamp) -> Dict[str, int]:
        """
        Get the rank of each stock at a specific date.
        
        Args:
            date: The date to get ranks for
            
        Returns:
            Dictionary mapping stock names to ranks
        """
        if date not in self.rank_map:
            raise ValueError(f"No rank information for date {date}")
            
        return self.rank_map[date]
    
    def get_stocks_at_rank(self, date: pd.Timestamp, ranks: Union[int, List[int]]) -> List[str]:
        """
        Get the stocks at specific ranks on a given date.
        
        Args:
            date: The date to look up
            ranks: Rank or list of ranks to look up
            
        Returns:
            List of stock names at the specified ranks
        """
        if date not in self.rank_map:
            raise ValueError(f"No rank information for date {date}")
            
        # Convert single rank to list
        if isinstance(ranks, int):
            ranks = [ranks]
            
        # Invert the rank map for this date
        inv_rank_map = {v: k for k, v in self.rank_map[date].items()}
        
        # Get stocks at specified ranks
        result = []
        for rank in ranks:
            if rank in inv_rank_map:
                result.append(inv_rank_map[rank])
                
        return result
    
    def calculate_concentration_ratio(self, date: pd.Timestamp, top_n: int) -> float:
        """
        Calculate the concentration ratio of the top n stocks at a specific date.
        
        The concentration ratio CR_n is defined as the sum of the weights of the 
        top n stocks:
        
        CR_n = Σ_{i=1}^n μ_(i)
        
        where μ_(i) is the weight of the stock with rank i.
        
        Args:
            date: The date to calculate for
            top_n: Number of top stocks to include
            
        Returns:
            Concentration ratio value
        """
        if top_n > self.n_stocks:
            raise ValueError(f"top_n must be <= number of stocks ({self.n_stocks})")
            
        # Get weights of top n stocks
        total_weight = 0
        for i in range(1, top_n + 1):
            col_name = f'rank_{i}'
            if col_name in self.ranked_weights.columns:
                total_weight += self.ranked_weights.loc[date, col_name]
                
        return total_weight
    
    def calculate_herfindahl_index(self, date: pd.Timestamp) -> float:
        """
        Calculate the Herfindahl-Hirschman Index (HHI) at a specific date.
        
        The HHI is defined as the sum of squared market weights:
        
        HHI = Σ_i μ_i^2
        
        Args:
            date: The date to calculate for
            
        Returns:
            HHI value
        """
        weights = self.market_weights.loc[date].values
        return np.sum(weights**2)
    
    def calculate_market_portfolio(self) -> pd.DataFrame:
        """
        Return the market portfolio (i.e., market weights).
        
        The market portfolio is simply the vector of market weights.
        
        Returns:
            DataFrame of market portfolio weights
        """
        return self.market_weights.copy()
