import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from scipy.optimize import minimize

from fernholz_spt.core.market_model import MarketModel
from fernholz_spt.core.portfolio_generation import FunctionallyGeneratedPortfolio

class DiversityOptimization:
    """
    Optimization methods for diversity-weighted portfolios.
    
    Diversity-weighted portfolios are parameterized by a diversity parameter p,
    which controls the tradeoff between market weighting and equal weighting.
    """
    
    def __init__(self, market_model: MarketModel):
        """
        Initialize with a market model.
        
        Args:
            market_model: Initialized MarketModel instance
        """
        self.market_model = market_model
        self.portfolio_generator = FunctionallyGeneratedPortfolio(market_model)
        
    def optimize_diversity_parameter(self, 
                                    start_date: pd.Timestamp, 
                                    end_date: pd.Timestamp,
                                    objective: str = 'growth',
                                    target_volatility: Optional[float] = None) -> float:
        """
        Find optimal diversity parameter p for a given objective.
        
        Args:
            start_date: Start date for optimization period
            end_date: End date for optimization period
            objective: Objective function ('growth', 'sharpe', 'sortino', or 'diversification')
            target_volatility: Target annualized volatility (only used if objective is 'growth')
            
        Returns:
            Optimal diversity parameter p
        """
        def objective_function(p):
            # Generate diversity-weighted portfolio with parameter p
            portfolio = self.portfolio_generator.diversity_weighted(p[0])
            
            # Filter to optimization period
            portfolio = portfolio.loc[start_date:end_date]
            
            # Calculate portfolio returns
            returns = self._calculate_portfolio_returns(portfolio)
            
            if len(returns) == 0:
                return 0  # No data
            
            if objective == 'growth':
                if target_volatility is not None:
                    # Calculate annualized volatility
                    vol = np.std(returns) * np.sqrt(252)
                    
                    # Penalize if volatility exceeds target
                    penalty = max(0, vol - target_volatility) * 10
                    
                    # Minimize negative growth rate with volatility penalty
                    return -np.mean(returns) * 252 + penalty
                else:
                    # Simply maximize growth rate
                    return -np.mean(returns) * 252
                
            elif objective == 'sharpe':
                # Maximize Sharpe ratio
                mean_return = np.mean(returns) * 252
                vol = np.std(returns) * np.sqrt(252)
                
                if vol == 0:
                    return 0  # Avoid division by zero
                    
                return -mean_return / vol
                
            elif objective == 'sortino':
                # Maximize Sortino ratio (downside risk only)
                mean_return = np.mean(returns) * 252
                downside_returns = returns[returns < 0]
                
                if len(downside_returns) == 0 or np.std(downside_returns) == 0:
                    return 0  # No downside risk or zero standard deviation
                    
                downside_risk = np.std(downside_returns) * np.sqrt(252)
                return -mean_return / downside_risk
                
            elif objective == 'diversification':
                # Maximize portfolio diversification
                # Use the effective number of stocks: 1 / Σ_i w_i^2
                effective_n = []
                
                for date in portfolio.index:
                    weights = portfolio.loc[date].values
                    effective_n.append(1.0 / np.sum(weights**2))
                    
                # Maximize average effective number of stocks
                return -np.mean(effective_n)
                
            else:
                raise ValueError(f"Unknown objective: {objective}")
        
        # Optimize p in (0, 1)
        result = minimize(objective_function, [0.5], bounds=[(0.01, 0.99)])
        
        return result.x[0]
    
    def _calculate_portfolio_returns(self, portfolio_weights: pd.DataFrame) -> pd.Series:
        """
        Calculate daily returns of a portfolio.
        
        Args:
            portfolio_weights: DataFrame of portfolio weights
            
        Returns:
            Series of portfolio returns
        """
        returns = pd.Series(index=portfolio_weights.index[1:])
        
        for i in range(1, len(portfolio_weights)):
            date = portfolio_weights.index[i]
            prev_date = portfolio_weights.index[i-1]
            
            if date not in self.market_model.log_returns.index:
                continue
                
            # Get stock returns
            stock_returns = self.market_model.log_returns.loc[date]
            
            # Calculate portfolio return using previous day's weights
            returns[date] = np.sum(portfolio_weights.loc[prev_date] * stock_returns)
            
        return returns
    
    def grid_search_diversity(self, 
                             start_date: pd.Timestamp, 
                             end_date: pd.Timestamp,
                             p_values: List[float],
                             metrics: List[str] = ['return', 'volatility', 'sharpe', 'max_drawdown']) -> pd.DataFrame:
        """
        Perform grid search over diversity parameters.
        
        Args:
            start_date: Start date for evaluation period
            end_date: End date for evaluation period
            p_values: List of diversity parameters to evaluate
            metrics: List of performance metrics to calculate
            
        Returns:
            DataFrame of performance metrics for each p value
        """
        results = pd.DataFrame(index=p_values, columns=metrics)
        
        for p in p_values:
            # Generate diversity-weighted portfolio
            portfolio = self.portfolio_generator.diversity_weighted(p)
            
            # Filter to evaluation period
            portfolio = portfolio.loc[start_date:end_date]
            
            # Calculate portfolio returns
            returns = self._calculate_portfolio_returns(portfolio)
            
            if len(returns) == 0:
                continue
                
            # Calculate metrics
            for metric in metrics:
                if metric == 'return':
                    results.loc[p, metric] = np.mean(returns) * 252
                elif metric == 'volatility':
                    results.loc[p, metric] = np.std(returns) * np.sqrt(252)
                elif metric == 'sharpe':
                    mean_return = np.mean(returns) * 252
                    vol = np.std(returns) * np.sqrt(252)
                    results.loc[p, metric] = mean_return / vol if vol > 0 else 0
                elif metric == 'max_drawdown':
                    # Calculate cumulative returns
                    cum_returns = (1 + returns).cumprod()
                    # Calculate running maximum
                    running_max = cum_returns.cummax()
                    # Calculate drawdown
                    drawdown = (cum_returns / running_max) - 1
                    # Get maximum drawdown
                    results.loc[p, metric] = drawdown.min()
                elif metric == 'turnover':
                    # Calculate turnover
                    turnover = pd.Series(index=portfolio.index[1:])
                    for i in range(1, len(portfolio)):
                        date = portfolio.index[i]
                        prev_date = portfolio.index[i-1]
                        weight_changes = np.abs(portfolio.loc[date] - portfolio.loc[prev_date])
                        turnover[date] = 0.5 * weight_changes.sum()
                    results.loc[p, metric] = turnover.mean()
                elif metric == 'effective_n':
                    # Calculate effective number of stocks
                    effective_n = []
                    for date in portfolio.index:
                        weights = portfolio.loc[date].values
                        effective_n.append(1.0 / np.sum(weights**2))
                    results.loc[p, metric] = np.mean(effective_n)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                    
        return results
    
    def rolling_optimal_diversity(self, 
                                 window: int = 252,
                                 objective: str = 'sharpe',
                                 min_window: int = 63) -> pd.Series:
        """
        Calculate optimal diversity parameter using a rolling window.
        
        Args:
            window: Rolling window size in days
            objective: Optimization objective
            min_window: Minimum window size for optimization
            
        Returns:
            Series of optimal diversity parameters over time
        """
        dates = self.market_model.market_weights.index
        optimal_p = pd.Series(index=dates[window:])
        
        for i in range(window, len(dates)):
            # Define optimization window
            start_date = dates[i - window]
            end_date = dates[i - 1]
            
            # Check if we have enough data
            returns_in_window = len(self.market_model.log_returns.loc[start_date:end_date])
            if returns_in_window < min_window:
                optimal_p[dates[i]] = np.nan
                continue
                
            # Optimize diversity parameter
            try:
                p_opt = self.optimize_diversity_parameter(start_date, end_date, objective)
                optimal_p[dates[i]] = p_opt
            except Exception as e:
                print(f"Optimization failed for window ending {end_date}: {e}")
                optimal_p[dates[i]] = np.nan
                
        return optimal_p
