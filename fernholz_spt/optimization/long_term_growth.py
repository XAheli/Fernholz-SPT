import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from scipy.optimize import minimize
import cvxpy as cp

from fernholz_spt.core.market_model import MarketModel
from fernholz_spt.core.portfolio_generation import FunctionallyGeneratedPortfolio
from fernholz_spt.core.rank_based import RankBasedPortfolio

class LongTermGrowthOptimizer:
    """
    Optimization methods for long-term growth using Fernholz's theory.
    
    Fernholz showed that the long-term growth rate of a portfolio can be
    decomposed into the market growth rate plus the drift process. This class
    implements optimization methods to maximize long-term growth.
    """
    
    def __init__(self, market_model: MarketModel):
        """
        Initialize with a market model.
        
        Args:
            market_model: Initialized MarketModel instance
        """
        self.market_model = market_model
        self.fg_portfolio = FunctionallyGeneratedPortfolio(market_model)
        self.rank_portfolio = RankBasedPortfolio(market_model)
        
    def optimize_growth_rate(self, 
                            date: pd.Timestamp,
                            lookback_window: int = 252,
                            regularization: float = 0.0,
                            min_weight: float = 0.0,
                            max_weight: float = 0.1) -> np.ndarray:
        """
        Optimize portfolio weights to maximize expected long-term growth rate.
        
        The long-term growth rate g(π) of a portfolio π is:
        
        g(π) = g(μ) + γ(π,μ)
        
        where g(μ) is the market growth rate and γ(π,μ) is the drift process.
        
        Args:
            date: Date for optimization
            lookback_window: Window for estimating covariance and growth
            regularization: L2 regularization parameter
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            
        Returns:
            Array of optimal weights
        """
        # Check if we have enough history
        market_history = self.market_model.market_weights.loc[:date]
        if len(market_history) < lookback_window:
            raise ValueError(f"Not enough history for date {date}. Required: {lookback_window}, available: {len(market_history)}")
            
        # Get start date for lookback window
        start_idx = market_history.index.get_loc(date) - lookback_window + 1
        if start_idx < 0:
            start_idx = 0
        start_date = market_history.index[start_idx]
        
        # Get market weights and returns
        mu = self.market_model.market_weights.loc[date].values
        
        # Get returns and estimate covariance
        returns_history = self.market_model.log_returns.loc[start_date:date]
        cov_matrix = returns_history.cov().values
        expected_returns = returns_history.mean().values
        
        n_stocks = len(mu)
        
        # Objective function: maximize growth rate
        def objective(weights):
            # Expected return component
            expected_return = np.dot(weights, expected_returns)
            
            # Variance penalty (from drift process)
            variance_penalty = 0.5 * weights.T @ cov_matrix @ weights
            
            # Drift component (excess growth rate)
            relative_weights = np.zeros_like(weights)
            nonzero_idx = mu > 0
            relative_weights[nonzero_idx] = weights[nonzero_idx] / mu[nonzero_idx]
            drift = 0.5 * np.sum(np.diag(cov_matrix) * (relative_weights - 1)**2)
            
            # Regularization penalty
            reg_penalty = regularization * np.sum(weights**2)
            
            # Maximize growth rate: expected_return - variance_penalty + drift
            return -(expected_return - variance_penalty + drift) + reg_penalty
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds: min_weight <= w_i <= max_weight
        bounds = [(min_weight, max_weight) for _ in range(n_stocks)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_stocks) / n_stocks
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        if not result.success:
            print(f"Warning: Optimization failed for date {date}: {result.message}")
            
        return result.x
    
    def construct_optimal_growth_portfolio(self, 
                                          lookback_window: int = 252,
                                          regularization: float = 0.0,
                                          min_weight: float = 0.0,
                                          max_weight: float = 0.1) -> pd.DataFrame:
        """
        Construct a portfolio that maximizes expected long-term growth rate.
        
        Args:
            lookback_window: Window for estimating covariance and growth
            regularization: L2 regularization parameter
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            
        Returns:
            DataFrame of optimal portfolio weights
        """
        weights = pd.DataFrame(index=self.market_model.market_weights.index[lookback_window:], 
                              columns=self.market_model.stock_names)
        
        # Get valid dates (those with enough history)
        valid_dates = self.market_model.market_weights.index[lookback_window:]
        
        for date in valid_dates:
            try:
                opt_weights = self.optimize_growth_rate(date, lookback_window, regularization,
                                                      min_weight, max_weight)
                weights.loc[date] = opt_weights
            except Exception as e:
                print(f"Optimization failed for date {date}: {e}")
                # Fallback to equal weights
                weights.loc[date] = 1.0 / self.market_model.n_stocks
                
        return weights
    
    def optimize_functionally_generated_portfolio(self, 
                                                start_date: pd.Timestamp, 
                                                end_date: pd.Timestamp,
                                                portfolio_type: str = 'diversity',
                                                objective: str = 'growth') -> Dict[str, float]:
        """
        Find optimal parameters for functionally generated portfolios.
        
        Args:
            start_date: Start date for optimization period
            end_date: End date for optimization period
            portfolio_type: Type of portfolio ('diversity', 'entropy', or 'volatility')
            objective: Objective function ('growth', 'sharpe', or 'sortino')
            
        Returns:
            Dictionary of optimal parameters
        """
        if portfolio_type == 'diversity':
            # Grid search for diversity parameter
            p_values = np.linspace(0.01, 0.99, 20)
            best_objective = -np.inf
            best_p = 0.5
            
            for p in p_values:
                portfolio = self.fg_portfolio.diversity_weighted(p)
                portfolio = portfolio.loc[start_date:end_date]
                
                returns = self._calculate_portfolio_returns(portfolio)
                
                if len(returns) == 0:
                    continue
                
                if objective == 'growth':
                    obj_value = np.mean(returns) * 252
                elif objective == 'sharpe':
                    mean_return = np.mean(returns) * 252
                    vol = np.std(returns) * np.sqrt(252)
                    obj_value = mean_return / vol if vol > 0 else 0
                elif objective == 'sortino':
                    mean_return = np.mean(returns) * 252
                    downside_returns = returns[returns < 0]
                    downside_risk = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
                    obj_value = mean_return / downside_risk if downside_risk > 0 else 0
                else:
                    raise ValueError(f"Unknown objective: {objective}")
                
                if obj_value > best_objective:
                    best_objective = obj_value
                    best_p = p
                    
            return {'diversity_parameter': best_p}
        
        elif portfolio_type == 'entropy':
            # No parameters to optimize for entropy-weighted portfolio
            return {}
        
        elif portfolio_type == 'volatility':
            # Grid search for volatility window
            window_values = [21, 63, 126, 252]
            best_objective = -np.inf
            best_window = 252
            
            for window in window_values:
                # Skip if window is too large for the data
                if window >= len(self.market_model.log_returns.loc[:end_date]):
                    continue
                    
                portfolio = self.fg_portfolio.volatility_weighted(window)
                portfolio = portfolio.loc[start_date:end_date]
                
                returns = self._calculate_portfolio_returns(portfolio)
                
                if len(returns) == 0:
                    continue
                
                if objective == 'growth':
                    obj_value = np.mean(returns) * 252
                elif objective == 'sharpe':
                    mean_return = np.mean(returns) * 252
                    vol = np.std(returns) * np.sqrt(252)
                    obj_value = mean_return / vol if vol > 0 else 0
                elif objective == 'sortino':
                    mean_return = np.mean(returns) * 252
                    downside_returns = returns[returns < 0]
                    downside_risk = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
                    obj_value = mean_return / downside_risk if downside_risk > 0 else 0
                else:
                    raise ValueError(f"Unknown objective: {objective}")
                
                if obj_value > best_objective:
                    best_objective = obj_value
                    best_window = window
                    
            return {'window': best_window}
        
        else:
            raise ValueError(f"Unknown portfolio type: {portfolio_type}")
    
    def optimize_rank_parameters(self, 
                               start_date: pd.Timestamp, 
                               end_date: pd.Timestamp,
                               portfolio_type: str = 'top_m',
                               objective: str = 'growth') -> Dict[str, Union[int, str]]:
        """
        Find optimal parameters for rank-based portfolios.
        
        Args:
            start_date: Start date for optimization period
            end_date: End date for optimization period
            portfolio_type: Type of portfolio ('top_m', 'bottom_m', or 'leaking')
            objective: Objective function ('growth', 'sharpe', or 'sortino')
            
        Returns:
            Dictionary of optimal parameters
        """
        if portfolio_type == 'top_m':
            # Grid search for top m and weighting scheme
            m_values = [int(self.market_model.n_stocks * x) for x in [0.1, 0.2, 0.3, 0.4, 0.5]]
            weighting_schemes = ['equal', 'cap', 'inverse_cap']
            
            best_objective = -np.inf
            best_params = {'m': m_values[0], 'weighting': weighting_schemes[0]}
            
            for m in m_values:
                for weighting in weighting_schemes:
                    portfolio = self.rank_portfolio.top_m_portfolio(m, weighting)
                    portfolio = portfolio.loc[start_date:end_date]
                    
                    returns = self._calculate_portfolio_returns(portfolio)
                    
                    if len(returns) == 0:
                        continue
                    
                    if objective == 'growth':
                        obj_value = np.mean(returns) * 252
                    elif objective == 'sharpe':
                        mean_return = np.mean(returns) * 252
                        vol = np.std(returns) * np.sqrt(252)
                        obj_value = mean_return / vol if vol > 0 else 0
                    elif objective == 'sortino':
                        mean_return = np.mean(returns) * 252
                        downside_returns = returns[returns < 0]
                        downside_risk = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
                        obj_value = mean_return / downside_risk if downside_risk > 0 else 0
                    else:
                        raise ValueError(f"Unknown objective: {objective}")
                    
                    if obj_value > best_objective:
                        best_objective = obj_value
                        best_params = {'m': m, 'weighting': weighting}
                        
            return best_params
        
        elif portfolio_type == 'bottom_m':
            # Grid search for bottom m and weighting scheme
            m_values = [int(self.market_model.n_stocks * x) for x in [0.1, 0.2, 0.3, 0.4, 0.5]]
            weighting_schemes = ['equal', 'cap', 'inverse_cap']
            
            best_objective = -np.inf
            best_params = {'m': m_values[0], 'weighting': weighting_schemes[0]}
            
            for m in m_values:
                for weighting in weighting_schemes:
                    portfolio = self.rank_portfolio.bottom_m_portfolio(m, weighting)
                    portfolio = portfolio.loc[start_date:end_date]
                    
                    returns = self._calculate_portfolio_returns(portfolio)
                    
                    if len(returns) == 0:
                        continue
                    
                    if objective == 'growth':
                        obj_value = np.mean(returns) * 252
                    elif objective == 'sharpe':
                        mean_return = np.mean(returns) * 252
                        vol = np.std(returns) * np.sqrt(252)
                        obj_value = mean_return / vol if vol > 0 else 0
                    elif objective == 'sortino':
                        mean_return = np.mean(returns) * 252
                        downside_returns = returns[returns < 0]
                        downside_risk = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
                        obj_value = mean_return / downside_risk if downside_risk > 0 else 0
                    else:
                        raise ValueError(f"Unknown objective: {objective}")
                    
                    if obj_value > best_objective:
                        best_objective = obj_value
                        best_params = {'m': m, 'weighting': weighting}
                        
            return best_params
        
        elif portfolio_type == 'leaking':
            # Grid search for alpha parameter
            alpha_values = np.linspace(0.1, 0.9, 9)
            
            best_objective = -np.inf
            best_alpha = 0.5
            
            for alpha in alpha_values:
                portfolio = self.rank_portfolio.leaking_portfolio(alpha)
                portfolio = portfolio.loc[start_date:end_date]
                
                returns = self._calculate_portfolio_returns(portfolio)
                
                if len(returns) == 0:
                    continue
                
                if objective == 'growth':
                    obj_value = np.mean(returns) * 252
                elif objective == 'sharpe':
                    mean_return = np.mean(returns) * 252
                    vol = np.std(returns) * np.sqrt(252)
                    obj_value = mean_return / vol if vol > 0 else 0
                elif objective == 'sortino':
                    mean_return = np.mean(returns) * 252
                    downside_returns = returns[returns < 0]
                    downside_risk = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
                    obj_value = mean_return / downside_risk if downside_risk > 0 else 0
                else:
                    raise ValueError(f"Unknown objective: {objective}")
                
                if obj_value > best_objective:
                    best_objective = obj_value
                    best_alpha = alpha
                    
            return {'alpha': best_alpha}
        
        else:
            raise ValueError(f"Unknown portfolio type: {portfolio_type}")
    
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
