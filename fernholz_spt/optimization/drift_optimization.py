import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from scipy.optimize import minimize

from fernholz_spt.core.market_model import MarketModel
from fernholz_spt.core.portfolio_generation import FunctionallyGeneratedPortfolio

class DriftOptimization:
    """
    Optimization methods based on the drift process from Stochastic Portfolio Theory.
    
    The drift process represents the excess growth rate of a portfolio relative
    to the market portfolio. Fernholz showed that maximizing the drift process
    leads to portfolios with superior long-term growth.
    """
    
    def __init__(self, market_model: MarketModel):
        """
        Initialize with a market model.
        
        Args:
            market_model: Initialized MarketModel instance
        """
        self.market_model = market_model
        self.portfolio_generator = FunctionallyGeneratedPortfolio(market_model)
        
    def optimize_weights_for_drift(self, 
                                   date: pd.Timestamp,
                                   min_weight: float = 0.0,
                                   max_weight: float = 0.1,
                                   regularization: float = 0.0) -> np.ndarray:
        """
        Optimize portfolio weights to maximize the drift process.
        
        According to Fernholz, the drift process (excess growth rate) is:
        gamma_t = 0.5 * Σ_i σ_ii * (π_i/μ_i - 1)^2
        
        Args:
            date: Date for optimization
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            regularization: L2 regularization parameter
            
        Returns:
            Array of optimal weights
        """
        if self.market_model.cov_matrices is None or date not in self.market_model.cov_matrices:
            raise ValueError(f"No covariance matrix available for date {date}")
            
        # Get market weights and covariance
        mu = self.market_model.market_weights.loc[date].values
        cov_matrix = self.market_model.cov_matrices[date]
        n_stocks = len(mu)
        
        # Objective function: maximize drift process
        def objective(weights):
            # Calculate relative weights
            relative_weights = np.zeros_like(weights)
            nonzero_idx = mu > 0
            relative_weights[nonzero_idx] = weights[nonzero_idx] / mu[nonzero_idx]
            
            # Calculate drift
            drift = 0.5 * np.sum(np.diag(cov_matrix) * (relative_weights - 1)**2)
            
            # Add regularization penalty
            penalty = regularization * np.sum(weights**2)
            
            # Minimize negative drift
            return -drift + penalty
        
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
    
    def construct_maximum_drift_portfolio(self, 
                                         min_weight: float = 0.0,
                                         max_weight: float = 0.1,
                                         regularization: float = 0.0) -> pd.DataFrame:
        """
        Construct a portfolio that maximizes the drift process for each date.
        
        Args:
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            regularization: L2 regularization parameter
            
        Returns:
            DataFrame of optimal portfolio weights
        """
        weights = pd.DataFrame(index=self.market_model.market_weights.index, 
                              columns=self.market_model.stock_names)
        
        # Only optimize for dates with covariance matrices
        if self.market_model.cov_matrices is None:
            print("Warning: No covariance matrices available. Returning equal-weighted portfolio.")
            return self.portfolio_generator.equal_weighted()
            
        dates_with_cov = [date for date in weights.index if date in self.market_model.cov_matrices]
        
        for date in dates_with_cov:
            try:
                opt_weights = self.optimize_weights_for_drift(date, min_weight, max_weight, regularization)
                weights.loc[date] = opt_weights
            except Exception as e:
                print(f"Optimization failed for date {date}: {e}")
                # Fallback to equal weights
                weights.loc[date] = 1.0 / self.market_model.n_stocks
                
        # For dates without covariance matrices, use equal weights
        missing_dates = [date for date in weights.index if date not in dates_with_cov]
        if missing_dates:
            weights.loc[missing_dates] = 1.0 / self.market_model.n_stocks
            
        return weights
    
    def optimize_convex_combination(self, 
                                   date: pd.Timestamp,
                                   portfolios: List[pd.DataFrame],
                                   min_weight: float = 0.0,
                                   max_weight: float = 1.0) -> np.ndarray:
        """
        Optimize a convex combination of portfolios to maximize drift.
        
        Args:
            date: Date for optimization
            portfolios: List of portfolio weight DataFrames
            min_weight: Minimum weight for each portfolio
            max_weight: Maximum weight for each portfolio
            
        Returns:
            Array of optimal portfolio weights
        """
        if self.market_model.cov_matrices is None or date not in self.market_model.cov_matrices:
            raise ValueError(f"No covariance matrix available for date {date}")
            
        n_portfolios = len(portfolios)
        if n_portfolios == 0:
            raise ValueError("No portfolios provided")
            
        # Extract portfolio weights for the given date
        portfolio_weights = np.array([p.loc[date].values if date in p.index else 
                                     np.ones(self.market_model.n_stocks) / self.market_model.n_stocks 
                                     for p in portfolios])
        
        # Get market weights and covariance
        mu = self.market_model.market_weights.loc[date].values
        cov_matrix = self.market_model.cov_matrices[date]
        
        # Objective function: maximize drift process
        def objective(combination_weights):
            # Calculate combined portfolio weights
            weights = np.dot(combination_weights, portfolio_weights)
            
            # Calculate relative weights
            relative_weights = np.zeros_like(weights)
            nonzero_idx = mu > 0
            relative_weights[nonzero_idx] = weights[nonzero_idx] / mu[nonzero_idx]
            
            # Calculate drift
            drift = 0.5 * np.sum(np.diag(cov_matrix) * (relative_weights - 1)**2)
            
            # Minimize negative drift
            return -drift
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds: min_weight <= w_i <= max_weight
        bounds = [(min_weight, max_weight) for _ in range(n_portfolios)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_portfolios) / n_portfolios
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        if not result.success:
            print(f"Warning: Optimization failed for date {date}: {result.message}")
            
        return result.x
    
    def construct_optimal_combination_portfolio(self, 
                                              portfolios: List[pd.DataFrame],
                                              min_weight: float = 0.0,
                                              max_weight: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Construct a portfolio that optimally combines multiple portfolios.
        
        Args:
            portfolios: List of portfolio weight DataFrames
            min_weight: Minimum weight for each portfolio
            max_weight: Maximum weight for each portfolio
            
        Returns:
            Tuple of (combined portfolio weights, combination coefficients)
        """
        if not portfolios:
            raise ValueError("No portfolios provided")
            
        # Get common dates
        common_dates = set(portfolios[0].index)
        for p in portfolios[1:]:
            common_dates = common_dates.intersection(set(p.index))
            
        # Convert to sorted list
        common_dates = sorted(list(common_dates))
        
        # Initialize result DataFrames
        combined_weights = pd.DataFrame(index=common_dates, 
                                       columns=self.market_model.stock_names)
        combination_weights = pd.DataFrame(index=common_dates, 
                                          columns=[f'Portfolio_{i+1}' for i in range(len(portfolios))])
        
        # Only optimize for dates with covariance matrices
        if self.market_model.cov_matrices is None:
            print("Warning: No covariance matrices available. Returning equal-weighted combination.")
            
            # Equal weight combination
            for date in common_dates:
                portfolio_weights = np.array([p.loc[date].values for p in portfolios])
                equal_combination = np.ones(len(portfolios)) / len(portfolios)
                combined_weights.loc[date] = np.dot(equal_combination, portfolio_weights)
                combination_weights.loc[date] = equal_combination
                
            return combined_weights, combination_weights
            
        dates_with_cov = [date for date in common_dates if date in self.market_model.cov_matrices]
        
        for date in dates_with_cov:
            try:
                # Optimize combination weights
                opt_combination = self.optimize_convex_combination(date, portfolios, min_weight, max_weight)
                combination_weights.loc[date] = opt_combination
                
                # Calculate combined portfolio weights
                portfolio_weights = np.array([p.loc[date].values for p in portfolios])
                combined_weights.loc[date] = np.dot(opt_combination, portfolio_weights)
                
            except Exception as e:
                print(f"Optimization failed for date {date}: {e}")
                # Fallback to equal weights
                equal_combination = np.ones(len(portfolios)) / len(portfolios)
                combination_weights.loc[date] = equal_combination
                
                portfolio_weights = np.array([p.loc[date].values for p in portfolios])
                combined_weights.loc[date] = np.dot(equal_combination, portfolio_weights)
                
        # For dates without covariance matrices, use equal weights
        missing_dates = [date for date in common_dates if date not in dates_with_cov]
        if missing_dates:
            for date in missing_dates:
                equal_combination = np.ones(len(portfolios)) / len(portfolios)
                combination_weights.loc[date] = equal_combination
                
                portfolio_weights = np.array([p.loc[date].values for p in portfolios])
                combined_weights.loc[date] = np.dot(equal_combination, portfolio_weights)
            
        return combined_weights, combination_weights
