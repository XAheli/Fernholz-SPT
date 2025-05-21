import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from scipy.stats import norm

from fernholz_spt.core.market_model import MarketModel
from fernholz_spt.core.stochastic_processes import StochasticProcesses

class MarketSimulation:
    """
    Simulation of market dynamics using the framework of Stochastic Portfolio Theory.
    
    This class implements various methods for simulating market dynamics,
    particularly the evolution of market weights according to stochastic processes.
    """
    
    def __init__(self, market_model: Optional[MarketModel] = None):
        """
        Initialize with an optional market model.
        
        Args:
            market_model: Initialized MarketModel instance
        """
        self.market_model = market_model
        self.stochastic_processes = StochasticProcesses()
        
    def simulate_market_weights(self, 
                               initial_weights: np.ndarray,
                               volatilities: np.ndarray,
                               correlations: np.ndarray,
                               drift: Optional[np.ndarray] = None,
                               n_paths: int = 100,
                               n_steps: int = 252,
                               dt: float = 1/252,
                               seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate the evolution of market weights.
        
        Market weights follow a stochastic process constrained to the unit simplex.
        The approach uses the transformation from Fernholz's work.
        
        Args:
            initial_weights: Initial market weights
            volatilities: Volatility parameters for each stock
            correlations: Correlation matrix for stock returns
            drift: Optional drift parameters
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step size
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_paths, n_steps+1, n_stocks) containing simulated market weights
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_stocks = len(initial_weights)
        
        if drift is None:
            drift = np.zeros(n_stocks)
            
        # Check input dimensions
        if len(volatilities) != n_stocks:
            raise ValueError(f"Volatilities must have length {n_stocks}")
        if correlations.shape != (n_stocks, n_stocks):
            raise ValueError(f"Correlations must have shape ({n_stocks}, {n_stocks})")
        if len(drift) != n_stocks:
            raise ValueError(f"Drift must have length {n_stocks}")
            
        # Check that initial weights sum to 1
        if not np.isclose(np.sum(initial_weights), 1.0):
            raise ValueError("Initial weights must sum to 1")
            
        # Calculate covariance matrix
        volatility_matrix = np.diag(volatilities)
        covariance = volatility_matrix @ correlations @ volatility_matrix
        
        # Cholesky decomposition for correlated random numbers
        L = np.linalg.cholesky(correlations)
        
        # Initialize market weights array
        weights = np.zeros((n_paths, n_steps+1, n_stocks))
        weights[:, 0, :] = initial_weights
        
        # Simulate market weight paths
        for t in range(1, n_steps+1):
            for i in range(n_paths):
                # Generate correlated random variables
                Z = np.random.normal(0, 1, n_stocks)
                correlated_Z = L @ Z
                
                # Current weights
                w = weights[i, t-1, :]
                
                # Log return for each stock (adjusted for market constraint)
                log_returns = (drift - 0.5 * np.diag(covariance)) * dt + np.sqrt(dt) * volatilities * correlated_Z
                
                # Calculate new unnormalized market values
                new_values = w * np.exp(log_returns)
                
                # Normalize to get new market weights
                weights[i, t, :] = new_values / np.sum(new_values)
                
        return weights
    
    def simulate_stock_returns(self,
                              initial_prices: np.ndarray,
                              expected_returns: np.ndarray,
                              volatilities: np.ndarray,
                              correlations: np.ndarray,
                              n_paths: int = 100,
                              n_steps: int = 252,
                              dt: float = 1/252,
                              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate stock price paths and corresponding market weights.
        
        This method simulates stock prices using geometric Brownian motion
        and then calculates the market weights.
        
        Args:
            initial_prices: Initial stock prices
            expected_returns: Expected return parameters
            volatilities: Volatility parameters
            correlations: Correlation matrix
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step size
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (stock_prices, market_weights) arrays
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_stocks = len(initial_prices)
        
        # Check input dimensions
        if len(expected_returns) != n_stocks:
            raise ValueError(f"Expected returns must have length {n_stocks}")
        if len(volatilities) != n_stocks:
            raise ValueError(f"Volatilities must have length {n_stocks}")
        if correlations.shape != (n_stocks, n_stocks):
            raise ValueError(f"Correlations must have shape ({n_stocks}, {n_stocks})")
            
        # Cholesky decomposition for correlated random numbers
        L = np.linalg.cholesky(correlations)
        
        # Initialize arrays
        prices = np.zeros((n_paths, n_steps+1, n_stocks))
        weights = np.zeros((n_paths, n_steps+1, n_stocks))
        
        # Set initial values
        prices[:, 0, :] = initial_prices
        
        # Calculate initial market weights
        initial_cap = np.sum(initial_prices)
        weights[:, 0, :] = initial_prices / initial_cap
        
        # Simulate price paths
        for t in range(1, n_steps+1):
            for i in range(n_paths):
                # Generate correlated random variables
                Z = np.random.normal(0, 1, n_stocks)
                correlated_Z = L @ Z
                
                # Calculate log returns using geometric Brownian motion
                log_returns = (expected_returns - 0.5 * volatilities**2) * dt + volatilities * np.sqrt(dt) * correlated_Z
                
                # Update prices
                prices[i, t, :] = prices[i, t-1, :] * np.exp(log_returns)
                
                # Calculate market weights
                total_market_cap = np.sum(prices[i, t, :])
                weights[i, t, :] = prices[i, t, :] / total_market_cap
                
        return prices, weights
    
    def simulate_rank_based_market(self,
                                  initial_weights: np.ndarray,
                                  rank_volatilities: np.ndarray,
                                  rank_drifts: np.ndarray,
                                  n_paths: int = 100,
                                  n_steps: int = 252,
                                  dt: float = 1/252,
                                  seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate a rank-based market model where volatility and drift depend on rank.
        
        Implements the model described in Fernholz's "Stochastic Portfolio Theory":
        - Volatility and drift parameters depend on the current rank of stocks
        - Maintains the rank-based structure of the market
        
        Args:
            initial_weights: Initial market weights (sorted by rank)
            rank_volatilities: Volatility parameters for each rank
            rank_drifts: Drift parameters for each rank
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step size
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_paths, n_steps+1, n_stocks) containing simulated market weights
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_stocks = len(initial_weights)
        
        # Initialize arrays
        weights = np.zeros((n_paths, n_steps+1, n_stocks))
        weights[:, 0, :] = initial_weights
        
        # Simulate market evolution
        for t in range(1, n_steps+1):
            for i in range(n_paths):
                # Get current ranked weights (descending order)
                current_weights = weights[i, t-1, :]
                ranked_order = np.argsort(current_weights)[::-1]
                
                # Generate independent Brownian motions
                dW = np.random.normal(0, np.sqrt(dt), n_stocks)
                
                # Calculate drift and volatility based on rank
                ranked_volatilities = rank_volatilities.copy()
                ranked_drifts = rank_drifts.copy()
                
                # Compute the stochastic differential equation
                noise_terms = ranked_volatilities * dW
                drift_terms = ranked_drifts * current_weights
                
                # Update weights using rank-based dynamics
                new_weights = current_weights * np.exp(drift_terms * dt + noise_terms)
                
                # Maintain market weights on simplex
                weights[i, t, :] = new_weights / np.sum(new_weights)
                
                # Re-sort to maintain rank order
                weights[i, t, :] = np.sort(weights[i, t, :])[::-1]
                
        return weights

