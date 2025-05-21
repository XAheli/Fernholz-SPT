import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from scipy.stats import norm
from scipy.integrate import quad

class StochasticProcesses:
    """
    Implementation of stochastic processes used in Stochastic Portfolio Theory.
    
    This class provides tools for simulating and analyzing stochastic processes,
    particularly those relevant to Fernholz's Stochastic Portfolio Theory, such as:
    - Standard Brownian motion
    - Geometric Brownian motion
    - Fractional Brownian motion
    - Volterra processes
    """
    
    @staticmethod
    def simulate_brownian_motion(n_paths: int, 
                                n_steps: int, 
                                T: float = 1.0, 
                                seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate paths of standard Brownian motion.
        
        Args:
            n_paths: Number of paths to simulate
            n_steps: Number of time steps
            T: Time horizon
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_paths, n_steps+1) containing simulated paths
        """
        if seed is not None:
            np.random.seed(seed)
            
        dt = T / n_steps
        dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
        W = np.zeros((n_paths, n_steps + 1))
        
        # Cumulative sum to get Brownian motion paths
        W[:, 1:] = np.cumsum(dW, axis=1)
        
        return W
    
    @staticmethod
    def simulate_geometric_brownian_motion(n_paths: int, 
                                          n_steps: int, 
                                          mu: float, 
                                          sigma: float, 
                                          S0: float = 1.0,
                                          T: float = 1.0, 
                                          seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate paths of geometric Brownian motion.
        
        The SDE for geometric Brownian motion is:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t
        
        Args:
            n_paths: Number of paths to simulate
            n_steps: Number of time steps
            mu: Drift parameter
            sigma: Volatility parameter
            S0: Initial value
            T: Time horizon
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_paths, n_steps+1) containing simulated paths
        """
        # Simulate Brownian motion
        W = StochasticProcesses.simulate_brownian_motion(n_paths, n_steps, T, seed)
        
        # Time grid
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        
        # Geometric Brownian motion formula
        S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
        
        return S
    
    @staticmethod
    def simulate_fractional_brownian_motion(n_paths: int, 
                                           n_steps: int, 
                                           hurst: float, 
                                           T: float = 1.0,
                                           method: str = 'cholesky',
                                           seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate paths of fractional Brownian motion.
        
        Fractional Brownian motion is a generalization of Brownian motion where
        the increments can be correlated. The correlation is determined by the
        Hurst parameter H. For H = 0.5, fBm reduces to standard Brownian motion.
        
        Args:
            n_paths: Number of paths to simulate
            n_steps: Number of time steps
            hurst: Hurst parameter (0 < H < 1)
            T: Time horizon
            method: Simulation method ('cholesky' or 'davies-harte')
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_paths, n_steps+1) containing simulated paths
        """
        if seed is not None:
            np.random.seed(seed)
            
        if hurst <= 0 or hurst >= 1:
            raise ValueError(f"Hurst parameter must be in (0, 1), got {hurst}")
            
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        # Initialize fbm array
        fbm = np.zeros((n_paths, n_steps + 1))
        
        if method == 'cholesky':
            # Construct covariance matrix
            cov = np.zeros((n_steps + 1, n_steps + 1))
            for i in range(n_steps + 1):
                for j in range(i, n_steps + 1):
                    cov[i, j] = 0.5 * (times[i]**(2*hurst) + times[j]**(2*hurst) - 
                                       np.abs(times[i] - times[j])**(2*hurst))
                    cov[j, i] = cov[i, j]
            
            # Cholesky decomposition
            L = np.linalg.cholesky(cov)
            
            # Simulate paths
            for i in range(n_paths):
                Z = np.random.normal(0, 1, n_steps + 1)
                fbm[i] = np.dot(L, Z)
        
        elif method == 'davies-harte':
            # Implementation of Davies-Harte method
            # This method is more efficient for large n_steps
            
            # Calculate autocovariance function
            def acf(k):
                if k == 0:
                    return 1.0
                else:
                    return 0.5 * (np.abs(k+1)**(2*hurst) - 2*np.abs(k)**(2*hurst) + 
                                  np.abs(k-1)**(2*hurst))
            
            # Construct first row of circulant matrix
            n = 2 * n_steps
            circulant_first_row = np.zeros(n)
            for k in range(n_steps + 1):
                circulant_first_row[k] = acf(k)
            circulant_first_row[n_steps+1:] = circulant_first_row[n_steps-1:0:-1]
            
            # Compute eigenvalues of circulant matrix
            eigenvalues = np.fft.fft(circulant_first_row).real
            
            for i in range(n_paths):
                # Generate normal random variables
                Z = np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)
                
                # Multiply by sqrt of eigenvalues
                W_hat = Z * np.sqrt(eigenvalues / n)
                
                # Inverse FFT
                W = np.fft.ifft(W_hat)
                
                # Extract real part of first n_steps + 1 elements
                fbm[i] = np.sqrt(T) * W[:n_steps+1].real
        
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return fbm
    
    @staticmethod
    def simulate_volterra_process(n_paths: int, 
                                 n_steps: int, 
                                 kernel_func: Callable[[float, float], float],
                                 T: float = 1.0,
                                 seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate paths of a Volterra process.
        
        A Volterra process X_t is defined as:
        
        X_t = ∫_0^t K(t, s) dW_s
        
        where K(t, s) is a kernel function and W_s is a standard Brownian motion.
        
        Args:
            n_paths: Number of paths to simulate
            n_steps: Number of time steps
            kernel_func: Kernel function K(t, s)
            T: Time horizon
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_paths, n_steps+1) containing simulated paths
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Simulate Brownian motion
        W = StochasticProcesses.simulate_brownian_motion(n_paths, n_steps, T, seed)
        
        # Time grid
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)
        
        # Initialize Volterra process
        X = np.zeros((n_paths, n_steps + 1))
        
        # Compute Volterra process using convolution
        for i in range(n_paths):
            for j in range(1, n_steps + 1):
                t = t_grid[j]
                
                # Compute kernel values for all previous times
                kernel_values = np.array([kernel_func(t, t_grid[k]) for k in range(j)])
                
                # Compute Brownian increments
                dW = W[i, 1:j+1] - W[i, 0:j]
                
                # Approximate the integral
                X[i, j] = np.sum(kernel_values * dW)
                
        return X
    
    @staticmethod
    def power_law_kernel(t: float, s: float, alpha: float = 0.1) -> float:
        """
        Power-law kernel function used in rough volatility models.
        
        K(t, s) = (t - s)^(alpha - 1/2) / Gamma(alpha + 1/2)
        
        Args:
            t: Current time
            s: Integration variable
            alpha: Roughness parameter (alpha < 0 for rough paths)
            
        Returns:
            Kernel value
        """
        from scipy.special import gamma
        
        if s >= t:
            return 0
        
        return (t - s)**(alpha - 0.5) / gamma(alpha + 0.5)
    
    @staticmethod
    def simulate_rough_bergomi(n_paths: int, 
                              n_steps: int, 
                              xi0: float, 
                              eta: float, 
                              rho: float,
                              hurst: float,
                              T: float = 1.0,
                              S0: float = 1.0,
                              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the rough Bergomi stochastic volatility model.
        
        The rough Bergomi model is defined by:
        
        dS_t = sqrt(V_t) * S_t * dW^S_t
        V_t = xi0 * exp(eta * X_t - 0.5 * eta^2 * t^(2H))
        X_t = ∫_0^t (t-s)^(H-1/2) dW^V_s
        corr(dW^S_t, dW^V_t) = rho
        
        Args:
            n_paths: Number of paths to simulate
            n_steps: Number of time steps
            xi0: Initial variance
            eta: Volatility of volatility
            rho: Correlation between price and variance processes
            hurst: Hurst parameter
            T: Time horizon
            S0: Initial stock price
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (stock_paths, variance_paths) arrays
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Time grid
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)
        
        # Simulate correlated Brownian motions
        W_S = StochasticProcesses.simulate_brownian_motion(n_paths, n_steps, T, seed)
        W_V = rho * W_S + np.sqrt(1 - rho**2) * StochasticProcesses.simulate_brownian_motion(n_paths, n_steps, T, seed + 1 if seed is not None else None)
        
        # Define power law kernel for rough volatility
        def kernel(t, s):
            return StochasticProcesses.power_law_kernel(t, s, hurst - 0.5)
            
        # Simulate variance process using Volterra representation
        X = StochasticProcesses.simulate_volterra_process(n_paths, n_steps, kernel, T, seed)
        
        # Compute variance process
        V = np.zeros((n_paths, n_steps + 1))
        for i in range(n_paths):
            for j in range(n_steps + 1):
                t = t_grid[j]
                V[i, j] = xi0 * np.exp(eta * X[i, j] - 0.5 * eta**2 * t**(2*hurst))
        
        # Simulate stock price process
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        
        for i in range(n_paths):
            for j in range(1, n_steps + 1):
                sqrt_v = np.sqrt(V[i, j-1])
                dW = W_S[i, j] - W_S[i, j-1]
                S[i, j] = S[i, j-1] * np.exp(-0.5 * V[i, j-1] * dt + sqrt_v * dW)
                
        return S, V
