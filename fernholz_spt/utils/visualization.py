import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd

class SPTVisualizer:
    """
    Visualization tools for Stochastic Portfolio Theory analysis.
    """
    
    
    @staticmethod
    def plot_weight_evolution(weights: pd.DataFrame,
                             top_n: int = 10,
                             figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot the evolution of portfolio weights over time.
        
        Args:
            weights: DataFrame of portfolio weights
            top_n: Number of top stocks to highlight
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        # Ensure the entire DataFrame is numeric
        weights = weights.apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with NaN values if necessary
        weights = weights.dropna(how='any', axis=0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Check if DataFrame is empty after cleaning
        if weights.empty:
            ax.text(0.5, 0.5, "No valid weight data available", 
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title("Portfolio Weight Evolution")
            return fig
        
        # Get top stocks by final weight
        top_stocks = weights.iloc[-1].nlargest(min(top_n, len(weights.columns))).index
                
        # Get top stocks by final weight
        top_stocks = weights.iloc[-1].nlargest(top_n).index
        
        
        # Plot weights
        for stock in weights.columns:
            if stock in top_stocks:
                ax.plot(weights.index, weights[stock], label=stock)
            else:
                ax.plot(weights.index, weights[stock], alpha=0.2, color='gray')
                
        ax.set_title("Portfolio Weight Evolution")
        ax.set_ylabel("Weight")
        ax.legend()
        return fig
    

    
    @staticmethod
    def plot_rank_distribution(ranked_weights: pd.DataFrame,
                              date: pd.Timestamp,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the distribution of market weights by rank.
        
        Args:
            ranked_weights: DataFrame of ranked market weights
            date: Date to visualize
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get weights for the specified date
        weights = ranked_weights.loc[date]
        
        # Create rank labels
        ranks = [f"Rank {i+1}" for i in range(len(weights))]
        
        ax.bar(ranks, weights)
        ax.set_title(f"Market Weight Distribution by Rank ({date.date()})")
        ax.set_ylabel("Market Weight")
        ax.tick_params(axis='x', rotation=45)
        return fig
    
    @staticmethod
    def plot_drift_analysis(drift_series: pd.Series,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot the drift process over time with statistical highlights.
        
        Args:
            drift_series: Series of drift values
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Time series plot
        drift_series.plot(ax=ax1, title="Drift Process Over Time")
        ax1.set_ylabel("Drift")
        
        # Distribution plot
        sns.histplot(drift_series.dropna(), ax=ax2, kde=True)
        ax2.set_title("Drift Distribution")
        ax2.set_xlabel("Drift Value")
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_cumulative_returns(returns: pd.Series,
                               benchmark: Optional[pd.Series] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot cumulative log returns of a portfolio with optional benchmark.
        
        Args:
            returns: Series of portfolio returns
            benchmark: Series of benchmark returns
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        cum_returns = returns.cumsum()
        ax.plot(cum_returns, label='Portfolio')
        
        if benchmark is not None:
            cum_bench = benchmark.cumsum()
            ax.plot(cum_bench, label='Benchmark')
            
        ax.set_title("Cumulative Log Returns")
        ax.set_ylabel("Log Return")
        ax.legend()
        return fig
    
    @staticmethod
    def plot_portfolio_comparison(portfolio_dict: Dict[str, pd.DataFrame],
                                 metric: str = 'Excess Growth',
                                 figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Compare multiple portfolios using a specified metric.
        
        Args:
            portfolio_dict: Dictionary of {name: portfolio_weights}
            metric: Metric to compare ('Excess Growth', 'Turnover', etc.)
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        analyzer = PortfolioAnalyzer()
        for name, weights in portfolio_dict.items():
            metrics = analyzer.calculate_fernhholz_metrics(weights)
            metrics[metric].plot(ax=ax, label=name)
            
        ax.set_title(f"Portfolio Comparison - {metric}")
        ax.legend()
        return fig
    
    @staticmethod
    def plot_optimization_surface(results: pd.DataFrame,
                                 param_1: str,
                                 param_2: str,
                                 z: str = 'Sharpe',
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create 3D surface plot of optimization results.
        
        Args:
            results: DataFrame with optimization results
            param_1: First parameter for x-axis
            param_2: Second parameter for y-axis
            z: Metric for z-axis
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        x = results[param_1]
        y = results[param_2]
        z_vals = results[z]
        
        ax.plot_trisurf(x, y, z_vals, cmap='viridis', edgecolor='none')
        ax.set_xlabel(param_1)
        ax.set_ylabel(param_2)
        ax.set_zlabel(z)
        ax.set_title(f"Optimization Surface: {param_1} vs {param_2} for {z}")
        return fig
