import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union # Added Union
import pandas as pd
# Removed mpl_toolkits import from top level, moved to method

# Import PortfolioAnalyzer to calculate metrics if needed, or expect metrics pre-calculated
from fernholz_spt.simulation.performance_analysis import PortfolioAnalyzer
from fernholz_spt.core.market_model import MarketModel


class SPTVisualizer:
    """
    Visualization tools for Stochastic Portfolio Theory analysis.
    """

    @staticmethod
    def plot_weight_evolution(weights_df: pd.DataFrame,
                             top_n: int = 10,
                             figsize: Tuple[int, int] = (12, 6),
                             title: str = "Portfolio Weight Evolution") -> plt.Figure:
        """
        Plot the evolution of portfolio weights over time.

        Args:
            weights_df: DataFrame of portfolio weights (dates as index, assets as columns).
            top_n: Number of top assets (by final weight) to highlight with labels.
            figsize: Figure size.
            title: Plot title.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        if not isinstance(weights_df, pd.DataFrame) or weights_df.empty:
            ax.text(0.5, 0.5, "No weight data to plot.",
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig

        # Ensure weights are numeric and handle potential NaNs from calculations
        weights_numeric = weights_df.apply(pd.to_numeric, errors='coerce')
        # weights_cleaned = weights_numeric.dropna(how='all', axis=0) # Drop rows if all weights are NaN
        weights_cleaned = weights_numeric # Keep all dates, plot NaNs as gaps if any

        if weights_cleaned.empty:
            ax.text(0.5, 0.5, "No valid weight data after cleaning.",
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        # Determine top N assets by their mean weight or final weight
        if not weights_cleaned.empty:
            # Using mean weight to determine "top" stocks over the period
            # Or use final weight: weights_cleaned.iloc[-1].nlargest(top_n).index
            mean_weights = weights_cleaned.mean().nlargest(min(top_n, len(weights_cleaned.columns)))
            top_assets = mean_weights.index
        else:
            top_assets = pd.Index([])


        # Plot weights
        for asset in weights_cleaned.columns:
            if asset in top_assets:
                ax.plot(weights_cleaned.index, weights_cleaned[asset], label=str(asset), lw=1.5)
            else:
                ax.plot(weights_cleaned.index, weights_cleaned[asset], alpha=0.3, color='grey', lw=1.0)

        ax.set_title(title)
        ax.set_ylabel("Weight")
        ax.set_xlabel("Date")
        if not top_assets.empty :
            ax.legend(loc='upper left', bbox_to_anchor=(1,1)) # Place legend outside
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
        return fig


    @staticmethod
    def plot_rank_distribution(market_model: MarketModel, # Takes MarketModel
                              date: Union[str, pd.Timestamp],
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the distribution of market weights by rank for a specific date.

        Args:
            market_model: Initialized MarketModel instance.
            date: Date string or Timestamp to visualize.
            figsize: Figure size.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        pd_date = pd.Timestamp(date)

        if pd_date not in market_model.ranked_weights.index:
            ax.text(0.5, 0.5, f"No ranked weight data for date {pd_date.date()}",
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f"Market Weight Distribution by Rank")
            return fig

        # Get ranked weights for the specified date (rank_1, rank_2, ...)
        # These are already sorted by rank in the DataFrame construction
        weights_at_date = market_model.ranked_weights.loc[pd_date].dropna() # Drop if some ranks are NaN

        if weights_at_date.empty:
            ax.text(0.5, 0.5, f"Ranked weight data is empty for {pd_date.date()}",
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
            ranks = [f"R{i+1}" for i in range(len(weights_at_date))]
            ax.bar(ranks, weights_at_date.values)
            ax.set_ylabel("Market Weight")
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            
        ax.set_title(f"Market Weight Distribution by Rank ({pd_date.date()})")
        plt.tight_layout()
        return fig


    @staticmethod
    def plot_drift_analysis(drift_series: pd.Series,
                           figsize: Tuple[int, int] = (12, 8), # Increased height
                           title: str = "Drift Process Analysis") -> plt.Figure:
        """
        Plot the drift process over time with statistical highlights.

        Args:
            drift_series: Series of drift values.
            figsize: Figure size.
            title: Main title for the figure.

        Returns:
            Matplotlib Figure object.
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(title, fontsize=14)

        if not isinstance(drift_series, pd.Series) or drift_series.empty or drift_series.dropna().empty:
            for ax_i, sub_title in zip(axes, ["Time Series", "Distribution"]):
                 ax_i.text(0.5, 0.5, f"No drift data for {sub_title}.",
                        horizontalalignment='center', verticalalignment='center', transform=ax_i.transAxes)
                 ax_i.set_title(sub_title)
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
            return fig

        drift_cleaned = drift_series.dropna()

        # Time series plot
        drift_cleaned.plot(ax=axes[0], lw=1.5)
        axes[0].set_title("Drift Process Over Time")
        axes[0].set_ylabel("Drift Value")
        axes[0].grid(True, linestyle=':', alpha=0.7)

        # Distribution plot
        sns.histplot(drift_cleaned, ax=axes[1], kde=True, stat="density", color=sns.color_palette()[1])
        axes[1].set_title("Drift Distribution")
        axes[1].set_xlabel("Drift Value")
        axes[1].set_ylabel("Density")
        axes[1].grid(True, linestyle=':', alpha=0.7)

        # Add some stats to distribution plot
        mean_val = drift_cleaned.mean()
        median_val = drift_cleaned.median()
        std_val = drift_cleaned.std()
        axes[1].axvline(mean_val, color='r', linestyle='--', lw=1, label=f'Mean: {mean_val:.4f}')
        axes[1].axvline(median_val, color='g', linestyle=':', lw=1, label=f'Median: {median_val:.4f}')
        axes[1].legend(fontsize=8)
        
        # Add overall stats text
        stats_text = f"Overall Mean: {mean_val:.4f}\nStd Dev: {std_val:.4f}\nMin: {drift_cleaned.min():.4f}\nMax: {drift_cleaned.max():.4f}"
        axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', alpha=0.8))


        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
        return fig

    @staticmethod
    def plot_cumulative_returns(portfolio_log_returns: pd.Series,
                               benchmark_log_returns: Optional[pd.Series] = None,
                               figsize: Tuple[int, int] = (12, 6),
                               title: str = "Cumulative Log Returns") -> plt.Figure:
        """
        Plot cumulative log returns of a portfolio with optional benchmark.

        Args:
            portfolio_log_returns: Series of portfolio log returns.
            benchmark_log_returns: Optional Series of benchmark log returns.
            figsize: Figure size.
            title: Plot title.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        if isinstance(portfolio_log_returns, pd.Series) and not portfolio_log_returns.empty:
            cum_returns_portfolio = portfolio_log_returns.cumsum()
            ax.plot(cum_returns_portfolio.index, cum_returns_portfolio, label='Portfolio', lw=1.5)
        else:
            ax.text(0.5, 0.6, "No portfolio return data.", ha='center', va='center', transform=ax.transAxes)


        if isinstance(benchmark_log_returns, pd.Series) and not benchmark_log_returns.empty:
            # Align benchmark returns with portfolio returns if necessary
            common_idx = portfolio_log_returns.index.intersection(benchmark_log_returns.index) if isinstance(portfolio_log_returns, pd.Series) else benchmark_log_returns.index
            
            if not common_idx.empty:
                cum_returns_benchmark = benchmark_log_returns.loc[common_idx].cumsum()
                ax.plot(cum_returns_benchmark.index, cum_returns_benchmark, label='Benchmark', lw=1.5, linestyle='--')
            else:
                 ax.text(0.5, 0.4, "Benchmark data not aligned or empty.", ha='center', va='center', transform=ax.transAxes)


        ax.set_title(title)
        ax.set_ylabel("Cumulative Log Return")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        return fig


    @staticmethod
    def plot_portfolio_comparison(
        market_model: MarketModel, # Requires MarketModel
        portfolio_weights_dict: Dict[str, pd.DataFrame],
        metric_to_plot: str = 'Excess Growth', # e.g., 'Excess Growth', 'Cumulative Relative Return', 'Entropy'
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Compare multiple portfolios using a specified metric calculated by PortfolioAnalyzer.

        Args:
            market_model: Initialized MarketModel instance.
            portfolio_weights_dict: Dictionary of {name: portfolio_weights_df}.
            metric_to_plot: Metric name (column from PortfolioAnalyzer.calculate_fernholz_metrics).
            figsize: Figure size.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        analyzer = PortfolioAnalyzer(market_model) # Analyzer initialized with the market model

        if not portfolio_weights_dict:
            ax.text(0.5, 0.5, "No portfolios provided for comparison.", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Portfolio Comparison - {metric_to_plot}")
            return fig

        for name, weights_df in portfolio_weights_dict.items():
            if not isinstance(weights_df, pd.DataFrame) or weights_df.empty:
                print(f"Skipping portfolio '{name}' due to empty or invalid weights DataFrame.")
                continue
            
            try:
                # Calculate all Fernholz metrics for the current portfolio
                all_metrics_df = analyzer.calculate_fernholz_metrics(weights_df)

                if metric_to_plot in all_metrics_df.columns and not all_metrics_df[metric_to_plot].dropna().empty:
                    metric_series = all_metrics_df[metric_to_plot].dropna()
                    ax.plot(metric_series.index, metric_series, label=name, lw=1.5)
                else:
                    print(f"Metric '{metric_to_plot}' not found or empty for portfolio '{name}'. Available: {all_metrics_df.columns.tolist()}")
            except Exception as e:
                print(f"Error processing portfolio '{name}': {e}")

        ax.set_title(f"Portfolio Comparison - {metric_to_plot}")
        ax.set_ylabel(metric_to_plot)
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        return fig


    @staticmethod
    def plot_optimization_surface(results_df: pd.DataFrame,
                                 param_1_name: str,
                                 param_2_name: str,
                                 metric_name: str = 'Sharpe', # Metric on z-axis
                                 figsize: Tuple[int, int] = (12, 8),
                                 title: Optional[str] = None) -> plt.Figure:
        """
        Create 3D surface plot of optimization results (e.g., from a grid search).

        Args:
            results_df: DataFrame with optimization results, must contain columns for
                        param_1_name, param_2_name, and metric_name.
            param_1_name: Name of the first parameter (x-axis).
            param_2_name: Name of the second parameter (y-axis).
            metric_name: Name of the metric to plot on the z-axis.
            figsize: Figure size.
            title: Optional plot title.

        Returns:
            Matplotlib Figure object.
        """
        from mpl_toolkits.mplot3d import Axes3D # Import here

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if not all(col in results_df.columns for col in [param_1_name, param_2_name, metric_name]):
            ax.text2D(0.5, 0.5, "One or more specified columns not in results DataFrame.",
                      horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            actual_title = title if title else f"Optimization Surface Plot Error"
            ax.set_title(actual_title)
            return fig

        x_data = results_df[param_1_name]
        y_data = results_df[param_2_name]
        z_data = results_df[metric_name]
        
        # Trisurf is good for scattered data, if data is on a grid, plot_surface is better
        # but requires meshgrid.
        try:
            ax.plot_trisurf(x_data, y_data, z_data, cmap='viridis', edgecolor='none', shade=True)
            ax.set_xlabel(param_1_name)
            ax.set_ylabel(param_2_name)
            ax.set_zlabel(metric_name)
            
            actual_title = title if title else f"Optimization: {metric_name} vs ({param_1_name}, {param_2_name})"
            ax.set_title(actual_title)
        except Exception as e:
            ax.text2D(0.5, 0.5, f"Error during 3D plotting: {e}",
                      horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            actual_title = title if title else f"Optimization Surface Plot Error"
            ax.set_title(actual_title)
            print(f"Plotting error: {e}")


        return fig