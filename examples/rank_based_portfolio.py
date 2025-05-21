import pandas as pd
import matplotlib.pyplot as plt
from fernholz_spt.core.market_model import MarketModel
from fernholz_spt.core.rank_based import RankBasedPortfolio
from fernholz_spt.utils.visualization import SPTVisualizer
from fernholz_spt.simulation.performance_analysis import PortfolioAnalyzer

def main():
    # Load sample data
    prices = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)
    market_model = MarketModel(prices)
    
    # Create rank-based portfolios
    rank_manager = RankBasedPortfolio(market_model)
    top_20 = rank_manager.top_m_portfolio(20, weighting='equal')
    bottom_20 = rank_manager.bottom_m_portfolio(20, weighting='equal')
    
    # Analyze performance
    analyzer = PortfolioAnalyzer(market_model)
    top_drift = analyzer.calculate_excess_growth(top_20)
    bottom_drift = analyzer.calculate_excess_growth(bottom_20)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    top_20.sum(axis=1).plot(ax=ax1, title="Top 20 Portfolio Value")
    bottom_20.sum(axis=1).plot(ax=ax2, title="Bottom 20 Portfolio Value")
    plt.tight_layout()
    
    drift_df = pd.DataFrame({'Top 20': top_drift, 'Bottom 20': bottom_drift})
    SPTVisualizer.plot_drift_analysis(drift_df)
    plt.show()

if __name__ == "__main__":
    main()
