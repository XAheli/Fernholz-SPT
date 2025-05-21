import pandas as pd
import matplotlib.pyplot as plt
from fernholz_spt.core.market_model import MarketModel
from fernholz_spt.optimization.long_term_growth import LongTermGrowthOptimizer
from fernholz_spt.utils.visualization import SPTVisualizer

def main():
    # Load sample data
    prices = pd.read_csv('data/NIFTY 50.csv', index_col=0, parse_dates=True)
    market_model = MarketModel(prices)
    
    # Optimize long-term growth
    growth_optimizer = LongTermGrowthOptimizer(market_model)
    optimal_weights = growth_optimizer.construct_optimal_growth_portfolio()
    
    # Analyze parameters
    params = growth_optimizer.optimize_functionally_generated_portfolio(
        start_date='2010-01-01',
        end_date='2020-01-01',
        portfolio_type='diversity'
    )
    
    print(f"Optimal diversity parameter: {params['diversity_parameter']:.2f}")
    
    # Visualize
    SPTVisualizer.plot_weight_evolution(optimal_weights)
    plt.title("Optimal Long-Term Growth Portfolio Weights")
    plt.show()

if __name__ == "__main__":
    main()
