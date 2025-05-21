import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fernholz_spt.core.market_model import MarketModel
from fernholz_spt.core.portfolio_generation import FunctionallyGeneratedPortfolio
from fernholz_spt.simulation.performance_analysis import PortfolioAnalyzer
from fernholz_spt.utils.visualization import SPTVisualizer


# Load sample data
prices = pd.read_csv('data/NIFTY 50.csv', index_col=0)
prices.index = pd.to_datetime(prices.index, errors='coerce', format='%d-%b-%Y')
market_model = MarketModel(prices)

# Create diversity-weighted portfolio
fg = FunctionallyGeneratedPortfolio(market_model)
div_weights = fg.diversity_weighted(p=0.5)

# Ensure the weights DataFrame is numeric and clean invalid data
div_weights = div_weights.apply(pd.to_numeric, errors='coerce').dropna(how='any', axis=0)

# Analyze performance
analyzer = PortfolioAnalyzer(market_model)
drift = analyzer.calculate_excess_growth(div_weights)

# Print cleaned DataFrame
print("Cleaned Weights DataFrame:")
print(div_weights.dtypes)
print(div_weights.head())
print(div_weights.tail())

# Visualize
SPTVisualizer.plot_weight_evolution(div_weights)
SPTVisualizer.plot_drift_analysis(drift)
plt.show()