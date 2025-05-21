# Fernholz Stochastic Portfolio Theory (SPT) Library

[![PyPI version](https://badge.fury.io/py/fernholz-spt.svg)](https://badge.fury.io/py/fernholz-spt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/fernholz-spt/badge/?version=latest)](https://fernholz-spt.readthedocs.io/en/latest/?badge=latest)

A comprehensive Python implementation of E. Robert Fernholz's Stochastic Portfolio Theory framework. This library provides tools for researchers, quantitative analysts, and portfolio managers to analyze, optimize, and simulate equity portfolios using the mathematical framework of Stochastic Portfolio Theory.

## Table of Contents

- [Introduction to Stochastic Portfolio Theory](#introduction-to-stochastic-portfolio-theory)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage Examples](#detailed-usage-examples)
- [API Documentation](#api-documentation)
- [Mathematical Background](#mathematical-background)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Introduction to Stochastic Portfolio Theory

Stochastic Portfolio Theory (SPT) is a mathematical framework for analyzing portfolio behavior and market structure, developed by E. Robert Fernholz in the early 2000s. Unlike traditional portfolio theory, SPT focuses on the long-term growth rate of portfolios and provides a framework for understanding the impact of market diversity and portfolio rebalancing.

Key insights from SPT include:

1. The ability to generate portfolios that outperform the market over long time horizons without relying on return forecasts
2. Understanding the structural reasons behind the size effect and other market anomalies
3. Quantifying the benefits of portfolio diversification in terms of excess growth rate
4. Analyzing how market concentration affects portfolio performance

This library implements the core mathematical concepts from Fernholz's work, making them accessible for practical portfolio management and academic research.

## Key Features

- **Market Model Implementation**: 
  - Market weight dynamics modeling
  - Rank-based analysis
  - Diversity and concentration measures

- **Portfolio Generation**:
  - Functionally generated portfolios
  - Diversity-weighted portfolios
  - Entropy-weighted portfolios
  - Rank-based portfolios

- **Optimization Tools**:
  - Long-term growth rate optimization
  - Drift process maximization
  - Diversity parameter optimization
  - Rank-based portfolio parameter optimization

- **Simulation Capabilities**:
  - Market weight simulation
  - Rank-based market simulation
  - Stochastic process implementation (Brownian, fractional Brownian, Volterra)
  - Monte Carlo analysis for long-term performance

- **Performance Analysis**:
  - Excess growth rate calculation
  - Relative return calculation
  - Turnover analysis
  - Diversification metrics

## Installation

### Using pip

```bash
pip install fernholz-spt
```

### From source

```bash
git clone https://github.com/xaheli/fernholz-spt.git
cd fernholz-spt
pip install -e 
```

### Dependencies

The library requires the following packages:
- numpy
- pandas
- scipy
- matplotlib
- scikit-learn
- cvxpy
- seaborn

For a complete list with version requirements, see `requirements.txt`.

## Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt
from fernholz_spt.core.market_model import MarketModel
from fernholz_spt.core.portfolio_generation import FunctionallyGeneratedPortfolio
from fernholz_spt.simulation.performance_analysis import PortfolioAnalyzer

# Load price data
prices = pd.read_csv('sample_data.csv', index_col=0, parse_dates=True)

# Initialize market model
market_model = MarketModel(prices)

# Create diversity-weighted portfolio
portfolio_generator = FunctionallyGeneratedPortfolio(market_model)
diversity_portfolio = portfolio_generator.diversity_weighted(p=0.5)

# Calculate performance metrics
analyzer = PortfolioAnalyzer(market_model)
excess_growth = analyzer.calculate_excess_growth(diversity_portfolio)

# Visualize results
plt.figure(figsize=(12, 6))
excess_growth.plot()
plt.title('Excess Growth Rate of Diversity-Weighted Portfolio')
plt.ylabel('Excess Growth Rate')
plt.show()
```

## Detailed Usage Examples

### Market Model and Basic Analysis

```python
from fernholz_spt.core.market_model import MarketModel
import pandas as pd

# Load price data
prices = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)

# Initialize market model with covariance estimation
market_model = MarketModel(
    stock_prices=prices,
    estimate_covariance=True,
    cov_window=252,
    cov_method='shrinkage'
)

# Analyze market concentration
concentration = pd.DataFrame({
    'Top 5': [market_model.calculate_concentration_ratio(date, 5) for date in market_model.dates],
    'Top 10': [market_model.calculate_concentration_ratio(date, 10) for date in market_model.dates],
    'HHI': [market_model.calculate_herfindahl_index(date) for date in market_model.dates]
}, index=market_model.dates)

# Analyze diversity measures
diversity = market_model.diversity
```

### Functionally Generated Portfolios

```python
from fernholz_spt.core.portfolio_generation import FunctionallyGeneratedPortfolio

# Initialize portfolio generator
fg_portfolio = FunctionallyGeneratedPortfolio(market_model)

# Create different types of functionally generated portfolios
diversity_portfolio = fg_portfolio.diversity_weighted(p=0.5)
entropy_portfolio = fg_portfolio.entropy_weighted()
equal_portfolio = fg_portfolio.equal_weighted()

# Custom generation function: using Shannon entropy
def shannon_entropy(weights):
    valid_weights = weights[weights > 0]
    return -np.sum(valid_weights * np.log(valid_weights))

def shannon_gradient(weights):
    grad = np.zeros_like(weights)
    nonzero = weights > 0
    grad[nonzero] = -np.log(weights[nonzero]) - 1
    return grad

shannon_portfolio = fg_portfolio.custom_generated(
    generating_function=shannon_entropy,
    gradient_function=shannon_gradient
)
```

### Rank-Based Portfolios

```python
from fernholz_spt.core.rank_based import RankBasedPortfolio

# Initialize rank-based portfolio generator
rank_portfolio = RankBasedPortfolio(market_model)

# Create portfolios based on market cap ranking
top_10_portfolio = rank_portfolio.top_m_portfolio(m=10, weighting='equal')
bottom_10_portfolio = rank_portfolio.bottom_m_portfolio(m=10, weighting='inverse_cap')
small_cap_decile = rank_portfolio.decile_portfolio(decile=10, weighting='equal')

# Create a leaking portfolio (overweights small stocks)
leaking_portfolio = rank_portfolio.leaking_portfolio(alpha=0.7)

# Custom rank function: power law with sharp decay
def power_law_rank_function(rank, n_stocks):
    return rank ** (-1.5)

power_law_portfolio = rank_portfolio.generic_rank_portfolio(
    rank_weight_function=power_law_rank_function
)
```

### Long-Term Growth Optimization

```python
from fernholz_spt.optimization.long_term_growth import LongTermGrowthOptimizer

# Initialize optimizer
growth_optimizer = LongTermGrowthOptimizer(market_model)

# Find optimal growth rate portfolio
optimal_portfolio = growth_optimizer.construct_optimal_growth_portfolio(
    lookback_window=252,
    regularization=0.1,
    min_weight=0.01,
    max_weight=0.05
)

# Optimize diversity parameter
start_date = pd.Timestamp('2010-01-01')
end_date = pd.Timestamp('2020-01-01')
params = growth_optimizer.optimize_functionally_generated_portfolio(
    start_date=start_date,
    end_date=end_date,
    portfolio_type='diversity',
    objective='sharpe'
)
optimal_p = params['diversity_parameter']
print(f"Optimal diversity parameter: {optimal_p:.3f}")
```

### Market Simulation

```python
from fernholz_spt.simulation.market_simulation import MarketSimulation

# Initialize simulator
simulator = MarketSimulation()

# Set parameters
n_stocks = 100
initial_weights = np.ones(n_stocks) / n_stocks
volatilities = np.linspace(0.2, 0.4, n_stocks)
correlations = np.eye(n_stocks) # Diagonal correlation matrix for simplicity

# Simulate market weight evolution
simulated_weights = simulator.simulate_market_weights(
    initial_weights=initial_weights,
    volatilities=volatilities,
    correlations=correlations,
    n_paths=100,
    n_steps=252
)

# Simulate rank-based market
# Higher volatility for smaller stocks
rank_volatilities = np.linspace(0.2, 0.5, n_stocks)[::-1]

# Small positive drift for smaller stocks
rank_drifts = np.linspace(0, 0.02, n_stocks)[::-1]

simulated_rank_market = simulator.simulate_rank_based_market(
    initial_weights=np.sort(initial_weights)[::-1], # Sorted by rank
    rank_volatilities=rank_volatilities,
    rank_drifts=rank_drifts,
    n_paths=100,
    n_steps=252
)
```

### Performance Analysis

```python
from fernholz_spt.simulation.performance_analysis import PortfolioAnalyzer
from fernholz_spt.utils.visualization import SPTVisualizer

# Initialize analyzer
analyzer = PortfolioAnalyzer(market_model)

# Calculate excess growth for different portfolios
diversity_excess_growth = analyzer.calculate_excess_growth(diversity_portfolio)
equal_excess_growth = analyzer.calculate_excess_growth(equal_portfolio)
optimal_excess_growth = analyzer.calculate_excess_growth(optimal_portfolio)

# Calculate relative returns vs. market
diversity_rel_return = analyzer.calculate_relative_return(
    diversity_portfolio,
    market_model.market_weights
)

# Calculate turnover statistics
turnover_stats = analyzer.analyze_turnover(optimal_portfolio)

# Visualize results
SPTVisualizer.plot_weight_evolution(optimal_portfolio)
SPTVisualizer.plot_drift_analysis(optimal_excess_growth)

# Compare portfolios
portfolio_dict = {
    'Diversity': diversity_portfolio,
    'Equal': equal_portfolio,
    'Optimal': optimal_portfolio
}
SPTVisualizer.plot_portfolio_comparison(portfolio_dict, metric='Excess Growth')
```

## API Documentation

### Core Module

- **MarketModel**: Implements the fundamental market dynamics
- **StochasticProcesses**: Provides tools for simulating various stochastic processes
- **FunctionallyGeneratedPortfolio**: Implements portfolios generated by functions
- **RankBasedPortfolio**: Implements rank-based portfolio construction

### Optimization Module

- **DiversityOptimization**: Optimizes diversity parameters
- **DriftOptimization**: Maximizes portfolio drift processes
- **LongTermGrowthOptimizer**: Optimizes long-term portfolio growth rate

### Simulation Module

- **MarketSimulation**: Simulates market dynamics
- **PortfolioAnalyzer**: Analyzes portfolio performance

### Utils Module

- **DataHandler**: Data processing and preparation
- **SPTVisualizer**: Visualization tools

## Mathematical Background

### Key Concepts in Stochastic Portfolio Theory

#### Market Weights and Portfolio Weights

The market weight of a stock is defined as its capitalization relative to the total market:

$$\mu_i(t) = \frac{S_i(t)}{\sum_{j=1}^n S_j(t)}$$

A portfolio is defined by its weights $\pi_i(t)$, which specify the proportion of the portfolio invested in each stock.

#### Drift Process

The excess growth rate (drift process) of a portfolio $\pi$ relative to the market portfolio $\mu$ is:

$$\gamma(\pi, \mu) = \frac{1}{2} \sum_{i=1}^n \sigma_{ii} \left(\frac{\pi_i}{\mu_i} - 1\right)^2$$

where $\sigma_{ii}$ is the variance of stock $i$.

#### Diversity and Entropy

The $p$-diversity of the market is defined as:

$$D_p(\mu) = \left(\sum_{i=1}^n \mu_i^p\right)^{1/p}$$

The entropy of the market is:

$$H(\mu) = -\sum_{i=1}^n \mu_i \log \mu_i$$

#### Functionally Generated Portfolios

A portfolio $\pi$ can be generated from a function $G$ on the simplex:

$$\pi_i = \mu_i(1 + D_i G(\mu))$$

where $D_i$ is the partial derivative with respect to $\mu_i$.

#### Long-Term Growth Rate

The long-term growth rate of a portfolio $\pi$ is:

$$g(\pi) = g(\mu) + \gamma(\pi, \mu)$$

where $g(\mu)$ is the growth rate of the market portfolio.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install development dependencies: `pip install -e ".[dev]"`
5. Run tests: `pytest tests/`

### Coding Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Ensure mathematical correctness and numerical stability

## Citation

If you use this library in your research, please cite:

```bibtex
@software{fernholz_spt_2025,
  author = {Aheli Poddar},
  title = {Fernholz SPT: A Python Implementation of Stochastic Portfolio Theory},
  url = {https://github.com/xaheli/fernholz-spt},
  version = {0.1.0},
  year = {2025},
}
```

### References

- Fernholz, E.R. (2002). Stochastic Portfolio Theory. Springer.
- Fernholz, E.R. & Karatzas, I. (2009). Stochastic Portfolio Theory: A Survey. In: Handbook of Numerical Analysis. Elsevier.
- Fernholz, R., Karatzas, I. & Kardaras, C. (2005). Diversity and Relative Arbitrage in Equity Markets. Finance and Stochastics, 9(1), 1-27.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This implementation draws inspiration from the original theoretical work of E. Robert Fernholz and subsequent contributions to the field of Stochastic Portfolio Theory.
