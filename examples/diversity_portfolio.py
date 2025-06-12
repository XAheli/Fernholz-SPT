import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf # Import yfinance

from fernholz_spt.core.market_model import MarketModel
from fernholz_spt.core.portfolio_generation import FunctionallyGeneratedPortfolio
from fernholz_spt.simulation.performance_analysis import PortfolioAnalyzer
from fernholz_spt.utils.visualization import SPTVisualizer

def run_diversity_portfolio_example():
    print("Running Diversity Portfolio Example with yfinance data...")

    # Define tickers and date range
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ', 'PG'] # Example: Tech + Finance + Consumer
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}")

    # Fetch price data using yfinance
    price_data_list = []
    for ticker_symbol in tickers:
        try:
            data = yf.Ticker(ticker_symbol).history(start=start_date, end=end_date, auto_adjust=True) # auto_adjust=True for simplicity
            if not data.empty and 'Close' in data.columns:
                price_data_list.append(data['Close'].rename(ticker_symbol))
            else:
                print(f"No 'Close' data for {ticker_symbol}, skipping.")
        except Exception as e:
            print(f"Could not fetch data for {ticker_symbol}: {e}")
    
    if not price_data_list:
        print("No price data fetched. Exiting example.")
        return

    prices_df = pd.concat(price_data_list, axis=1)
    
    # Handle potential missing data after concat (e.g., some stocks IPO'd later)
    # For this example, we'll keep only common trading days and ffill initial NaNs for stocks that listed later
    prices_df = prices_df.dropna(axis=0, how='all') # Drop days where no stock has data
    prices_df = prices_df.ffill().bfill() # Fill remaining NaNs (e.g. from staggered IPOs or missing days)
    
    # Filter for dates where all selected stocks have price data to avoid issues with market cap calculation
    # or ensure MarketModel handles it gracefully.
    # For simplicity, let's take a period where all are likely trading.
    # More robust handling would involve dynamic universe or careful NaN treatment in MarketModel.
    prices_df = prices_df.dropna(axis=0, how='any') 


    if prices_df.empty or prices_df.shape[1] < 2: # Need at least 2 stocks
        print("Not enough valid price data after cleaning. Exiting example.")
        return
    print(f"Price data shape after yfinance and cleaning: {prices_df.shape}")
    print("Price data tail:\n", prices_df.tail())


    # Initialize market model
    # market_caps=None will trigger yfinance fetching for sharesOutstanding proxy
    try:
        market_model = MarketModel(stock_prices=prices_df, market_caps=None, estimate_covariance=True, cov_window=63) # Shorter cov window
    except Exception as e:
        print(f"Error initializing MarketModel: {e}")
        return

    # Create diversity-weighted portfolio
    fg_gen = FunctionallyGeneratedPortfolio(market_model)
    diversity_param_p = 0.5
    div_weights_df = fg_gen.diversity_weighted(p=diversity_param_p)
    print(f"\nDiversity-weighted portfolio (p={diversity_param_p}) weights (tail):\n", div_weights_df.tail())

    # Analyze performance
    analyzer = PortfolioAnalyzer(market_model)
    
    # Calculate excess growth
    # Ensure weights align with market_model.cov_matrices dates
    common_dates_for_drift = div_weights_df.index.intersection(pd.Index(market_model.cov_matrices.keys()))
    if common_dates_for_drift.empty:
        print("No common dates for drift calculation. Covariance matrices might not be available for the portfolio's date range.")
        drift_series = pd.Series(dtype=float)
    else:
        drift_series = analyzer.calculate_excess_growth(div_weights_df.loc[common_dates_for_drift])
    
    print("\nExcess Growth Series (tail):\n", drift_series.tail())

    # Visualize
    fig1 = SPTVisualizer.plot_weight_evolution(div_weights_df.dropna(how='all'), top_n=5, title=f"Diversity Portfolio (p={diversity_param_p}) Weights")
    plt.show(block=False) # Show plots non-blockingly for scripts

    fig2 = SPTVisualizer.plot_drift_analysis(drift_series.dropna(), title=f"Diversity Portfolio (p={diversity_param_p}) Excess Growth")
    plt.show(block=False)

    # Example: Compare with market portfolio
    market_portfolio_weights = market_model.get_market_portfolio()
    portfolio_dict_for_comparison = {
        f"Diversity (p={diversity_param_p})": div_weights_df,
        "Market Portfolio": market_portfolio_weights
    }
    
    fig3 = SPTVisualizer.plot_portfolio_comparison(
        market_model,
        portfolio_dict_for_comparison,
        metric_to_plot='Cumulative Relative Return' # Or 'Excess Growth', 'Entropy'
    )
    plt.suptitle("Portfolio Comparison: Cumulative Relative Return vs Market")
    plt.show() # Last plot can be blocking

if __name__ == "__main__":
    run_diversity_portfolio_example()