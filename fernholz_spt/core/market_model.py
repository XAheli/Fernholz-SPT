import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import yfinance as yf # Added for market cap fetching

class MarketModel:
    """
    Core market model based on Fernholz's Stochastic Portfolio Theory.

    This class implements the fundamental market dynamics described in Fernholz's
    "Stochastic Portfolio Theory" (2002), including market weight calculations,
    rank-based transformations, and diversity measures.
    """

    def __init__(self,
                 stock_prices: pd.DataFrame,
                 market_caps: Optional[pd.DataFrame] = None,
                 estimate_covariance: bool = True,
                 cov_window: int = 252,
                 cov_method: str = 'standard'):
        """
        Initialize the market model with historical stock data.

        Args:
            stock_prices: DataFrame with dates as index and stock tickers as columns.
                          Prices should be cleaned (e.g., no NaNs from non-trading days,
                          forward-filled or appropriately handled).
            market_caps: Optional DataFrame of market capitalizations.
                         If None, it will attempt to fetch shares outstanding using yfinance
                         and calculate market_caps as stock_prices * shares_outstanding.
                         This yfinance-based calculation is a proxy and may have limitations.
            estimate_covariance: Whether to estimate the covariance matrix of log returns.
            cov_window: Rolling window size for covariance estimation (days).
            cov_method: Covariance estimation method ('standard', 'shrinkage', or 'exponential').
        """
        if not isinstance(stock_prices, pd.DataFrame) or not isinstance(stock_prices.index, pd.DatetimeIndex):
            raise ValueError("stock_prices must be a pandas DataFrame with a DatetimeIndex.")
        if stock_prices.empty:
            raise ValueError("stock_prices DataFrame cannot be empty.")
        if stock_prices.isnull().values.any():
            print("Warning: stock_prices contains NaN values. Consider cleaning the data (e.g., ffill, dropna).")
            # stock_prices = stock_prices.ffill().bfill() # Example: forward then backward fill

        self.stock_prices = stock_prices
        self.n_stocks = stock_prices.shape[1]
        self.stock_names = stock_prices.columns.tolist() # Ensure it's a list
        self.dates = stock_prices.index

        # Calculate log returns
        self.log_returns = np.log(self.stock_prices / self.stock_prices.shift(1)).dropna()

        # Handle market caps
        if market_caps is not None:
            if not isinstance(market_caps, pd.DataFrame) or not isinstance(market_caps.index, pd.DatetimeIndex):
                raise ValueError("market_caps must be a pandas DataFrame with a DatetimeIndex.")
            if not self.stock_prices.index.equals(market_caps.index) or \
               not self.stock_prices.columns.equals(market_caps.columns):
                print("Warning: stock_prices and market_caps indices or columns do not match. Attempting to align.")
                # Align market_caps to stock_prices
                shared_index = self.stock_prices.index.intersection(market_caps.index)
                shared_columns = self.stock_prices.columns.intersection(market_caps.columns)
                self.stock_prices = self.stock_prices.loc[shared_index, shared_columns]
                self.market_caps = market_caps.loc[shared_index, shared_columns]
                # Update attributes after alignment
                self.n_stocks = self.stock_prices.shape[1]
                self.stock_names = self.stock_prices.columns.tolist()
                self.dates = self.stock_prices.index
                self.log_returns = np.log(self.stock_prices / self.stock_prices.shift(1)).dropna()

            else:
                 self.market_caps = market_caps.copy()
        else:
            print("Market caps not provided. Attempting to calculate using yfinance shares outstanding (proxy).")
            shares_outstanding_dict = {}
            for ticker_symbol in self.stock_names:
                try:
                    ticker_obj = yf.Ticker(ticker_symbol)
                    # .info can be slow; sharesOutstanding is also in balance_sheet, financials
                    shares = ticker_obj.info.get('sharesOutstanding')
                    if shares is None: # Fallback if not in .info
                        # Try to get from financials (might be latest annual/quarterly)
                        bs = ticker_obj.balance_sheet
                        if not bs.empty and 'Ordinary Shares Number' in bs.index:
                            shares = bs.loc['Ordinary Shares Number'].iloc[0] # Take latest available
                        elif not bs.empty and 'Share Issued' in bs.index: # Another possible name
                            shares = bs.loc['Share Issued'].iloc[0]

                    if shares is not None and shares > 0:
                        shares_outstanding_dict[ticker_symbol] = shares
                    else:
                        print(f"Warning: Could not fetch valid sharesOutstanding for {ticker_symbol}. Market cap will be based on price only.")
                        shares_outstanding_dict[ticker_symbol] = 1 # Fallback to price as proxy
                except Exception as e:
                    print(f"Warning: Error fetching yfinance data for {ticker_symbol}: {e}. Market cap will be based on price only.")
                    shares_outstanding_dict[ticker_symbol] = 1 # Fallback
            
            shares_series = pd.Series(shares_outstanding_dict)
            # Align shares_series with stock_prices columns just in case some tickers failed
            aligned_shares = shares_series.reindex(self.stock_prices.columns).fillna(1)
            self.market_caps = self.stock_prices.multiply(aligned_shares, axis=1)

        # Calculate market weights and ranked weights
        self.market_weights = self._calculate_market_weights()
        self.ranked_weights = self._calculate_ranked_weights() # Also populates self.rank_map
        self.rank_crossovers = self._calculate_rank_crossovers()

        # Calculate covariance matrix if requested
        if estimate_covariance:
            self.cov_matrices = self._estimate_covariance(window=cov_window, method=cov_method)
        else:
            self.cov_matrices = {} # Use empty dict instead of None for easier checking

        # Calculate capitalization curves and diversity measures
        self.cap_curves = self._calculate_capitalization_curves()
        self.diversity = self._calculate_diversity()
        self.entropy = self._calculate_entropy()

    def _calculate_market_weights(self) -> pd.DataFrame:
        """
        Calculate market weights (capitalization proportions) for each stock.
        """
        total_market_cap = self.market_caps.sum(axis=1)
        # Avoid division by zero if total_market_cap is zero for some dates
        market_weights_df = self.market_caps.div(total_market_cap, axis=0)
        # Handle cases where total_market_cap might be 0 by setting weights to NaN or uniform
        market_weights_df[total_market_cap == 0] = np.nan # Or 1/self.n_stocks
        return market_weights_df.fillna(1.0/self.n_stocks if self.n_stocks > 0 else 0)


    def _calculate_ranked_weights(self) -> pd.DataFrame:
        """
        Calculate market weights sorted by rank (largest to smallest).
        Also populates self.rank_map.
        """
        # Ensure market_weights is available and not empty
        if self.market_weights.empty:
            # print("Warning: market_weights is empty. Cannot calculate ranked_weights.")
            self.rank_map = {}
            return pd.DataFrame(index=self.dates, columns=[f'rank_{i+1}' for i in range(self.n_stocks)])

        ranked_weights_list = []
        rank_map_dict = {}

        for date, row in self.market_weights.iterrows():
            sorted_series = row.sort_values(ascending=False)
            ranked_weights_list.append(sorted_series.values)
            rank_map_dict[date] = {stock: i + 1 for i, stock in enumerate(sorted_series.index)}

        self.rank_map = rank_map_dict
        
        # Determine the maximum number of ranks dynamically
        max_ranks = self.market_weights.shape[1]
        column_names = [f'rank_{i+1}' for i in range(max_ranks)]
        
        ranked_weights_df = pd.DataFrame(ranked_weights_list, index=self.market_weights.index, columns=column_names)
        return ranked_weights_df

    def _calculate_rank_crossovers(self) -> pd.DataFrame:
        """
        Calculate the frequency of rank changes between consecutive time periods.
        """
        if not hasattr(self, 'rank_map') or not self.rank_map:
            # print("Warning: Rank map not calculated or empty. Cannot calculate rank crossovers.")
            return pd.DataFrame(columns=['total_changes', 'change_rate'])

        dates = sorted(list(self.rank_map.keys()))
        if len(dates) < 2:
            return pd.DataFrame(index=dates[1:], columns=['total_changes', 'change_rate'])

        crossovers_data = []
        crossover_index = []

        for i in range(1, len(dates)):
            curr_date = dates[i]
            prev_date = dates[i-1]
            crossover_index.append(curr_date)

            changed_stocks = 0
            # Ensure stocks are present in both date's rank_map
            # Common stocks should be based on self.stock_names that are in both maps
            # However, rank_map is derived from market_weights, so columns should be consistent
            
            prev_ranks = self.rank_map[prev_date]
            curr_ranks = self.rank_map[curr_date]
            
            # Iterate over stocks present in the current market model
            num_comparable_stocks = 0
            for stock in self.stock_names:
                if stock in prev_ranks and stock in curr_ranks:
                    num_comparable_stocks +=1
                    if prev_ranks[stock] != curr_ranks[stock]:
                        changed_stocks += 1
            
            change_rate = changed_stocks / num_comparable_stocks if num_comparable_stocks > 0 else 0
            crossovers_data.append({'total_changes': changed_stocks, 'change_rate': change_rate})

        if not crossovers_data: # If loop didn't run
            return pd.DataFrame(columns=['total_changes', 'change_rate'])
            
        return pd.DataFrame(crossovers_data, index=pd.DatetimeIndex(crossover_index))


    def _estimate_covariance(self, window: int = 252, method: str = 'standard') -> Dict[pd.Timestamp, np.ndarray]:
        """
        Estimate the covariance matrix of log returns.
        """
        cov_matrices = {}
        # Ensure log_returns is not empty and has enough data for the window
        if self.log_returns.empty or len(self.log_returns) < window:
            # print(f"Warning: Not enough log return data (length {len(self.log_returns)}) for covariance window {window}.")
            return cov_matrices # Return empty if not enough data

        if method == 'standard':
            rolling_cov = self.log_returns.rolling(window=window).cov()
            for date_idx in range(window -1 , len(self.log_returns)): # Start from when a full window is available
                date = self.log_returns.index[date_idx]
                # .cov() result is a MultiIndex DataFrame. We need to extract the matrix for the specific date.
                # The rolling().cov() gives pairs of (date, ticker) for the outer index.
                # We need to unstack and then select.
                try:
                    # Get the covariance matrix for the window ending at 'date'
                    # The result of .cov() is already the covariance matrix, but for all assets.
                    # window_returns = self.log_returns.iloc[date_idx - window + 1 : date_idx + 1]
                    # cov_matrices[date] = window_returns.cov().values * 252 # Annualize
                    
                    # Alternative for rolling_cov:
                    # The date in rolling_cov.index corresponds to the *end* of the window.
                    # The first level of the index is the date, the second is the stock ticker.
                    # We need to select the matrix for the current 'date'.
                    # This can be tricky with the direct output of rolling().cov()
                    # A simpler way is to re-calculate cov for each window end:
                    window_data = self.log_returns.iloc[date_idx - window + 1 : date_idx + 1]
                    cov_matrices[date] = window_data.cov().values * 252.0 # Annualize

                except Exception as e:
                    # print(f"Error calculating standard covariance for date {date}: {e}")
                    pass # Or handle more robustly

        elif method == 'shrinkage':
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(assume_centered=False) # Data is log-returns, not necessarily zero mean
            for i in range(window -1, len(self.log_returns)):
                date = self.log_returns.index[i]
                window_returns = self.log_returns.iloc[i - window + 1 : i + 1].dropna(axis=1, how='all').dropna(axis=0, how='any')
                if window_returns.shape[0] < 2 or window_returns.shape[1] <1 : # Need at least 2 samples and 1 feature for LedoitWolf
                    # print(f"Skipping shrinkage for date {date} due to insufficient data after dropna: {window_returns.shape}")
                    continue
                try:
                    lw.fit(window_returns.values) # LedoitWolf expects np.ndarray
                    # Reconstruct DataFrame to match original columns for consistent output
                    shrunk_cov_df = pd.DataFrame(lw.covariance_, index=window_returns.columns, columns=window_returns.columns)
                    # Align to original stock_names order and fill missing with 0 or np.nan
                    cov_matrices[date] = shrunk_cov_df.reindex(index=self.stock_names, columns=self.stock_names).fillna(0).values * 252.0 # Annualize
                except Exception as e:
                    # print(f"Error calculating shrinkage covariance for date {date}: {e}")
                    pass


        elif method == 'exponential':
            # Using pandas.DataFrame.ewm().cov()
            # com (center of mass) = (span - 1) / 2. decay = 1 / (1 + com)
            # Standard EWMA decay factor often cited is 0.94.
            # alpha (smoothing factor) = 1 - decay = 0.06
            # span = (2 / alpha) - 1 = (2 / 0.06) - 1 approx 32.33
            # com = (span - 1) / 2 approx 15.66
            # Or, if decay = lambda, then alpha = 1 - lambda. Span = 2/(1-lambda) -1.
            span = window # Interpret window as span for EWM
            ewm_cov = self.log_returns.ewm(span=span, min_periods=window // 2).cov(pairwise=True) # Ensure we have enough periods
            
            for date_idx in range(window -1, len(self.log_returns)):
                date = self.log_returns.index[date_idx]
                try:
                    # Extract the covariance matrix for the specific date
                    # ewm_cov.loc[date] will give a DataFrame if date is unique in the first level
                    # If date is not unique, it needs careful handling.
                    # Assuming date is unique for simplicity here.
                    cov_matrix_for_date = ewm_cov.loc[pd.IndexSlice[date, :], :]
                    # This results in a DataFrame where columns are original stock names,
                    # and index is a MultiIndex (original_stock_name, stock_name_for_cov_pair)
                    # We need to unstack it.
                    unstacked_cov = cov_matrix_for_date.unstack()
                    # Now, columns are MultiIndex (level_0=original_stock_name, level_1=stock_name_for_cov_pair)
                    # We want just the stock_name_for_cov_pair as columns
                    # This should already be in the correct matrix form if indexed properly
                    cov_matrices[date] = unstacked_cov.droplevel(0, axis=1).reindex(index=self.stock_names, columns=self.stock_names).fillna(0).values * 252.0
                except KeyError:
                     # print(f"Date {date} not found in EWM covariance index. This can happen if min_periods not met.")
                     # Fallback to calculating for the specific window end if direct loc fails
                    window_data = self.log_returns.iloc[:date_idx + 1] # Use all data up to current date for EWM
                    if len(window_data) >= window //2 :
                        temp_ewm_cov = window_data.ewm(span=span, min_periods=window // 2).cov(pairwise=True)
                        if not temp_ewm_cov.empty and date in temp_ewm_cov.index.get_level_values(0):
                             cov_matrix_for_date = temp_ewm_cov.loc[pd.IndexSlice[date, :], :]
                             unstacked_cov = cov_matrix_for_date.unstack()
                             cov_matrices[date] = unstacked_cov.droplevel(0, axis=1).reindex(index=self.stock_names, columns=self.stock_names).fillna(0).values * 252.0
                except Exception as e:
                    # print(f"Error calculating exponential covariance for date {date}: {e}")
                    pass

        else:
            raise ValueError(f"Unknown covariance method: {method}")

        return cov_matrices


    def _calculate_capitalization_curves(self) -> pd.DataFrame:
        """
        Calculate the capitalization curves for each rank.
        """
        if self.ranked_weights.empty:
            # print("Warning: ranked_weights is empty. Cannot calculate capitalization curves.")
            return pd.DataFrame(index=self.dates)
        # ranked_weights already has columns rank_1, rank_2, ...
        # We just need to rename them for clarity if desired, or use as is.
        cap_curves = self.ranked_weights.copy()
        cap_curves.columns = [f'cap_{col.split("_")[1]}' for col in self.ranked_weights.columns if "rank_" in col]
        return cap_curves


    def _calculate_diversity(self, p_values: List[float] = [0.5, 0.75, 0.9]) -> pd.DataFrame:
        """
        Calculate market diversity using the p-diversity measure.
        D_p(μ) = (Σ_i μ_i^p)^(1/p)
        """
        diversity_df = pd.DataFrame(index=self.market_weights.index)
        if self.market_weights.empty:
            return diversity_df

        for p in p_values:
            if not (0 < p < 1): # p must be strictly between 0 and 1
                print(f"Warning: p-value {p} for diversity is not in (0,1). Skipping.")
                continue

            col_name = f'diversity_{p}'
            # Apply power and sum row-wise, then apply power 1/p
            # (μ_i^p)
            mu_p = self.market_weights.pow(p, axis=1)
            # Σ_i μ_i^p
            sum_mu_p = mu_p.sum(axis=1)
            # (Σ_i μ_i^p)^(1/p)
            diversity_df[col_name] = sum_mu_p.pow(1/p)

        return diversity_df


    def _calculate_entropy(self) -> pd.Series:
        """
        Calculate the entropy of market weights.
        H(μ) = -Σ_i μ_i * log(μ_i)
        """
        if self.market_weights.empty:
            return pd.Series(index=self.dates, dtype=float)

        # Ensure μ_i > 0 for log
        # μ_i * log(μ_i)
        # Weights very close to zero can cause -0.0 * inf = NaN. Use a small epsilon.
        epsilon = 1e-18
        term = self.market_weights * np.log(self.market_weights.clip(lower=epsilon))
        # Sum term row-wise and negate
        entropy_series = -term.sum(axis=1)
        return entropy_series


    def get_rank_at_date(self, date: pd.Timestamp) -> Dict[str, int]:
        """
        Get the rank of each stock at a specific date.
        """
        pd_date = pd.Timestamp(date)
        if pd_date not in self.rank_map:
            raise ValueError(f"No rank information for date {pd_date}. Available dates: {list(self.rank_map.keys())[:5]}...")
        return self.rank_map[pd_date]

    def get_stocks_at_rank(self, date: pd.Timestamp, ranks: Union[int, List[int]]) -> List[str]:
        """
        Get the stocks at specific ranks on a given date.
        """
        pd_date = pd.Timestamp(date)
        if pd_date not in self.rank_map:
            raise ValueError(f"No rank information for date {pd_date}")

        if isinstance(ranks, int):
            ranks = [ranks]

        # Invert the rank map for this date: {rank_num: stock_name}
        inv_rank_map = {v: k for k, v in self.rank_map[pd_date].items()}

        result = []
        for rank in ranks:
            if rank in inv_rank_map:
                result.append(inv_rank_map[rank])
            # else:
                # print(f"Warning: Rank {rank} not found for date {pd_date}.")
        return result

    def calculate_concentration_ratio(self, date: pd.Timestamp, top_n: int) -> float:
        """
        Calculate the concentration ratio of the top n stocks at a specific date.
        CR_n = Σ_{i=1}^n μ_(i)
        """
        pd_date = pd.Timestamp(date)
        if top_n <= 0 or top_n > self.n_stocks:
            raise ValueError(f"top_n must be > 0 and <= number of stocks ({self.n_stocks})")
        if pd_date not in self.ranked_weights.index:
            raise ValueError(f"Date {pd_date} not in ranked_weights index.")

        # ranked_weights has columns 'rank_1', 'rank_2', ...
        # Sum the weights of the top_n ranks
        # Ensure column names exist, e.g. if top_n > actual available ranks (should not happen if n_stocks is source of truth)
        cols_to_sum = [f'rank_{i+1}' for i in range(min(top_n, self.ranked_weights.shape[1]))]
        
        # Check if all required columns exist
        missing_cols = [col for col in cols_to_sum if col not in self.ranked_weights.columns]
        if missing_cols:
            # print(f"Warning: Missing rank columns for concentration ratio: {missing_cols}. Result may be partial.")
            cols_to_sum = [col for col in cols_to_sum if col in self.ranked_weights.columns]
            if not cols_to_sum: return 0.0

        return self.ranked_weights.loc[pd_date, cols_to_sum].sum()

    def calculate_herfindahl_index(self, date: pd.Timestamp) -> float:
        """
        Calculate the Herfindahl-Hirschman Index (HHI) at a specific date.
        HHI = Σ_i μ_i^2
        """
        pd_date = pd.Timestamp(date)
        if pd_date not in self.market_weights.index:
            raise ValueError(f"Date {pd_date} not in market_weights index.")

        weights = self.market_weights.loc[pd_date].values
        return np.sum(weights**2)

    def get_market_portfolio(self) -> pd.DataFrame: # Renamed for clarity
        """
        Return the market portfolio (i.e., market weights).
        """
        return self.market_weights.copy()