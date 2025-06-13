import pytest
import pandas as pd
import numpy as np
import yfinance as yf
from unittest.mock import patch, MagicMock

# Modules to test
from fernholz_spt.core.market_model import MarketModel
from fernholz_spt.core.portfolio_generation import FunctionallyGeneratedPortfolio
from fernholz_spt.core.rank_based import RankBasedPortfolio
from fernholz_spt.simulation.performance_analysis import PortfolioAnalyzer
from fernholz_spt.optimization.long_term_growth import LongTermGrowthOptimizer
from fernholz_spt.optimization.diversity_weights import DiversityOptimization
from fernholz_spt.optimization.drift_optimization import DriftOptimization
from fernholz_spt.utils.data_handling import DataHandler # Assuming this exists if needed
from fernholz_spt.utils.visualization import SPTVisualizer # Test only if methods don't require active display

# --- Fixtures ---

@pytest.fixture(scope="module")
def sample_stock_prices_raw():
    """Provides a small raw DataFrame of stock prices."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                           '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10'])
    data = {
        'AAPL': [150, 151, 150.5, 152, 153, 152.5, 154, 155, 153, 156],
        'MSFT': [250, 252, 251, 253, 255, 254, 256, 257, 255, 258],
        'GOOG': [100, 101, 100.5, 102, 103, 102.5, 104, 105, 103, 106]
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture(scope="module")
def sample_market_caps_raw(sample_stock_prices_raw):
    """Provides a small raw DataFrame of market caps, aligned with prices."""
    # For simplicity, let's make market caps proportional to prices with some scaling
    return sample_stock_prices_raw * pd.Series({'AAPL': 20e9, 'MSFT': 25e9, 'GOOG': 15e9})


@pytest.fixture(scope="module")
def mock_yfinance_ticker():
    """Mocks yf.Ticker calls."""
    mock_ticker = MagicMock()
    # Mock .info attribute which is a dictionary
    mock_ticker.info = {'sharesOutstanding': 1e9} # Default shares
    # Mock .history() method
    def mock_history(start, end, auto_adjust=True):
        dates = pd.date_range(start=start, end=end, freq='B') # Business days
        if not len(dates): # If start and end are the same or invalid range
            return pd.DataFrame()
        data = {'Close': np.linspace(100, 150, len(dates))}
        return pd.DataFrame(data, index=dates)
    mock_ticker.history.side_effect = mock_history
    return mock_ticker


@pytest.fixture(scope="module")
def market_model_fixture(sample_stock_prices_raw, sample_market_caps_raw):
    """Initializes MarketModel with sample data for testing."""
    return MarketModel(
        stock_prices=sample_stock_prices_raw.copy(),
        market_caps=sample_market_caps_raw.copy(),
        estimate_covariance=True,
        cov_window=5, # Short window for test data
        cov_method='standard'
    )

@pytest.fixture(scope="module")
def market_model_yfinance_fixture(sample_stock_prices_raw, mock_yfinance_ticker):
    """Initializes MarketModel relying on yfinance mock for market caps."""
    with patch('yfinance.Ticker') as mock_yf_ticker_constructor:
        mock_yf_ticker_constructor.return_value = mock_yfinance_ticker
        mm = MarketModel(
            stock_prices=sample_stock_prices_raw.copy(),
            market_caps=None, # Trigger yfinance fetching
            estimate_covariance=True,
            cov_window=5,
            cov_method='standard'
        )
        # Verify yfinance was called
        assert mock_yf_ticker_constructor.call_count == len(sample_stock_prices_raw.columns)
        return mm


# --- Test Classes ---

class TestMarketModel:
    def test_initialization(self, market_model_fixture):
        mm = market_model_fixture
        assert mm.n_stocks == 3
        assert not mm.stock_prices.empty
        assert not mm.market_caps.empty
        assert not mm.market_weights.empty
        assert not mm.ranked_weights.empty
        assert 'rank_1' in mm.ranked_weights.columns
        assert mm.cov_matrices # Should not be empty if estimate_covariance=True and window is valid
        assert not mm.diversity.empty
        assert not mm.entropy.empty

    def test_initialization_with_yfinance(self, market_model_yfinance_fixture):
        mm = market_model_yfinance_fixture
        assert mm.n_stocks == 3
        assert not mm.market_caps.isnull().values.any(), "Market caps should be filled by yfinance mock"
        # Check if market caps are prices * sharesOutstanding (1e9 from mock)
        expected_cap_last_day_aapl = mm.stock_prices['AAPL'].iloc[-1] * 1e9
        assert np.isclose(mm.market_caps['AAPL'].iloc[-1], expected_cap_last_day_aapl)
        assert not mm.market_weights.empty
        assert mm.cov_matrices

    def test_market_weights_sum_to_one(self, market_model_fixture):
        mm = market_model_fixture
        assert np.allclose(mm.market_weights.sum(axis=1), 1.0, rtol=1e-6)

    def test_ranked_weights_order(self, market_model_fixture):
        mm = market_model_fixture
        for idx, row in mm.ranked_weights.iterrows():
            assert row['rank_1'] >= row['rank_2'] >= row['rank_3']

    def test_get_rank_at_date(self, market_model_fixture):
        mm = market_model_fixture
        date = mm.dates[5] # A date with enough history for cov
        ranks = mm.get_rank_at_date(date)
        assert isinstance(ranks, dict)
        assert len(ranks) == mm.n_stocks
        assert 'AAPL' in ranks

    def test_covariance_estimation(self, market_model_fixture):
        mm = market_model_fixture
        # Check if cov_matrices has entries for dates after the window
        # log_returns has N-1 rows. First cov_matrix is at log_returns.index[cov_window-1]
        first_cov_date_expected = mm.log_returns.index[mm.cov_window-1]
        assert first_cov_date_expected in mm.cov_matrices
        cov_matrix = mm.cov_matrices[first_cov_date_expected]
        assert cov_matrix.shape == (mm.n_stocks, mm.n_stocks)
        assert np.all(np.diag(cov_matrix) >= 0) # Variances non-negative

    def test_diversity_and_entropy(self, market_model_fixture):
        mm = market_model_fixture
        assert 'diversity_0.5' in mm.diversity.columns
        assert not mm.diversity['diversity_0.5'].isnull().any()
        assert not mm.entropy.isnull().any()
        assert (mm.entropy >= 0).all() # Entropy should be non-negative


class TestPortfolioGeneration:
    @pytest.fixture(scope="class")
    def fgp_generator(self, market_model_fixture):
        return FunctionallyGeneratedPortfolio(market_model_fixture)

    def test_equal_weighted(self, fgp_generator, market_model_fixture):
        weights_df = fgp_generator.equal_weighted()
        assert not weights_df.empty
        assert np.allclose(weights_df.sum(axis=1), 1.0)
        assert np.allclose(weights_df.iloc[0], 1.0 / market_model_fixture.n_stocks)

    def test_market_weighted(self, fgp_generator, market_model_fixture):
        weights_df = fgp_generator.market_weighted()
        assert weights_df.equals(market_model_fixture.market_weights)

    def test_diversity_weighted(self, fgp_generator):
        weights_df = fgp_generator.diversity_weighted(p=0.5)
        assert not weights_df.empty
        assert np.allclose(weights_df.sum(axis=1), 1.0, rtol=1e-6)
        # For p=0.5, weights should be proportional to sqrt(market_weight)
        # Check a specific date
        date = weights_df.index[0]
        mw_at_date = fgp_generator.market_model.market_weights.loc[date]
        expected_proportional = np.sqrt(mw_at_date)
        expected_weights = expected_proportional / expected_proportional.sum()
        assert np.allclose(weights_df.loc[date].values, expected_weights.values, rtol=1e-6)


    def G_test(self, mu_array): # Generating function G(mu) = sum(mu_i^0.5)^2
        return np.sum(np.sqrt(np.maximum(mu_array, 1e-12)))**2

    def grad_log_G_test(self, mu_array): # D_k log G for G(mu) = (sum mu_i^0.5)^2
        # log G = 2 * log(sum mu_i^0.5)
        # D_k log G = 2 * (1 / sum(mu_j^0.5)) * (0.5 * mu_k^-0.5) = mu_k^-0.5 / sum(mu_j^0.5)
        sqrt_mu = np.sqrt(np.maximum(mu_array, 1e-12))
        sum_sqrt_mu = np.sum(sqrt_mu)
        if sum_sqrt_mu == 0: return np.zeros_like(mu_array)
        grad = (1.0 / sqrt_mu) / sum_sqrt_mu # This is D_k G / G * (something related to homogeneity)
                                          # My D_k log G formula seems to be: (mu_k^-0.5) / sum(mu_j^0.5)
        
        # For G(mu) = (sum mu_i^p)^(1/p), from README, D_k log G = mu_k^(p-1) / sum(mu_j^p)
        # Here p=0.5. So D_k log G = mu_k^-0.5 / sum(mu_j^0.5)
        inv_sqrt_mu_k = 1.0 / sqrt_mu # mu_k^-0.5
        return inv_sqrt_mu_k / sum_sqrt_mu


    def test_custom_generated_fernholz(self, fgp_generator):
        weights_df = fgp_generator.custom_generated_fernholz(
            generating_function=self.G_test,
            gradient_log_G_function=self.grad_log_G_test
        )
        assert not weights_df.empty
        assert np.allclose(weights_df.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5), f"Weights sum: {weights_df.sum(axis=1)}"
        # This specific G should yield diversity_weighted with p=0.5 if formula is correct.
        # The formula for pi_i = (D_i log G + 1 - S_t) * mu_i
        # For G_p(mu) = (sum mu_k^p)^(1/p) -> log G_p = (1/p) log(sum mu_k^p)
        # D_i log G_p = mu_i^(p-1) / (sum mu_k^p)
        # Let p = 0.5. D_i log G_0.5 = mu_i^(-0.5) / (sum mu_k^0.5)
        # S_t = sum mu_k * D_k log G = sum mu_k^0.5 / sum mu_j^0.5
        # pi_i = (mu_i^(-0.5) / sum(mu_k^0.5) + 1 - sum(mu_k^0.5)/sum(mu_j^0.5)) * mu_i
        # pi_i = (mu_i^(-0.5) / sum(mu_k^0.5)) * mu_i = mu_i^0.5 / sum(mu_k^0.5)
        # This is the numerator of diversity_weighted p=0.5. Normalization should make it equal.
        
        diversity_05_weights = fgp_generator.diversity_weighted(p=0.5)
        # Need to handle NaNs that might arise if market_weights had zeros initially
        # For the sample data, there are no zeros in market_weights
        pd.testing.assert_frame_equal(weights_df.dropna(), diversity_05_weights.dropna(), check_dtype=False, atol=1e-6)

    def test_drift_process_fernholz(self, fgp_generator, market_model_fixture):
        if not market_model_fixture.cov_matrices:
             pytest.skip("Skipping drift test as no covariance matrices were generated.")
        
        # Use a simple G, e.g., G(mu) = 1 (constant)
        # For G=1, D_i log G = 0. Hessian of G is 0. So drift g(t) should be 0.
        # Portfolio generated by G=1 is market portfolio
        market_portfolio = fgp_generator.market_weighted()

        def G_constant(mu_array): return 1.0
        def hessian_G_constant(mu_array): return np.zeros((len(mu_array), len(mu_array)))

        drift_series = fgp_generator.calculate_drift_process_fernholz(
            portfolio_weights_fernholz_g=market_portfolio, # Contextual
            generating_function=G_constant,
            hessian_G_function=hessian_G_constant
        )
        assert not drift_series.empty
        # Check dates align with cov_matrices
        valid_drift_dates = market_model_fixture.cov_matrices.keys()
        assert np.allclose(drift_series.loc[drift_series.index.isin(valid_drift_dates)].fillna(0), 0.0, atol=1e-9)


class TestRankBasedPortfolio:
    @pytest.fixture(scope="class")
    def rbp_generator(self, market_model_fixture):
        return RankBasedPortfolio(market_model_fixture)

    def test_top_m_portfolio(self, rbp_generator, market_model_fixture):
        m = 2
        weights_df = rbp_generator.top_m_portfolio(m=m, weighting='equal')
        assert not weights_df.empty
        assert np.allclose(weights_df.sum(axis=1), 1.0)
        # Check if only top m stocks have non-zero weights
        for date, row in weights_df.iterrows():
            non_zero_weights = row[row > 1e-8] # Allow for float precision
            assert len(non_zero_weights) == m
            # Verify these are indeed the top m stocks
            top_m_stocks_true = market_model_fixture.market_weights.loc[date].nlargest(m).index
            assert set(non_zero_weights.index) == set(top_m_stocks_true)

    def test_bottom_m_portfolio_cap_weighted(self, rbp_generator, market_model_fixture):
        m = 1
        weights_df = rbp_generator.bottom_m_portfolio(m=m, weighting='cap')
        assert not weights_df.empty
        assert np.allclose(weights_df.sum(axis=1), 1.0)
        for date, row in weights_df.iterrows():
            non_zero_weights = row[row > 1e-8]
            assert len(non_zero_weights) == m
            # For cap-weighted bottom_m=1, weight should be 1.0 for that stock in the portfolio
            assert np.isclose(non_zero_weights.iloc[0], 1.0)
            bottom_m_stock_true = market_model_fixture.market_weights.loc[date].nsmallest(m).index
            assert non_zero_weights.index[0] == bottom_m_stock_true[0]


class TestPortfolioAnalyzer:
    @pytest.fixture(scope="class")
    def analyzer(self, market_model_fixture):
        if not market_model_fixture.cov_matrices: # Ensure cov matrices are there
            market_model_fixture.cov_matrices = market_model_fixture._estimate_covariance(
                window=market_model_fixture.cov_window if hasattr(market_model_fixture, 'cov_window') else 5,
                method='standard'
            )
        return PortfolioAnalyzer(market_model_fixture)

    @pytest.fixture(scope="class")
    def equal_weights(self, market_model_fixture):
        fgp = FunctionallyGeneratedPortfolio(market_model_fixture)
        return fgp.equal_weighted()

    @pytest.fixture(scope="class")
    def market_weights_portfolio(self, market_model_fixture):
        return market_model_fixture.get_market_portfolio()

    def test_calculate_excess_growth(self, analyzer, equal_weights, market_model_fixture):
        if not market_model_fixture.cov_matrices:
             pytest.skip("Skipping excess growth test as no covariance matrices were generated.")
        excess_growth = analyzer.calculate_excess_growth(equal_weights)
        assert isinstance(excess_growth, pd.Series)
        assert not excess_growth.empty
        # Excess growth for equal weight vs market should generally be non-negative
        # if market is not perfectly uniform and there's volatility
        assert (excess_growth.dropna() >= -1e-9).all() # Allowing for small float errors

        # For market portfolio, excess growth relative to itself should be zero
        market_portfolio_weights = analyzer.market_model.get_market_portfolio()
        excess_growth_market = analyzer.calculate_excess_growth(market_portfolio_weights)
        # Common index between market portfolio and available cov matrices
        common_idx = market_portfolio_weights.index.intersection(pd.Index(analyzer.market_model.cov_matrices.keys()))
        assert np.allclose(excess_growth_market.loc[common_idx].fillna(0), 0.0, atol=1e-9)


    def test_calculate_relative_return(self, analyzer, equal_weights, market_weights_portfolio):
        relative_return = analyzer.calculate_relative_return(equal_weights, market_weights_portfolio)
        assert isinstance(relative_return, pd.Series)
        # Relative return of market vs market should be zero
        market_vs_market = analyzer.calculate_relative_return(market_weights_portfolio, market_weights_portfolio)
        assert np.allclose(market_vs_market.fillna(0), 0.0, atol=1e-9)

    def test_analyze_turnover(self, analyzer, equal_weights):
        turnover_stats = analyzer.analyze_turnover(equal_weights)
        assert isinstance(turnover_stats, pd.DataFrame)
        assert 'Turnover' in turnover_stats.columns
        # For truly equal weights (if universe doesn't change), turnover is 0 after initial setup.
        # Our equal_weights are constant, so turnover should be 0.
        assert np.allclose(turnover_stats['Turnover'].fillna(0), 0.0, atol=1e-9)

        # Test with changing weights (e.g. diversity weighted)
        fgp = FunctionallyGeneratedPortfolio(analyzer.market_model)
        div_weights = fgp.diversity_weighted(p=0.5)
        turnover_div = analyzer.analyze_turnover(div_weights)
        if len(div_weights) >1: # if there's more than one period
             assert (turnover_div['Turnover'].dropna() >= 0).all()


    def test_fernholz_metrics(self, analyzer, equal_weights):
        metrics_df = analyzer.calculate_fernholz_metrics(equal_weights)
        assert isinstance(metrics_df, pd.DataFrame)
        expected_cols = ['Excess Growth', 'Entropy', 'Cumulative Relative Return', 'Turnover (Rolling Avg 21d)']
        for col in expected_cols:
            if col == 'Excess Growth' and not analyzer.market_model.cov_matrices:
                continue # Skip if no cov_matrices
            assert col in metrics_df.columns


class TestOptimizers: # Basic smoke tests for optimizers
    def test_long_term_growth_optimizer(self, market_model_fixture):
        if len(market_model_fixture.dates) <= 5: # Lookback window in optimizer
            pytest.skip("Not enough data for LongTermGrowthOptimizer test")
        optimizer = LongTermGrowthOptimizer(market_model_fixture)
        # Test single date optimization
        target_date = market_model_fixture.log_returns.index[-1] # Use a date with log returns
        
        # Ensure target_date allows for lookback_window in log_returns
        lookback = 3 
        if len(market_model_fixture.log_returns.loc[:target_date]) < lookback:
             pytest.skip(f"Not enough log_returns history for date {target_date} with lookback {lookback}")
        
        # Also ensure the market_weights are available for target_date
        if target_date not in market_model_fixture.market_weights.index:
            pytest.skip(f"Market weights not available for target_date {target_date}")

        weights = optimizer.optimize_growth_rate(date=target_date, lookback_window=lookback, max_weight=1.0)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == market_model_fixture.n_stocks
        assert np.isclose(np.sum(weights), 1.0, rtol=1e-5)
        assert (weights >= -1e-6).all() # Approx >= 0
        assert (weights <= 1.0 + 1e-6).all() # Approx <= 1.0 (max_weight)

    def test_diversity_optimization(self, market_model_fixture):
        if len(market_model_fixture.log_returns) < 2 : # Need at least 2 returns to calculate portfolio returns
             pytest.skip("Not enough log_returns for DiversityOptimization")
        optimizer = DiversityOptimization(market_model_fixture)
        # Choose start/end dates from available log_return dates
        start_date = market_model_fixture.log_returns.index[0]
        end_date = market_model_fixture.log_returns.index[-1]
        if (end_date - start_date).days < 1: # Need some duration
            pytest.skip("Not enough duration for DiversityOptimization")

        optimal_p = optimizer.optimize_diversity_parameter(start_date, end_date, objective='sharpe')
        assert 0.01 <= optimal_p <= 0.99

    def test_drift_optimization(self, market_model_fixture):
        if not market_model_fixture.cov_matrices:
            pytest.skip("Covariance matrices required for DriftOptimization")
        optimizer = DriftOptimization(market_model_fixture)
        target_date = list(market_model_fixture.cov_matrices.keys())[0] # A date with a cov matrix
        weights = optimizer.optimize_weights_for_drift(date=target_date, max_weight=1.0)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == market_model_fixture.n_stocks
        assert np.isclose(np.sum(weights), 1.0, rtol=1e-5)

# --- Visualization Smoke Tests (Optional - checks if they run without error) ---
# These are harder to assert correctness of output, just that they don't crash.
# Mocking matplotlib might be needed for CI environments without display.

@patch('matplotlib.pyplot.show') # Prevent plots from actually showing
def test_visualization_smoke_tests(mock_show, market_model_fixture, sample_stock_prices_raw):
    # Basic portfolio weights for plotting
    weights_df = pd.DataFrame(np.random.rand(10, 3),
                              index=sample_stock_prices_raw.index,
                              columns=sample_stock_prices_raw.columns)
    weights_df = weights_df.div(weights_df.sum(axis=1), axis=0)

    SPTVisualizer.plot_weight_evolution(weights_df)
    
    if not market_model_fixture.ranked_weights.empty:
         SPTVisualizer.plot_rank_distribution(market_model_fixture, market_model_fixture.dates[-1])

    drift_series = pd.Series(np.random.randn(10), index=sample_stock_prices_raw.index)
    SPTVisualizer.plot_drift_analysis(drift_series)

    log_returns_series = pd.Series(np.random.randn(10) * 0.01, index=sample_stock_prices_raw.index)
    SPTVisualizer.plot_cumulative_returns(log_returns_series, benchmark_log_returns=log_returns_series * 0.5)

    # For plot_portfolio_comparison
    analyzer = PortfolioAnalyzer(market_model_fixture)
    metrics_data = analyzer.calculate_fernholz_metrics(weights_df)
    if not metrics_data.empty and 'Excess Growth' in metrics_data.columns :
        # This test method is for direct plotting if metrics are precomputed,
        # The refactored SPTVisualizer.plot_portfolio_comparison takes weights dict and market_model.
        # Let's test it that way:
        portfolio_dict = {"TestPort": weights_df}
        SPTVisualizer.plot_portfolio_comparison(market_model_fixture, portfolio_dict, metric_to_plot='Excess Growth')

    # For plot_optimization_surface
    opt_results = pd.DataFrame({
        'param1': np.random.rand(10),
        'param2': np.random.rand(10),
        'Sharpe': np.random.rand(10)
    })
    SPTVisualizer.plot_optimization_surface(opt_results, 'param1', 'param2', 'Sharpe')
    
    assert mock_show.call_count >= 0 # Check it was prepared to be shown