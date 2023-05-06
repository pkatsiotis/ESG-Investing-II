import pandas as pd
import numpy as np
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import cvxpy as cp
import datetime
import urllib.request
import json
import yfinance as yf
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.integrate import quad
import matplotlib.pyplot as plt
import joypy
import seaborn as sns
from scipy.stats import mannwhitneyu

class stock_universe():
    '''
    A class used to create an investing universe of stocks

    Attributes:
    tickers (list): A list of strings that are tickers of stocks
    year_start (str): Starting year of historical data to be kept
    year_end (str): Ending year of historical data to be kept
    esg_df (DataFrame): A DataFrame that contains the historial total ESG scores of the given tickers
    e_df (DataFrame): A DataFrame that contains the historial Environmental (E) scores of the given tickers
    s_df (DataFrame): A DataFrame that contains the historial Social (S) scores of the given tickers
    g_df (DataFrame): A DataFrame that contains the historial Governance (G) scores of the given tickers
    stocks (DataFrame): A DataFrame that contains the historical stock price data of the given tickers
    stocks_dict_in (dictionary): Dictionary that contains the in-sample stock prices of the different investing strategies
        for the optimization
    stocks_dict_out (dictionary): Dictionary that contains the out-of-sample stock prices of the different investing strategies

    Methods:
    get_score_data()
        Get the historical score data of the tickers given from yahoo finance
    get_stock_data()
        Get historical stock price data of the tickers given
    create_universe()
        Create universe of stocks, includes stock price, score score, e score, s score & g score data
    screening()
        Function for screening stocks based on their score, eliminate bottom 30% quantile
    '''
    
    def __init__(self, tickers, year_start, year_end, year_insample):
        '''
        Parameters:
        tickers (list): A list of strings that are tickers of stocks
        year_start (str): Starting year of historical data to be kept
        year_end (str): Ending year of historical data to be kept
        year_insample (str): Ending year to use as insample data, year_start to year_insample will be the training period
        '''
        self.tickers = tickers
        self.year_start = year_start
        self.year_end = year_end
        self.year_insample = year_insample

    def get_score_data(self):
        '''
        Get the historical score data of the tickers given from yahoo finance
        
        Returns:
        esg_df (DataFrame): A DataFrame that contains the historial total score scores of the given tickers
        e_df (DataFrame): A DataFrame that contains the historial Environmental (E) scores of the given tickers
        s_df (DataFrame): A DataFrame that contains the historial Social (S) scores of the given tickers
        g_df (DataFrame): A DataFrame that contains the historial Governance (G) scores of the given tickers
        '''

        # Lists for saving the historical score scores for each ticker, and missing list in case data are missing for a ticker
        esg_list = []
        e_list = []
        s_list = []
        g_list = []
        missing = []

        for ticker in self.tickers:
            # Try fetching score data from yahoo finance
            try:
                url = "https://query2.finance.yahoo.com/v1/finance/esgChart?symbol={}".format(ticker)

                with urllib.request.urlopen(url) as connection:

                    data_connection = connection.read()
                    data_json = json.loads(data_connection)
                    formatdata = data_json["esgChart"]["result"][0]["symbolSeries"]
                    df_data = pd.DataFrame(formatdata)
                    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"], unit="s")
                    df_data = df_data.set_index('timestamp')
                    # Filter to keep only historical data for the years given
                    df_data = df_data.loc[self.year_start:self.year_end]
                    esg_list.append(df_data['esgScore'])
                    e_list.append(df_data['environmentScore'])
                    s_list.append(df_data['socialScore'])
                    g_list.append(df_data['governanceScore'])
            except:
                missing.append(ticker)
                continue

        # If ticker is missing data remove ticker from investment sample universe
        if missing != None:
            for tick in missing:
                self.tickers.remove(tick)

        esg_df = pd.concat(esg_list, axis=1)
        esg_df.columns = self.tickers

        e_df = pd.concat(e_list, axis=1)
        e_df.columns = self.tickers

        s_df = pd.concat(s_list, axis=1)
        s_df.columns = self.tickers

        g_df = pd.concat(g_list, axis=1)
        g_df.columns = self.tickers

        return esg_df, e_df, s_df, g_df

    def get_stock_data(self):
        '''
        Get historical stock price data of the tickers given

        Returns:
        stocks (DataFrame): A DataFrame that contains the historical stock price data of the given tickers
        '''

        year_start = int(self.year_start)
        year_end = int(self.year_end)

        # First calendar day of historical data
        start = datetime.date(year_start,1,1)
        # Last calendar day of historical data
        end = datetime.date(year_end,12,31)

        stocks_list = []

        for tick in self.tickers:
            get_tick = yf.Ticker(tick)
            stocks_list.append(get_tick.history(start=start, end=end)[['Close']].add_suffix('_'+tick))

        stocks = pd.concat(stocks_list, axis = 1)
        stocks = stocks.loc[self.year_start:]
        non_missing_ticks = stocks.columns[~stocks.isnull().any()].tolist()
        stocks = stocks[non_missing_ticks]
        clean_ticks = list(map(lambda x: x.split('_')[1], stocks.columns))
        stocks.columns = clean_ticks

        return stocks

    def screening(self, scores_df, stocks_df):
            '''
            Function for screening stocks based on their score, eliminate bottom 30% quantile

            Parameters:
            scores_df (DataFrame): DataFrame that contains the sustainability scores of the stocks

            stocks_df (DataFrame): DataFrame that contains the historical stock price data

            Returns:
            screen_stocks (DataFrame): DataFrame that contains the historical stock price data for the screened stocks

            screen_ticks (list): List that contains the tickers of the screened stocks
            '''
            
            mean_score = scores_df.mean()
            thresh = mean_score.quantile(0.3)
            mean_score = mean_score[mean_score > thresh]
            screen_stocks = stocks_df[mean_score.index]
            # screen_ticks = list(screen_stocks.columns)
            return screen_stocks

    def market_returns(self, stocks_df):
        stocks_returns = stocks_df.pct_change().dropna()
        market_returns = (stocks_returns * 1/stocks_returns.shape[1]).sum(axis=1)
        return market_returns

    def create_universe(self):
        '''Create universe of stocks, includes stock price, score score, e score, s score & g score data'''
        print('Fetching ESG data...')
        esg_df, e_df, s_df, g_df = self.get_score_data()
        print('ESG data done.')
        print()

        def miss_cols(df):
            '''
            Function for selecting tickers where more than 20% of the scores are missing 
            in order to drop them
            
            Parameters:
            df (DataFrame): DataFrame that contains the sustainability scores of the stocks

            Returns:
            cols (set): A set that contains the required tickers 
            '''
            cols = df.isnull().sum()[df.isnull().sum()>df.shape[0]*0.2].index.to_list()
            return set(cols)

        # Find the tickers with missing data
        esg_nan_cols = miss_cols(esg_df)
        e_nan_cols = miss_cols(e_df)
        s_nan_cols = miss_cols(s_df)
        g_nan_cols = miss_cols(g_df)

        # Find union of all the tickers with more than 20% of scores missing
        all_nan_tickers = esg_nan_cols.union(e_nan_cols, s_nan_cols, g_nan_cols)

        # Drop the tickers
        esg_df = esg_df.drop(all_nan_tickers, axis=1)
        e_df = e_df.drop(all_nan_tickers, axis=1)
        s_df = s_df.drop(all_nan_tickers, axis=1)
        g_df = g_df.drop(all_nan_tickers, axis=1)

        # For the tickers that less than 20% of the scores are missing use interpolation to fill
        esg_df = esg_df.interpolate()
        e_df = e_df.interpolate()
        s_df = s_df.interpolate()
        g_df = g_df.interpolate()

        self.esg_df = esg_df
        self.e_df = e_df
        self.s_df = s_df
        self.g_df = g_df

        # Update tickers attribute
        self.tickers = list(esg_df.columns)

        print('Fetching stock price data...')
        # Get stock price data
        self.stocks = self.get_stock_data()
        print('Stock price data done.')
        print()
        # Keep clean tickers of stocks DataFrame for consistency with the ESG DataFrames
        self.tickers = list(self.stocks.columns)
        self.e_df = self.e_df[self.tickers]
        self.s_df = self.s_df[self.tickers]
        self.g_df = self.g_df[self.tickers]
        self.esg_df = self.esg_df[self.tickers]
        
        # Create dictionary of stocks for different strategies for in and out of sample
        year_oos = str(int(self.year_insample)+1)
        
        # Market returns for in-sample and out-of-sample
        self.market_returns_in = self.market_returns(self.stocks.loc[:self.year_insample])
        self.market_returns_oof = self.market_returns(self.stocks.loc[year_oos:])

        self.stocks_dict_in = {'no_screen':self.stocks.loc[:self.year_insample]}
        self.stocks_dict_out = {'no_screen':self.stocks.loc[year_oos:]}

        print('Implementing screening strategies...')
        # Apply screening strategies
        e_screen = self.screening(self.e_df.loc[:self.year_insample], self.stocks)
        self.stocks_dict_in['e_screen'] = e_screen.loc[:self.year_insample]
        self.stocks_dict_out['e_screen'] = e_screen.loc[year_oos:]

        s_screen = self.screening(self.s_df.loc[:self.year_insample], self.stocks)
        self.stocks_dict_in['s_screen'] = s_screen.loc[:self.year_insample]
        self.stocks_dict_out['s_screen'] = s_screen.loc[year_oos:]

        g_screen = self.screening(self.g_df.loc[:self.year_insample], self.stocks)
        self.stocks_dict_in['g_screen'] = g_screen.loc[:self.year_insample]
        self.stocks_dict_out['g_screen'] = g_screen.loc[year_oos:]

        esg_screen = self.screening(self.esg_df.loc[:self.year_insample], self.stocks)
        self.stocks_dict_in['esg_screen'] = esg_screen.loc[:self.year_insample]
        self.stocks_dict_out['esg_screen'] = esg_screen.loc[year_oos:]
        print('Screening done.')
        print('Stock universe created. ( ͡° ͜ʖ ͡°)')

def calculate_portfolio_return(weights, mean_returns):
    """
    Calculate the expected return of a portfolio.

    Parameters:
    weights (numpy array): The stock weights in the portfolio.
    mean_returns (Series): The mean returns of each stock.

    Returns:
    float: The expected return of the portfolio.
    """
    if not isinstance(weights, np.ndarray):
        raise ValueError("weights must be a numpy array.")
    
    if not isinstance(mean_returns, pd.Series):
        raise ValueError("mean_returns must be a pandas Series.")
    
    return mean_returns.to_numpy().T @ weights


def calculate_portfolio_volatility(weights, s):
    """
    Calculate the expected volatility of a portfolio.

    Parameters:
    weights (numpy array): The stock weights in the portfolio.
    s (DataFrame): The covariance matrix of stock returns.

    Returns:
    float: The expected volatility of the portfolio.
    """
    if not isinstance(weights, np.ndarray):
        raise ValueError("weights must be a numpy array.")
    
    if not isinstance(s, pd.DataFrame):
        raise ValueError("s must be a pandas DataFrame.")
    
    return np.sqrt(np.dot(weights.T, np.dot(s, weights)))


def negative_sharpe_ratio(weights, mean_returns, s, risk_free_rate):
    """
    Calculate the negative Sharpe ratio of a portfolio.

    Parameters:
    weights (numpy array): The stock weights in the portfolio.
    mean_returns (Series): The mean returns of each stock.
    s (DataFrame): The covariance matrix of stock returns.
    risk_free_rate (float): The risk-free rate used to calculate the Sharpe ratio.

    Returns:
    float: The negative Sharpe ratio of the portfolio.
    """
    if not isinstance(weights, np.ndarray):
        raise ValueError("weights must be a numpy array.")
    
    if not isinstance(mean_returns, pd.Series):
        raise ValueError("mean_returns must be a pandas Series.")
    
    if not isinstance(s, pd.DataFrame):
        raise ValueError("s must be a pandas DataFrame.")
    
    if not isinstance(risk_free_rate, (int, float)):
        raise ValueError("risk_free_rate must be a number.")
    
    portfolio_return = calculate_portfolio_return(weights, mean_returns)
    portfolio_volatility = calculate_portfolio_volatility(weights, s)
    return -(portfolio_return - risk_free_rate) / portfolio_volatility


def min_volatility(weights, s):
    """
    Calculate the expected volatility of a portfolio.

    Parameters:
    weights (numpy array): The stock weights in the portfolio.
    s (DataFrame): The covariance matrix of stock returns.

    Returns:
    float: The expected volatility of the portfolio.
    """
    if not isinstance(weights, np.ndarray):
        raise ValueError("weights must be a numpy array.")
    
    if not isinstance(s, pd.DataFrame):
        raise ValueError("s must be a pandas DataFrame.")
    
    portfolio_volatility = calculate_portfolio_volatility(weights, s)
    return portfolio_volatility

def optimize_portfolio(stocks, l_bound, u_bound, risk_free_rate=0.02, opt_method='neg_sharpe'):
    """
    Optimize a portfolio of stocks using either negative Sharpe ratio or minimum volatility method.

    Parameters:
    stocks (DataFrame): A DataFrame containing historical stock prices, with columns being stock tickers.
    l_bound (float): The lower bound for stock weights in the portfolio.
    u_bound (float): The upper bound for stock weights in the portfolio.
    risk_free_rate (float): The risk-free rate used to calculate the Sharpe ratio.
    opt_method (str): The optimization method to use, either 'neg_sharpe' or 'min_volatility'.

    Returns:
    optimal_weights (numpy array): The optimal stock weights for the given optimization method.
    portfolio_return (float): The expected return of the optimized portfolio.
    portfolio_volatility (float): The expected volatility of the optimized portfolio.
    """

    if not isinstance(stocks, pd.DataFrame):
        raise ValueError("stocks must be a pandas DataFrame.")

    if not (isinstance(l_bound, (int, float)) and isinstance(u_bound, (int, float))):
        raise ValueError("l_bound and u_bound must be numbers.")
    
    if l_bound > u_bound:
        raise ValueError("l_bound must be less than or equal to u_bound.")
    
    if not isinstance(risk_free_rate, (int, float)):
        raise ValueError("risk_free_rate must be a number.")
    
    if opt_method not in ['neg_sharpe', 'min_volatility']:
        raise ValueError("opt_method must be either 'neg_sharpe' or 'min_volatility'.")

    # Compute exponentially weighted historical mean returns
    mu = ema_historical_return(stocks)

    # Compute covariance matrix using ledoit-wolf shrinkage 
    s = CovarianceShrinkage(stocks).ledoit_wolf()

    # Define the constraint function for the optimization problem
    def constraints_func(weights):
        return np.sum(weights) - 1

    # Set up the optimization problem
    num_stocks = len(stocks.columns)
    initial_weights = np.full(num_stocks, 1/num_stocks) # More efficient way to create array with same values
    bounds = [(l_bound, u_bound) for _ in range(num_stocks)]
    constraints = {'type': 'eq', 'fun': constraints_func}

    # Choose optimization method and corresponding arguments
    if opt_method == 'neg_sharpe':
        opt_function = negative_sharpe_ratio
        opt_args = (mu, s, risk_free_rate)
    else:
        opt_function = min_volatility
        opt_args = (s,)

    # Perform the optimization
    result = minimize(
        opt_function, 
        initial_weights, args=opt_args,
        bounds=bounds, 
        constraints=constraints
    )

    # Check if optimization was successful
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    # Extract the optimal weights, portfolio return, and portfolio volatility
    optimal_weights = result.x
    portfolio_return = calculate_portfolio_return(optimal_weights, mu)
    portfolio_volatility = calculate_portfolio_volatility(optimal_weights, s)

    return optimal_weights, portfolio_return, portfolio_volatility

def compute_portfolio_performance(df: pd.DataFrame, weights: np.ndarray) -> tuple:
    """
    Calculate the daily and cumulative performance of a portfolio given its asset returns and weights.

    Parameters:
    -----------
    df: pandas DataFrame
        Historical daily returns for each asset in the portfolio.
        Each column represents an asset, and each row represents a day.
    weights: numpy ndarray
        An array containing the weights of the assets in the portfolio.

    Returns:
    --------
    tuple
        A tuple containing two pandas Series:
        1. Daily portfolio returns
        2. Cumulative portfolio returns
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Invalid input: 'df' must be a non-empty pandas DataFrame.")
    
    if not isinstance(weights, np.ndarray) or weights.size == 0:
        raise ValueError("Invalid input: 'weights' must be a non-empty numpy ndarray.")

    # Calculate daily percentage returns for each stock
    stock_rets = df.pct_change().dropna()

    # Apply weights to the stock returns
    weighted_stock_rets = stock_rets * weights

    # Calculate the daily portfolio returns by summing the weighted stock returns
    port_rets = weighted_stock_rets.sum(axis=1)

    # Calculate the cumulative portfolio returns using the daily portfolio returns
    cum_port_rets = (port_rets + 1).cumprod()

    return port_rets, cum_port_rets 

def portfolio_value_at_risk(returns, var_level=0.05):
    """
    Calculate the Value at Risk (VaR) of a portfolio using the fitted t-distribution method.

    Parameters:
    -----------
    returns: pandas Series
        Historical daily returns of the portfolio.
    var_level: float
        The desired VaR level (e.g., 0.01 for 1% VaR, 0.05 for 5% VaR).

    Returns:
    --------
    float
        The Value at Risk (VaR) for the given portfolio.
    """

    if not isinstance(returns, pd.Series) or returns.empty:
        raise ValueError("Invalid input: 'returns' must be a non-empty pandas Series.")

    if not (0 < var_level < 1):
        raise ValueError("Invalid input: 'var_level' must be a float between 0 and 1.")

    try:
        # Fit a t-distribution to the portfolio returns
        df, mu, sigma = stats.t.fit(returns)

        # Calculate the VaR using the fitted t-distribution
        var = stats.t.ppf(var_level, df, mu, sigma)
    except Exception as e:
        raise RuntimeError(f"Error calculating VaR: {str(e)}")

    return var

def t_cvar_integrand(x, df, mu, sigma):
    return x * stats.t.pdf(x, df, mu, sigma)

def compute_conditional_value_at_risk_t(port_returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calculate the Conditional Value at Risk (CVaR) for a given portfolio's returns using a t-distribution.

    Parameters:
    -----------
    port_returns: pandas Series
        A numpy ndarray containing the daily returns of the portfolio.
    alpha: float, optional, default = 0.05
        The significance level (quantile) at which to calculate the CVaR.
        For example, 0.05 represents the 5% quantile. Default is 0.05.

    Returns:
    --------
    float
        The Conditional Value at Risk (CVaR) for the given portfolio's returns.
    """

    if not isinstance(port_returns, pd.Series) or port_returns.empty:
        raise ValueError("Invalid input: 'port_returns' must be a non-empty pandas Series.")
    
    if not isinstance(alpha, float) or alpha < 0 or alpha > 1:
        raise ValueError("Invalid input: 'alpha' must be a float between 0 and 1.")

    # Fit a t-distribution to the returns
    df, mu, sigma = stats.t.fit(port_returns)

    # Calculate the Value at Risk (VaR) using the fitted t-distribution and specified quantile
    var = stats.t.ppf(alpha, df, mu, sigma)

    # Compute the CVaR by integrating the product of the t-distribution and return values over the range of [-inf, VaR]
    cvar, _ = quad(t_cvar_integrand, -np.inf, var, args=(df, mu, sigma))

    # Divide the result by the probability of the returns falling below VaR (alpha) to obtain the final CVaR value
    cvar /= alpha

    return cvar

def compute_maximum_drawdown(port_returns: pd.Series) -> float:
    """
    Calculate the maximum drawdown for a given portfolio's returns.

    Parameters:
    -----------
    port_returns: pandas Series
        A numpy ndarray containing the daily returns of the portfolio.

    Returns:
    --------
    float
        The maximum drawdown for the given portfolio's returns.
    """

    # Check if input is a non-empty numpy ndarray
    if not isinstance(port_returns, pd.Series) or port_returns.empty:
        raise ValueError("Invalid input: 'port_returns' must be a non-empty pandas Series.")
    
    # Calculate the cumulative returns
    cum_returns = np.cumprod(1 + port_returns)

    # Initialize the maximum drawdown and the maximum cumulative return seen so far
    max_drawdown = 0
    max_cum_return = cum_returns[0]

    # Loop through the cumulative returns, updating the maximum cumulative return and maximum drawdown as needed
    for curr_cum_return in cum_returns[1:]:
        # Update the maximum cumulative return if the current cumulative return is greater than the previous maximum
        max_cum_return = max(max_cum_return, curr_cum_return)

        # Calculate the drawdown by dividing the current cumulative return by the maximum cumulative return seen so far
        drawdown = (curr_cum_return - max_cum_return) / max_cum_return

        # Update the maximum drawdown if the current drawdown is greater than the previous maximum drawdown
        max_drawdown = min(max_drawdown, drawdown)

    return -max_drawdown


def compute_portfolio_beta(port_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate the beta of a portfolio given the portfolio returns, market returns, and a risk-free rate.

    Parameters:
    -----------
    port_returns: pandas Series
        A pandas Series containing the daily returns of the portfolio.
    market_returns: pandas Series
        A pandas Series containing the daily returns of the market.

    Returns:
    --------
    float
        The beta of the given portfolio.
    """

    # Check if inputs are valid
    if not isinstance(port_returns, pd.Series) or port_returns.empty:
        raise ValueError("Invalid input: 'port_returns' must be a non-empty pandas Series.")
    if not isinstance(market_returns, pd.Series) or market_returns.empty:
        raise ValueError("Invalid input: 'market_returns' must be a non-empty pandas Series.")

    # Check if the lengths of port_returns and market_returns are the same
    if port_returns.shape[0] != market_returns.shape[0]:
        raise ValueError("Invalid input: 'port_returns' and 'market_returns' must have the same length.")

    # Calculate the covariance between the excess portfolio returns and excess market returns
    covariance = np.cov(port_returns, market_returns)[0, 1]

    # Calculate the variance of the excess market returns
    market_variance = np.var(market_returns)

    # Calculate the beta by dividing the covariance by the market variance
    beta = covariance / market_variance

    return beta

class Portfolio_Simulation():
    """
    A class to simulate portfolio performance using different investment strategies.
    """
    def __init__(self, stock_universe, portfolio_size, l_bound, up_bound, risk_free_rate, opt_method, value_a_risk, runs):
        """
        Initialize the Portfolio_Simulation class.

        Parameters:
        -----------
        stock_universe: object
            A stock universe object containing stock data.
        portfolio_size: int
            The number of stocks in the portfolio.
        l_bound: float
            The lower bound for the stock weights.
        up_bound: float
            The upper bound for the stock weights.
        risk_free_rate: float
            The risk-free rate used for portfolio optimization.
        opt_method: str
            The optimization method used for portfolio optimization.
        value_a_risk: float
            The value-at-risk level used for computing the portfolio value-at-risk.
        runs: int
            The number of simulation runs.

        Methods:
        --------
        compute_portfolio_metrics()
            Compute the portfolio metrics for a given stock dataframe and weights array.
        run()
            Run the portfolio simulation.
        simulate()
            Run simulation for all portfolios.
        plot_joyplot()
            Plots a joyplot of the given numpy arrays using the joypy library.
        create_plot()
            Create joyplot for given metric.
        mann_whitney_u_test()
            Performs the Mann-Whitney U test between the first array and all other arrays.
        statistical_test()
            Compute statistical test and create results report.
        """
        self.universe = stock_universe
        self.portfolio_size = portfolio_size
        self.l_bound = l_bound
        self.up_bound = up_bound
        self.rfr = risk_free_rate
        self.opt_method = opt_method
        self.var = value_a_risk
        self.runs = runs

    def compute_portfolio_metrics(self, stocks: pd.DataFrame, weights: np.ndarray):
        """
        Compute the portfolio metrics for a given stock dataframe and weights array.

        Parameters:
        -----------
        stocks: pd.DataFrame
            The stock price dataframe.
        weights: np.ndarray
            The weights array for the stocks in the portfolio.

        Returns:
        --------
        tuple
            A tuple containing the computed portfolio metrics.
        """
        # Make sure the required functions are implemented
        if not callable(compute_portfolio_performance) or not callable(portfolio_value_at_risk) or not callable(compute_conditional_value_at_risk_t) or not callable(compute_maximum_drawdown):
            raise NotImplementedError("Required functions are not implemented.")

        port_ret, cum_port_ret = compute_portfolio_performance(stocks, weights)
        port_vol = port_ret.std()
        port_var = portfolio_value_at_risk(port_ret,self.var)
        port_cvar = compute_conditional_value_at_risk_t(port_ret)
        port_draw = compute_maximum_drawdown(port_ret)

        return cum_port_ret[-1], port_vol, port_var, port_cvar, port_draw

    def run(self, stocks_in_sample: pd.DataFrame, stocks_out_of_sample: pd.DataFrame, runs: int):
        """
        Run the portfolio simulation.

        Parameters:
        -----------
        stocks_in_sample: pd.DataFrame
            The in-sample stock price dataframe.
        stocks_out_of_sample: pd.DataFrame
            The out-of-sample stock price dataframe.
        runs: int
            The number of simulation runs.

        Returns:
        --------
        dict
            A dictionary containing the computed portfolio metrics for the in-sample and out-of-sample data.
        """
        if not callable(optimize_portfolio):
            raise NotImplementedError("Required function 'optimize_portfolio' is not implemented.")

        cum_port_ret_list_in = []
        port_vol_list_in = []
        port_var_list_in = []
        port_cvar_list_in = []
        port_drawdown_list_in = []

        cum_port_ret_list_oof = []
        port_vol_list_oof = []
        port_var_list_oof = []
        port_cvar_list_oof = []
        port_drawdown_list_oof = []

        for i in range(runs):
            tickers_sample = list(np.random.choice(stocks_in_sample.columns, self.portfolo_size))
            stock_sample_in = stocks_in_sample[tickers_sample]
            stock_weights, exp_port_annual_ret, exp_port_annual_vol = optimize_portfolio(
                stock_sample_in, 
                self.l_bound, 
                self.up_bound, 
                self.rfr, 
                self.opt_method
            )

            cum_port_ret_in, port_vol_in, port_var_in, port_cvar_in, port_draw_in = self.compute_portfolio_metrics(stock_sample_in, stock_weights)

            cum_port_ret_list_in.append(cum_port_ret_in)
            port_vol_list_in.append(port_vol_in)
            port_var_list_in.append(port_var_in)
            port_cvar_list_in.append(port_cvar_in)
            port_drawdown_list_in.append(port_draw_in)

            stock_sample_oof = stocks_out_of_sample[tickers_sample]
            cum_port_ret_oof, port_vol_oof, port_var_oof, port_cvar_oof, port_draw_oof = self.compute_portfolio_metrics(stock_sample_oof, stock_weights)

            cum_port_ret_list_oof.append(cum_port_ret_oof)
            port_vol_list_oof.append(port_vol_oof)
            port_var_list_oof.append(port_var_oof)
            port_cvar_list_oof.append(port_cvar_oof)
            port_drawdown_list_oof.append(port_draw_oof)

            if i == runs-1:
                print('Simulation completed.')
                print()

        cum_port_ret_dict = {'in':np.array(cum_port_ret_list_in), 'oof':np.array(cum_port_ret_list_oof)}
        port_vol_list_dict = {'in':np.array(port_vol_list_in), 'oof':np.array(port_vol_list_oof)}
        port_var_list_dict = {'in':np.array(port_var_list_in), 'oof':np.array(port_var_list_oof)}
        port_cvar_list_dict = {'in':np.array(port_cvar_list_in), 'oof':np.array(port_cvar_list_oof)}
        port_drawdown_list_dict = {'in':np.array(port_drawdown_list_in), 'oof':np.array(port_drawdown_list_oof)}

        return_dict = {
            'cummulative_returns':cum_port_ret_dict, 
            'daily_volatility':port_vol_list_dict, 
            'value_at_risk':port_var_list_dict, 
            'conditional_value_at_risk':port_cvar_list_dict, 
            'drawdown':port_drawdown_list_dict
        }

        return return_dict

    def simulate(self):
        """
        Simulates portfolio performance for different investment strategies and stores the results in the
        'metrics_distributions' attribute.

        Returns:
        --------
        None
        """
        if not hasattr(self, "universe") or not hasattr(self, "runs"):
            raise ValueError("Portfolio_Simulation instance must be properly initialized before calling 'simulate()'.")

        if not all(key in self.universe.stocks_dict_in and key in self.universe.stocks_dict_out for key in ['no_screen', 'e_screen', 's_screen', 'g_screen', 'esg_screen']):
            raise ValueError("Invalid universe attribute. Please ensure that the 'stocks_dict_in' and 'stocks_dict_out' attributes of the 'universe' instance contain the required keys.")
        
        self.metrics_distributions = {
            'no_screen': self.run(self.universe.stocks_dict_in['no_screen'], self.universe.stocks_dict_out['no_screen'], self.runs),
            'e_screen': self.run(self.universe.stocks_dict_in['e_screen'], self.universe.stocks_dict_out['e_screen'], self.runs),
            's_screen': self.run(self.universe.stocks_dict_in['s_screen'], self.universe.stocks_dict_out['s_screen'], self.runs),
            'g_screen': self.run(self.universe.stocks_dict_in['g_screen'], self.universe.stocks_dict_out['g_screen'], self.runs),
            'esg_screen': self.run(self.universe.stocks_dict_in['esg_screen'], self.universe.stocks_dict_out['esg_screen'], self.runs),
        }


    def plot_joyplot(self, *arrays, metric=None, labels=None, figsize=None):
        """
        Plots a joyplot of the given numpy arrays using the joypy library.

        Parameters:
        -----------
        *arrays: numpy ndarrays
            A dynamic number of 1D numpy arrays for which the joyplot will be plotted.
        metric: str, optional
            The metric of interest
        labels: list of str, optional
            A list of labels for the joyplot, one for each array (default is None).
        figsize: tuple, optional
            A tuple of the width and height in inches for the figure (default is None).
        """

        if not arrays:
            raise ValueError("At least one array must be provided.")

        if labels is not None and len(arrays) != len(labels):
            raise ValueError("The number of labels must match the number of arrays.")

        data = pd.DataFrame()
        for i, arr in enumerate(arrays):
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise ValueError(f"Invalid input: array {i} must be a 1D numpy ndarray.")
            data[f'Array {i}'] = arr

        if labels is not None:
            data.columns = labels

        if figsize is not None:
            plt.figure(figsize=figsize)

        fig, axes = joypy.joyplot(data, colormap=sns.color_palette("crest", as_cmap=True), alpha=0.5)
        
        plt.xlabel(metric)
        plt.show()

    def create_plot(self, metric, sample_type):
    
        metric_no = self.metrics_distributions['no_screen'][metric][sample_type]
        metric_e = self.metrics_distributions['e_screen'][metric][sample_type]
        metric_s = self.metrics_distributions['s_screen'][metric][sample_type]
        metric_g = self.metrics_distributions['g_screen'][metric][sample_type]
        metric_esg = self.metrics_distributions['esg_screen'][metric][sample_type]

        labels = ['No Screen', 'E Screen', 'S Screen', 'G Screen', 'ESG Screen']
        self.plot_joyplot(metric_no, metric_e, metric_s, metric_g, metric_esg, metric=metric, labels=labels, figsize=(10,6))

    def mann_whitney_u_test(self, *arrays, metric, labels):
        """
        Performs the Mann-Whitney U test between the first array and all other arrays.

        Parameters:
        -----------
        *arrays: numpy ndarrays
            A dynamic number of 1D numpy arrays for which the Mann-Whitney U test will be performed.
        metric: str
            The metric compared.
        labels: list
            A list with the labels that correspond to the arrays given.
        Returns:
        --------
        None
        """
        if not all(isinstance(arr, np.ndarray) for arr in arrays):
            raise ValueError("All input arguments must be numpy ndarrays.")

        if len(arrays) < 2:
            raise ValueError("At least two arrays must be provided for comparison.")

        first_array = arrays[0]

        for i, array in enumerate(arrays[1:], start=1):
            u_stat, p_value = mannwhitneyu(first_array, array, alternative='two-sided')

            print(f"Comparing '{labels[0]}' strategy with '{labels[i]}' strategy:")
            print(f"Metric: {metric}")
            print(f"Mann-Whitney U Statistic: {u_stat:.2f}")
            print(f"p-value: {p_value:.4f}")

            if p_value < 0.05:
                print("Result: The two samples are significantly different (p < 0.05).\n")
            else:
                print("Result: The two samples are not significantly different (p >= 0.05).\n")

    def statistical_test(self, metric, sample_type):
        """
        Performs Mann-Whitney U tests on the specified metric for different investment strategies.

        Parameters:
        -----------
        metric: str
            The performance metric to be compared, e.g., 'cummulative_returns', 'daily_volatility', etc.
        sample_type: str
            The type of sample to be considered, either 'in' (in-sample) or 'oof' (out-of-sample).

        Returns:
        --------
        None
        """
        if metric not in self.metrics_distributions['no_screen']:
            raise ValueError("Invalid metric. Please provide a valid metric present in metrics_distributions.")
        
        if sample_type not in ['in', 'oof']:
            raise ValueError("Invalid sample_type. Please provide either 'in' or 'oof'.")

        metric_no = self.metrics_distributions['no_screen'][metric][sample_type]
        metric_e = self.metrics_distributions['e_screen'][metric][sample_type]
        metric_s = self.metrics_distributions['s_screen'][metric][sample_type]
        metric_g = self.metrics_distributions['g_screen'][metric][sample_type]
        metric_esg = self.metrics_distributions['esg_screen'][metric][sample_type]

        labels = ['No Screen', 'E Screen', 'S Screen', 'G Screen', 'ESG Screen']
        self.mann_whitney_u_test(metric_no, metric_e, metric_s, metric_g, metric_esg, metric=metric, labels=labels)
    
    
    