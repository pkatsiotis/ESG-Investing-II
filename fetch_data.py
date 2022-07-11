import pandas as pd
import urllib.request
import json
import datetime
from pandas_datareader.data import DataReader

def get_sp_tickers(date):
    '''Get S&P 500 stock tickers from Wikipedia page
    
    Parameters
    ----------
    date: str
        Date in the format %Y-%m-%d to filter the stocks, only stocks added to S&P 500
        before the given date will be kept
    
    Returns
    -------
    tickers: list
        A list of strings that are tickers of stocks
    '''
    
    try:
        datetime.datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")

    wiki = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    first_table = wiki[0]

    # Keep tickers that were first added before 'date'
    df = first_table
    df_date_filter = df[df['Date first added'] < date].copy()
    tickers = df_date_filter['Symbol'].to_list()
    return tickers

def get_esg_data(tickers, year_start, year_end):
    '''Get the historical ESG data of the tickers given from yahoo finance
    
    Parameters
    ----------
    tickers: list
        A list of strings that are tickers of stocks
    year_start: str
        Starting year of historical data to be kept
    year_end: str
        Ending year of historical data to be kept

    Returns
    -------
    esg_df: pandas DataFrame
        A DataFrame that contains the historial total ESG scores of the given tickers
    e_df: pandas DataFrame
        A DataFrame that contains the historial Environmental (E) scores of the given tickers
    s_df: pandas DataFrame
        A DataFrame that contains the historial Social (S) scores of the given tickers
    g_df: pandas DataFrame
        A DataFrame that contains the historial Governance (G) scores of the given tickers
    '''

    # Lists for saving the historical esg scores for each ticker, and missing list in case data are missing for a ticker
    esg_list = []
    e_list = []
    s_list = []
    g_list = []
    missing = []

    for ticker in tickers:
        # Try fetching ESG data from yahoo finance
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
                df_data = df_data.loc[year_start:year_end]
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
            tickers.remove(tick)

    esg_df = pd.concat(esg_list, axis=1)
    esg_df.columns = tickers

    e_df = pd.concat(e_list, axis=1)
    e_df.columns = tickers

    s_df = pd.concat(s_list, axis=1)
    s_df.columns = tickers

    g_df = pd.concat(g_list, axis=1)
    g_df.columns = tickers

    return esg_df, e_df, s_df, g_df

def get_stock_data(tickers, year_start, year_end):
    '''Get historical stock price data of the tickers given
    
    Parameters
    ----------
    tickers: list
        A list of strings that are tickers of stocks
    year_start: str
        Starting year of historical data to be kept
    year_end: str
        Ending year of historical data to be kept

    Returns
    -------
    stocks: pandas DataFrame
        A DataFrame that contains the historical stock price data of the given tickers
    '''

    year_start = int(year_start)
    year_end = int(year_end)

    # First calendar day of historical data
    start = datetime.date(year_start,1,1)
    # Last calendar day of historical data
    end = datetime.date(year_end,12,31)

    data_source = 'yahoo'
    
    stocks_list = []
    for tick in tickers:
        try:
            stocks = DataReader(tick, data_source, start, end)
            stocks_list.append(stocks[['Close']].add_suffix('_'+tick))
        except:
            continue
        
    stocks = pd.concat(stocks_list, axis = 1)
    clean_ticks = list(map(lambda x: x.split('_')[1], stocks.columns))
    stocks.columns = clean_ticks

    return stocks