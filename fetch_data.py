import pandas as pd
import urllib.request
import json
import datetime
import yfinance as yf

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
    df_date_filter = df[df['Date added'] < date].copy()
    tickers = df_date_filter['Symbol'].to_list()
    return tickers