# Fetch data
import yfinance as yf
from datetime import datetime
import pandas as pd

def fetch_data(ticker="AAPL", start="2018-01-01", end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    return yf.download(ticker, start=start, end=end)

def fetch_vix_spy(start="2015-01-01", end=None):
    end = end or pd.Timestamp.today().strftime('%Y-%m-%d')
    spy = yf.download("SPY", start=start, end=end)['Close']
    spy.name = "SPY"
    vix = yf.download("^VIX", start=start, end=end)['Close']
    vix.name = "VIX"
    df = pd.concat([spy, vix], axis=1).dropna()
    return df

df = fetch_vix_spy()
df = df.rename(columns={"^VIX": "VIX"})