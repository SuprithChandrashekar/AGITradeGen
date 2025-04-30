from agent.data_fetch import fetch_intraday_data
from agent.strategy import generate_strategy, execute_strategy
from agent.eval import backtest_strategy, plot_backtest

import pandas as pd
import numpy as np


data = fetch_intraday_data("TSLA", "5m", "5d", True)

df = fetch_intraday_data("TSLA", "5m", "5d", use_cache=True)
code, description = generate_strategy(df)
df = execute_strategy(df, code)
results_str, results, df = backtest_strategy(df, capital=10000, fee_per_trade=0.001, verbose=True)

print("\n[STRATEGY DESCRIPTION]\n", description)
print("\n[STRATEGY CODE]\n", code)
print("\n[DATA HEAD]\n", df.head())
print("\n[BACKTEST RESULTS]\n", results_str)
