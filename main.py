from agent.data_fetch import fetch_intraday_data
from agent.strategy import generate_strategy, execute_strategy, improve_strategy
from agent.eval import backtest_strategy, plot_backtest

import pandas as pd
import numpy as np


data = fetch_intraday_data("TSLA", "5m", "5d", True)

df = fetch_intraday_data("TSLA", "5m", "5d", use_cache=True)
code, description = generate_strategy(df)
df1 = execute_strategy(df, code)
results_str, results, df1 = backtest_strategy(df1, capital=10000, fee_per_trade=0.001, verbose=True)
second_code, second_description = improve_strategy(df1, code, results_str, ticker="TSLA")
df2 = execute_strategy(df, second_code)
results_str2, results2, df2 = backtest_strategy(df2, capital=10000, fee_per_trade=0.001, verbose=True)


print("\n[STRATEGY DESCRIPTION]\n", description)
print("\n[STRATEGY CODE]\n", code)
print("\n[BACKTEST RESULTS]\n", results_str)
print("\n[IMPROVED STRATEGY DESCRIPTION]\n", second_description)
print("\n[IMPROVED STRATEGY CODE]\n", second_code)
print("\n[IMPROVED BACKTEST RESULTS]\n", results_str2)
