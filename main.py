from agent.data_fetch import fetch_intraday_data
from agent.data_split import leave_one_out_cv, ransac_regression, split_data
from agent.strategy import generate_strategy, execute_strategy, improve_strategy
from agent.eval import backtest_strategy, plot_backtest

import pandas as pd
import numpy as np

"""
 - Create agent loop, prompt the user and then execute all of this based of prompt
 - Data fetch/cache
 - Error handling from generated
 - Testing/Continue prompt engineering
 - Backtesting is currently on same exact set of data it initially fetched (train/test split)
 - Potentially add new agent to externally judge and improve the strategy
"""



df_raw = fetch_intraday_data("TSLA", "5m", "5d", use_cache=True)
ransac_results = ransac_regression(
    df_raw,
    feature_cols=["Open", "High", "Low", "Close", "Volume"],
    target_col="Close"
)
df_clean = df_raw[ransac_results["inliers"]].reset_index(drop=True)
splits = split_data(df_clean, test_size=0.2, shuffle=False)
df = splits['train']
test_df = splits['test']

code, description = generate_strategy(df)
df1 = execute_strategy(test_df, code)
results_str, results, df1 = backtest_strategy(df1, capital=10000, fee_per_trade=0.001, verbose=True)
second_code, second_description = improve_strategy(df1, code, results_str, ticker="TSLA")
df2 = execute_strategy(df, second_code)
results_str2, results2, df2 = backtest_strategy(df2, capital=10000, fee_per_trade=0.001, verbose=True)
#plot_backtest(df2)


print("\n[STRATEGY DESCRIPTION]\n", description)
print("\n[STRATEGY CODE]\n", code)
print("\n[BACKTEST RESULTS]\n", results_str)
print("\n[IMPROVED STRATEGY DESCRIPTION]\n", second_description)
print("\n[IMPROVED STRATEGY CODE]\n", second_code)
print("\n[IMPROVED BACKTEST RESULTS]\n", results_str2)
print(df2.head())
