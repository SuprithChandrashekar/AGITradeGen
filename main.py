from agent.data_fetch import fetch_intraday_data
from agent.data_split import leave_one_out_cv, ransac_regression, split_data
from agent.strategy import generate_strategy, execute_strategy, improve_strategy
from agent.supervision import best_historical
from agent.eval import backtest_strategy, plot_backtest
from reports.append_results import append_results

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
train_df = splits['train']
test_df = splits['test']

# ─── 1st Strategy Cycle ─────────────────────────────────────
code1, desc1 = generate_strategy(train_df)
df1 = execute_strategy(test_df, code1)
results_str1, results1, df1 = backtest_strategy(
    df1, capital=10_000, fee_per_trade=0.001, verbose=True
)

log_entry = {
    "timestamp": pd.Timestamp.now(),
    "strategy_description": desc1,
    "strategy_code": code1,
    "backtest_results": results_str1
}

try:
    second_code, desc2 = improve_strategy(df1, code1, results_str1, ticker="TSLA")
    df2 = execute_strategy(test_df, second_code)
    results_str2, results2, df2 = backtest_strategy(df2, capital=10000, fee_per_trade=0.001, verbose=True)

    log_entry["improved_strategy_description"] = desc2
    log_entry["improved_strategy_code"] = second_code
    log_entry["improved_backtest_results"] = results_str2

except Exception as e:
    print(f"[ERROR] Failed to improve strategy: {e}")
    log_entry["improved_strategy_description"] = "(Improvement Failed)"
    log_entry["improved_strategy_code"] = ""
    log_entry["improved_backtest_results"] = ""

append_results(log_entry)
