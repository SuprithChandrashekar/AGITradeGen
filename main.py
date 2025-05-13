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

# Log initial run
append_results({
    "timestamp":                   pd.Timestamp.now(),
    "strategy_description":        desc1,
    "strategy_code":               code1,
    "backtest_results":            results_str1,
})

# ─── Get Historical Champion ───────────────────────────────
try:
    champ = best_historical()
    champion_ctx = (
        f"HISTORICAL DESCRIPTION:\n{champ.get('strategy_description','')}\n\n"
        f"HISTORICAL CODE:\n```python\n{champ.get('strategy_code','')}\n```\n\n"
        f"HISTORICAL METRICS:\n{champ.get('improved_backtest_results','')}"
    )
except Exception as e:
    print("[SUPERVISOR] no historical context:", e)
    champion_ctx = None

# ─── 2nd Strategy Cycle (Improved) ─────────────────────────
code2, desc2 = improve_strategy(
    df1,
    code1,
    results_str1,
    ticker="TSLA",
    historical_context=champion_ctx,
)
df2 = execute_strategy(test_df, code2)
results_str2, results2, df2 = backtest_strategy(
    df2, capital=10_000, fee_per_trade=0.001, verbose=True
)

# Log improved run
append_results({
    "timestamp":                         pd.Timestamp.now(),
    "strategy_description":              desc1,
    "strategy_code":                     code1,
    "backtest_results":                  results_str1,
    "improved_strategy_description":     desc2,
    "improved_strategy_code":            code2,
    "improved_backtest_results":         results_str2,
})

print("✓ Finished full two-cycle run with supervisor feedback")