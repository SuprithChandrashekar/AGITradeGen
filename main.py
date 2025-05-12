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

# ─── initial strategy cycle ───────────────────────────────────────────
code, description = generate_strategy(train_df)
df1 = execute_strategy(test_df, code)
results_str, results, df1 = backtest_strategy(df1, capital=10_000,
                                             fee_per_trade=0.001, verbose=True)

# log first cycle
append_results({
    "strategy_description":          description,
    "strategy_code":                 code,
    "backtest_results":              results_str,
})

# ─── get historic champion for the LLM ───────────────────────────────
try:
    champ = best_historical()
    champion_ctx = (
        f"HISTORICAL DESCRIPTION:\n{champ.get('strategy_description','')}\n\n"
        f"HISTORICAL CODE:\n```python\n{champ.get('strategy_code','')}\n```\n\n"
        f"HISTORICAL METRICS:\n{champ.get('improved_backtest_results','')}"
    )
except Exception as err:
    print("[SUPERVISOR] no historical context:", err)
    champion_ctx = None

# ─── improved strategy cycle ─────────────────────────────────────────
second_code, second_description = improve_strategy(
    df1, code, results_str, ticker="TSLA",
    historical_context=champion_ctx,         # ← NEW
)
df2 = execute_strategy(test_df, second_code)
results_str2, results2, df2 = backtest_strategy(df2, capital=10_000,
                                                fee_per_trade=0.001, verbose=True)

print("✓ finished full two‑cycle run with supervisor feedback")
#plot_backtest(df2)


print("\n[STRATEGY DESCRIPTION]\n", description)
print("\n[STRATEGY CODE]\n", code)
print("\n[BACKTEST RESULTS]\n", results_str)
print("\n[IMPROVED STRATEGY DESCRIPTION]\n", second_description)
print("\n[IMPROVED STRATEGY CODE]\n", second_code)
print("\n[IMPROVED BACKTEST RESULTS]\n", results_str2)
print(df2.head())

report_data = {
    "timestamp": None,  # This will be set by the append_report function if None
    "strategy_description": description,
    "strategy_code": code,
    "backtest_results": results_str,
    "improved_strategy_description": second_description,
    "improved_strategy_code": second_code,
    "improved_backtest_results": results_str2
}
append_results(report_data)
print("Results have been appended to the report Excel file.")