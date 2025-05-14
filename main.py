from agent.data_fetch import fetch_intraday_data
from agent.data_split import leave_one_out_cv, ransac_regression, split_data
from agent.strategy import generate_strategy, execute_strategy, improve_strategy
from agent.supervision import best_historical
from agent.eval import backtest_strategy, plot_backtest
from reports.append_results import append_results

import pandas as pd
import numpy as np
import json
import traceback
import sys
import time

"""
 - Create agent loop, prompt the user and then execute all of this based of prompt
 - Data fetch/cache
 - Error handling from generated
 - Testing/Continue prompt engineering
 - Backtesting is currently on same exact set of data it initially fetched (train/test split)
 - Potentially add new agent to externally judge and improve the strategy
"""

from datetime import datetime, timezone
import json
import traceback
import os

start_time = time.time()

def get_logger(module_name):
    DEBUG_LEVEL = os.environ.get("DEBUG_LEVEL", "info").lower()
    level_priority = {"verbose": 0, "info": 1, "warn": 2, "error": 3}

    def log(level, tag, message, data=None, symbol=None, line=None):
        if level_priority[level] < level_priority.get(DEBUG_LEVEL, 1):
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        origin = f"[{module_name.upper()}]"
        symbol_str = f" - Called from {symbol}:{line}" if symbol and line else ""
        try:
            if data is not None:
                if isinstance(data, (dict, list)):
                    data_str = json.dumps(data, default=str)
                else:
                    data_str = str(data)
                message = f"{message} - {data_str}"
        except Exception:
            message = f"{message} - [ERROR SERIALIZING DATA]"
        print(f"[{timestamp}] {origin} {tag.upper()}{symbol_str} - {message}")
    return log

log = get_logger("main")

args = sys.argv
env_vars = {
    "DEBUG_LEVEL": os.environ.get("DEBUG_LEVEL", "info"),
    "PYTHON_ENV": os.environ.get("PYTHON_ENV", "dev")
}

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
log("info", "MODULE IN", "generate_strategy()", {"ticker": "TSLA"}, symbol="main", line=42)
code1, desc1 = generate_strategy(train_df)
log("info", "MODULE OUT", "generate_strategy()", {"desc": desc1[:100]})

log("info", "MODULE IN", "execute_strategy()", {"sample_code": code1[:100]}, symbol="main", line=48)

try:
    df1 = execute_strategy(test_df, code1)
    log("info", "MODULE OUT", "execute_strategy()", {"rows": len(df1)})
except Exception as e:
    log("error", "ERROR", "Failed to execute_strategy()", {
        "error": str(e),
        "trace": traceback.format_exc()
    }, symbol="main", line=51)
    raise

results_str1, results1, df1 = backtest_strategy(
    df1, capital=10_000, fee_per_trade=0.001, verbose=True
)
log("info", "MODULE OUT", "backtest_strategy() [original]", results1, symbol="main", line=56)
print("[RESULT_JSON_ORIG] " + json.dumps(results1, indent=2))


log_entry = {
    "timestamp": pd.Timestamp.now(),
    "strategy_description": desc1,
    "strategy_code": code1,
    "backtest_results": results_str1
}

try:
    log("info", "MODULE IN", "improve_strategy()", {
    "base_code_sample": code1[:100],
    "base_metrics": results1
    }, symbol="main", line=60)

    second_code, desc2 = improve_strategy(df1, code1, results_str1, ticker="TSLA")
    log("info", "MODULE OUT", "improve_strategy()", {"desc": desc2[:100]})

    log("info", "MODULE IN", "execute_strategy() [improved]", {"code_sample": second_code[:100]}, symbol="main", line=67)
    print("[FULL_IMPROVED_CODE]\n" + second_code + "\n[END_FULL_IMPROVED_CODE]")
    df2 = execute_strategy(test_df, second_code)
    log("info", "MODULE OUT", "execute_strategy() [improved]", {"rows": len(df2)})
    results_str2, results2, df2 = backtest_strategy(df2, capital=10000, fee_per_trade=0.001, verbose=True)
    log("info", "MODULE OUT", "backtest_strategy() [improved]", results2, symbol="main", line=71)
    print("[RESULT_JSON_IMPROVED] " + json.dumps(results2, indent=2))

    log_entry["improved_strategy_description"] = desc2
    log_entry["improved_strategy_code"] = second_code
    log_entry["improved_backtest_results"] = results_str2

except Exception as e:
    log("error", "ERROR", "Improved strategy pipeline failed", {
        "error": str(e),
        "trace": traceback.format_exc()
    }, symbol="main", line=75)
    print(f"[ERROR] Failed to improve strategy: {e}")
    log_entry["improved_strategy_description"] = "(Improvement Failed)"
    log_entry["improved_strategy_code"] = ""
    log_entry["improved_backtest_results"] = ""

log("info", "MAIN END", "Appending final results", {
    "orig_return_pct": results1.get("Total Return (%)"),
    "improved_return_pct": results2.get("Total Return (%)") if 'results2' in locals() else None
})

append_results(log_entry)
log("info", "MAIN END", f"Completed in {round(time.time() - start_time, 2)}s")