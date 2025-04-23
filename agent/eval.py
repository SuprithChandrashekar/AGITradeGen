import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from statistics import stdev as StdDev  # or define a rolling one
import traceback


def backtest_strategy(df, strategy_code: str):
    try:
        # Prepare local namespace and exec strategy code
        local_env = {}
        exec(strategy_code, globals(), local_env)

        # Find the Strategy subclass
        strategy_class = None
        for obj in local_env.values():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                strategy_class = obj
                break

        if strategy_class is None:
            raise ValueError("No valid Strategy subclass found in generated code.")

        # Prepare DataFrame: make sure it has proper OHLCV columns
        bt_df = df.copy()
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(bt_df.columns):
            raise ValueError("Input DataFrame missing required OHLCV columns.")

        # Drop rows with NaNs (indicators might produce some)
        bt_df.dropna(inplace=True)

        # Run the backtest
        bt = Backtest(bt_df, strategy_class, cash=10_000, commission=0.001, exclusive_orders=True)
        stats = bt.run()
        bt.plot()  # comment this out if running headless or non-GUI

        return {
            "Cumulative Return": stats["Equity Final [$]"] / stats["Equity Start [$]"] - 1,
            "Sharpe Ratio": stats["Sharpe Ratio"],
            "Max Drawdown": stats["Max Drawdown [%]"],
            "Win Rate": stats["Win Rate [%]"],
            "Trades": stats["# Trades"],
            "Raw Stats": stats 
        }

    except Exception as e:
        print("[ERROR] Backtest failed:")
        traceback.print_exc()
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def generate_report(results: dict, strategy_description: str):
    print("\n--- STRATEGY REPORT ---")
    print("Description:", strategy_description)
    print("Cumulative Return:", round(results["cumulative_return"], 4))
    print("Sharpe Ratio:", round(results["sharpe"], 4))
    print("------------------------")