import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from statistics import stdev as StdDev  # or define a rolling one
import traceback
import matplotlib.pyplot as plt


def backtest_strategy(df, capital=10000, fee_per_trade=0.001, verbose=True):
    """
    Vectorized backtest based on a DataFrame with a 'signal' column: 1 (buy), -1 (sell), 0 (hold).
    Assumes trades happen at the next bar's open price.
    
    Returns a summary dictionary.
    """
    df = df.copy()
    if 'signal' not in df.columns:
        raise ValueError("DataFrame must contain a 'signal' column with values -1, 0, 1")

    # Shift signal so trade occurs on the next bar
    df['position'] = df['signal'].shift().fillna(0)

    # Calculate returns
    df['return'] = df['Close'].pct_change().fillna(0)
    df['strategy_return'] = df['position'] * df['return']

    # Calculate cumulative returns
    df['cumulative_market'] = (1 + df['return']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod()

    # Simulate capital
    df['capital'] = capital * df['cumulative_strategy']

    # Count trades
    df['trade'] = df['position'].diff().fillna(0).abs()
    total_trades = df['trade'].sum()

    # Apply trading fees
    total_fees = total_trades * fee_per_trade * capital
    final_value = df['capital'].iloc[-1] - total_fees

    mean_return = df['strategy_return'].mean()
    std_return = df['strategy_return'].std()
    if std_return == 0 or np.isnan(std_return):
        sharpe = 0.0
    else:
        sharpe = mean_return / std_return * np.sqrt(252 * 78)

    # Generate summary
    results = {
        "Initial Capital": float(capital),
        "Final Capital": round(final_value, 2),
        "Total Return (%)": round((final_value / capital - 1) * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(100 * ((df['capital'].cummax() - df['capital']) / df['capital'].cummax()).max(), 2),
        "Total Trades": int(total_trades),
        "Fee Cost": round(total_fees, 2),
    }

    summary_lines = ["📊 [BACKTEST RESULTS]", "-" * 35]
    for k, v in results.items():
        if isinstance(v, (float, int, np.integer, np.floating)):
            summary_lines.append(f"{k:<20}: {v:,.2f}")
        else:
            summary_lines.append(f"{k:<20}: {v}")
    summary_lines.append("-" * 35)

    results_str = "\n".join(summary_lines)
    return results_str, results, df


def plot_backtest(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Datetime'], df['cumulative_market'], label='Market Return')
    plt.plot(df['Datetime'], df['cumulative_strategy'], label='Strategy Return')
    plt.legend()
    plt.title("Backtest Performance")
    plt.xlabel("Datetime")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()