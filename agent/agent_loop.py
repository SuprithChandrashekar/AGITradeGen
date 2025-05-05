from agent.data_fetch import fetch_intraday_data
from agent.strategy import generate_strategy, execute_strategy, improve_strategy
from agent.eval import backtest_strategy, plot_backtest


"""
- Either loop n times or until the strategy achieves desired return
"""