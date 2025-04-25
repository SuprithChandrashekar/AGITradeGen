from agent.data_fetch import fetch_intraday_data
from agent.strategy import generate_strategy, generate_and_validate_strategy
from agent.eval import backtest_strategy

import pandas as pd
import numpy as np


data = fetch_intraday_data("TSLA", "5m", "5d", True)

df = fetch_intraday_data("TSLA", "5m", "5d", use_cache=True)
code, description = generate_and_validate_strategy(df)

print("\n[STRATEGY DESCRIPTION]\n", description)
print("\n[STRATEGY CODE]\n", code)