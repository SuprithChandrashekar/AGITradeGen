from execution.execution_broker import ExecutionBroker
from strategies.strategy_1_volatility_breakout import generate_volatility_compression_signals
from datasets.data_fetch import fetch_data
from strategies.s2 import generate_vix_fade_signals
# Step 1: Get historical data
df = fetch_data("SPY", start="2020-01-01") # 1-2 years of context
ticker = df.columns[0][1]

# Step 2: Run strategy — only today's signal is used
today_signal, entry_price, exit_price, stop_loss = generate_vix_fade_signals(df)
# Calculate position size based on entry and stop loss
def calculate_position(entry, stop, capital=100_000, max_risk_per_trade=0.01):
    risk_per_share = abs(entry - stop)
    max_risk_dollars = capital * max_risk_per_trade
    size = int(max_risk_dollars / risk_per_share)
    return size

# Step 3: Live or paper trade
broker = ExecutionBroker(mode='paper', client_id=123)

if today_signal == 1:
    print(f"📈 Buy Signal Triggered Today")
    print(f"Entry: {entry_price}, Exit: {exit_price}, SL: {stop_loss}")
    broker.submit_order(
        ticker= ticker,  # Dynamically fetch the ticker from the data
        signal=1,
        price=entry_price,
        size=calculate_position(entry_price, stop_loss)
    )

else:
    print(f"No actionable signal today. Entry: {entry_price}, Exit: {exit_price}, SL: {stop_loss}")

broker.disconnect()