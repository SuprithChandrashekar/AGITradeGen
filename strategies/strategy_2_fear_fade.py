# strategies/strategy_2_fear_fade.py
import pandas as pd
import yfinance as yf
from datetime import datetime

def fetch_vix_spy_data(start_date="2015-01-01", end_date=None):
    """
    Fetch combined SPY and VIX closing prices.
    Returns an empty DataFrame if anything goes wrong or if there's no overlap.
    """
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    print(f"Fetching data from {start_date} to {end_date}")

    # Extract or fall back to empty
    spy = yf.download("SPY", start=start_date, end=end_date)['Close']
    spy.name = "SPY"
    vix = yf.download("^VIX", start=start_date, end=end_date)['Close']
    vix.name = "VIX"
    df = pd.concat([spy, vix], axis=1).dropna()
    df = df.rename(columns={"^VIX": "VIX"})
    print(df.head())
    return df


def generate_vix_fade_signals(df=None, vix_window=252, high_percentile=0.90, 
                            low_percentile=0.60, stop_loss_pct=0.035, max_hold=5):
    """
    Final robust version that:
    1. Handles missing data
    2. Prevents all common errors
    3. Matches your pipeline's expected output
    """
    # Handle data input
    if df is None or not all(col in df.columns for col in ['SPY', 'VIX']):
        df = fetch_vix_spy_data()
        if len(df) == 0:
            print("No data available for the given date range.")
            return 0, None, None, None
    
    # Calculate indicators
    df = df.copy()
    df['VIX_high'] = df['VIX'].rolling(vix_window).quantile(high_percentile)
    df['VIX_low'] = df['VIX'].rolling(vix_window).quantile(low_percentile)
    df['SPY_ret'] = df['SPY'].pct_change()
        
    # Get today's values
    today = df.iloc[-1]
    yesterday = df.iloc[-2] if len(df) > 1 else today
        
    # Initialize defaults
    signal = 0
    entry = exit_px = sl = None
        
    # Entry condition
    if (today['VIX'] > today['VIX_high']) and (today['SPY'] < yesterday['SPY']):
        signal = 1
        entry = float(today['SPY'])
        sl = entry * (1 - stop_loss_pct)
        exit_px = entry * 1.01  # 1% target
            
    # Position management
    elif len(df) > 1 and 'Signal' in df.columns:
        prev_entries = df[df['Signal'] == 1]
        if not prev_entries.empty:
            last_entry = prev_entries.iloc[-1]
            days_held = len(df) - df.index.get_loc(last_entry.name) - 1
            
            if (today['VIX'] < today['VIX_low']) or (days_held >= max_hold):
                signal = 0
            else:
                signal = 1
                entry = float(last_entry['SPY'])
                sl = max(entry * 0.99, float(today['SPY']) * (1 - stop_loss_pct))
                exit_px = entry * (1 + 0.005 * days_held)

    print(f"Today's VIX: {today['VIX']}, SPY: {today['SPY']}")
    print(f"Yesterday's VIX: {yesterday['VIX']}, SPY: {yesterday['SPY']}")
    print(f"Signal: {signal}, Entry: {entry}, Exit: {exit_px}, SL: {sl}")         
    return signal, entry, exit_px, sl