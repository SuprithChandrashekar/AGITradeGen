# strategies/strategy_1_volatility_breakout.py
def generate_volatility_compression_signals(df, bb_window: int = 20, percentile: float = 0.10, holding_period: int = 5):
    """
    Generates trading signals for TODAY only with:
    - Signal (0/1)
    - Entry price
    - Exit price
    - Stop-loss price
    
    Returns: Tuple of (signal, entry_price, exit_price, stop_loss)
    """
    df = df.copy()
    
    # Calculate indicators
    df['MA'] = df['Close'].rolling(window=bb_window).mean()
    df['STD'] = df['Close'].rolling(window=bb_window).std()
    df['Upper'] = df['MA'] + 0.1 * df['STD']
    df['Lower'] = df['MA'] - 0.1 * df['STD']
    df['Band Width'] = (df['Upper'] - df['Lower']) / df['MA']
    
    # Get scalar values for today
    today_bw = df['Band Width'].iloc[-1]
    today_close = df['Close'].iloc[-1].item()
    today_std = df['STD'].iloc[-1]
    today_lower = df['Lower'].iloc[-1]  

    # Get scalar threshold values
    bw_threshold = df['Band Width'].rolling(window=252).quantile(percentile).iloc[-1]
    bw_40th = df['Band Width'].rolling(window=252).quantile(0.40).iloc[-1]
    print(f"Threshold BW: {bw_threshold}, 40th BW: {bw_40th}")

    # Initialize default values
    signal = 0
    entry_price = None
    exit_price = None
    stop_loss = None
    
    # Check entry condition (all scalar comparisons)
    if today_bw < bw_threshold:
        signal = 1
        entry_price = float(today_close)
        exit_price = float(today_close + (today_std * 0.5))
        stop_loss = float(max(today_lower, today_close * 0.99))
    
    # Check existing position
    elif len(df) > 1 and 'Signal' in df.columns and df['Signal'].iloc[-2] == 1:
        if today_bw > bw_40th:
            signal = 0  # Exit signal
        else:
            # Maintain position
            signal = 1
            entry_price = float(df['Close'].iloc[-2])
            exit_price = float(entry_price + (df['STD'].iloc[-2] * 0.5))
            stop_loss = float(today_lower)
    
    return signal, entry_price, exit_price, stop_loss