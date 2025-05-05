import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_intraday_data(name: str = "SPY", interval: str = '5m', period: str = '5d', use_cache: bool = True) -> pd.DataFrame:
    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"{name}_{interval}_{period}.csv"
    cache_path = os.path.join(cache_dir, cache_filename)

    # Try to load from cache
    if use_cache and os.path.exists(cache_path):
        try:
            print(f"[INFO] Loading cached data from {cache_path}")
            df = pd.read_csv(cache_path, parse_dates=["Datetime"])
        except Exception as e:
            print(f"[WARNING] Cache loading failed: {e}, refetching...")
            use_cache = False  # fallback to fresh fetch

    if not use_cache:
        print(f"[INFO] Fetching YFinance data for {name} with interval={interval}, period={period}")
        df = yf.download(name, interval=interval, period=period, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {name} with interval={interval}, period={period}")
        df = df.reset_index()
        df.to_csv(cache_path, index=False)

    print("[INFO] Data fetched!")

    # Clean: drop bad string headers like row 0 with TSLA strings
    df = df[df["Close"] != "Close"]

    # Ensure column types are numeric
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"[ERROR] Missing column: {col}")

    # Drop rows with NaNs in required columns
    df.dropna(subset=required_cols, inplace=True)

    # Final cleanup
    df = df.reset_index(drop=True)
    return df