import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

"""
def fetch_intraday_data(name: str = "SPY", interval: str = '5m', start: str = None, end: str = None, use_cache: bool = True) -> pd.DataFrame:
    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)

    # If no dates provided, default to last 5 days
    if start is None:
        start_dt = datetime.now() - timedelta(days=5)
    else:
        start_dt = pd.to_datetime(start)

    if end is None:
        end_dt = datetime.now()
    else:
        end_dt = pd.to_datetime(end)

    interval_mins = int(interval.replace('m', '')) if 'm' in interval else 1
    # Format dates
    start_str = start_dt.strftime('%Y%m%d')
    end_str = end_dt.strftime('%Y%m%d')

    # Find matching cached file
    matching_cache = None
    for file in os.listdir(cache_dir):
        if not file.endswith(".csv"):
            continue
        parts = file.replace(".csv", "").split("_")
        if len(parts) != 4:
            continue
        ticker, file_start, file_end, file_interval = parts
        if ticker != name or int(file_interval) != interval_mins:
            continue
        file_start_dt = datetime.strptime(file_start, "%Y%m%d")
        file_end_dt = datetime.strptime(file_end, "%Y%m%d")

        # Check if file partially or fully covers the requested range
        if (file_start_dt <= end_dt and file_end_dt >= start_dt):
            matching_cache = file
            break

    if matching_cache:
        cache_path = os.path.join(cache_dir, matching_cache)
        print(f"[INFO] Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["Datetime"])
    else:
        df = pd.DataFrame()

    # Determine if data needs to be fetched
    need_fetch = df.empty
    if not df.empty:
        cached_start = df["Datetime"].min()
        cached_end = df["Datetime"].max()

        if start_dt < cached_start or end_dt > cached_end:
            need_fetch = True

    if need_fetch:
        print(f"[INFO] Fetching missing YFinance data for {name} interval={interval}")
        fetched_df = yf.download(name, interval=interval, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), progress=False)
        if fetched_df.empty:
            raise ValueError(f"No data returned for {name} from {start_dt} to {end_dt}")
        fetched_df = fetched_df.reset_index()

        # Merge if cache partially exists
        if not df.empty:
            df = pd.concat([df, fetched_df]).drop_duplicates(subset="Datetime").sort_values(by="Datetime").reset_index(drop=True)
        else:
            df = fetched_df

        # Save updated cache
        updated_start = df["Datetime"].min().strftime('%Y%m%d')
        updated_end = df["Datetime"].max().strftime('%Y%m%d')
        cache_filename = f"{name}_{updated_start}_{updated_end}_{interval_mins}.csv"
        cache_path = os.path.join(cache_dir, cache_filename)
        df.to_csv(cache_path, index=False)
        print(f"[INFO] Cache updated and saved to {cache_path}")
    else:
        print(f"[INFO] Cache fully covers requested range, no fetch needed.")

    # Clean and enforce correct types
    df = df[df["Close"] != "Close"]
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"[ERROR] Missing column: {col}")

    df.dropna(subset=required_cols, inplace=True)
    df = df.reset_index(drop=True)

    return df



"""
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