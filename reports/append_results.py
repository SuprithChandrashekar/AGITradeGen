import os
import re
import pandas as pd
from datetime import datetime

def append_results(report_row: dict, filename="strategy_results.xlsx"):
    """
    Appends a new report row to ./reports/filename.
      • Normalizes timestamp → datetime
      • Renames raw keys to snake_case
      • Flattens backtest metrics (string or dict) into numeric cols
      • Extracts basic hyper-parameters from code strings
      • Splits SMA-windows list into separate columns
    """

    # ─── 1) prepare output folder & path ───────────────────────────────────────
    reports_dir = os.path.dirname(os.path.abspath(__file__))
    out_path    = os.path.join(reports_dir, "strategy_results.xlsx")

    # ─── 2) normalize timestamp ───────────────────────────────────────────────
    ts = report_row.get("timestamp")
    if not ts:
        report_row["timestamp"] = datetime.now()
    else:
        report_row["timestamp"] = pd.to_datetime(ts)

    # ─── 3) rename incoming fields to snake_case ──────────────────────────────
    rename_map = {
        "strategy_description":           "orig_desc",
        "strategy_code":                  "orig_code",
        "backtest_results":               "orig_metrics_raw",
        "results":                        "orig_metrics_raw",
        "improved_strategy_description":  "improved_desc",
        "improved_strategy_code":         "improved_code",
        "improved_backtest_results":      "improved_metrics_raw",
        "results2":                       "improved_metrics_raw",
    }
    for old, new in rename_map.items():
        if old in report_row:
            report_row[new] = report_row.pop(old)

    # ─── 4) helper: extract numeric metrics from raw string or dict ──────────
    def extract_metrics(raw):
        """
        Extract performance metrics from a result string or dict.
        Ensures all expected keys are present even if missing in raw input.
        """
        mapping = {
            "Initial Capital": "initial_capital",
            "Final Capital": "final_capital",
            "Total Return": "return_pct",
            "Sharpe Ratio": "sharpe",
            "Max Drawdown": "max_drawdown_pct",
            "Total Trades": "n_trades",
            "Fee Cost": "commission"
        }

        out = {v: None for v in mapping.values()}  # Default all fields to None

        if isinstance(raw, dict):
            for label, key in mapping.items():
                for k, val in raw.items():
                    if label in k:
                        try:
                            out[key] = float(val)
                        except:
                            out[key] = None
        elif isinstance(raw, str):
            for label, key in mapping.items():
                import re
                pat = re.compile(rf"{label}.*?:\s*(-?\d+(?:\.\d+)?|nan|inf)", re.IGNORECASE)
                match = pat.search(raw)
                if match:
                    val = match.group(1)
                    try:
                        out[key] = float(val) if val.lower() not in ("nan", "inf") else 0.0
                    except:
                        out[key] = None
        return out

    # ─── 5) flatten both original & improved metrics ──────────────────────────
    for side in ("orig", "improved"):
        raw_key = f"{side}_metrics_raw"
        if raw_key in report_row:
            mets = extract_metrics(report_row.pop(raw_key))
            for k, v in mets.items():
                report_row[f"{side}_{k}"] = v

    # ─── 6) helper: parse hyper-parameters from code ─────────────────────────
    def parse_hyperparams(code: str):
        p = {}
        # rolling(window=XX)
        wins = re.findall(r"rolling\(window=(\d+)\)", code)
        if wins:
            p["sma_windows"] = list(map(int, wins))
        # pct_diff thresholds
        m = re.search(r"pct_diff(?:_short)?\s*<\s*-?([\d\.]+)", code)
        if m:
            p["entry_threshold_pct"] = float(m.group(1))
        m = re.search(r"pct_diff(?:_short)?\s*>\s*([\d\.]+)", code)
        if m:
            p["exit_threshold_pct"] = float(m.group(1))
        # bollinger std-dev
        m = re.search(r"std_dev_bb\s*=\s*([\d\.]+)", code)
        if m:
            p["bb_std_dev"] = float(m.group(1))
        # position size factor
        m = re.search(r"position_size\s*=.*\*\s*([\d\.]+)", code)
        if m:
            p["position_size_factor"] = float(m.group(1))
        return p

    # ─── 7) extract hyper-parameters for both variants ────────────────────────
    for side in ("orig", "improved"):
        key = f"{side}_code"
        if key in report_row:
            params = parse_hyperparams(report_row[key])
            # split sma_windows list into separate columns
            wins = params.pop("sma_windows", [])
            if len(wins) >= 2:
                report_row[f"{side}_sma_short_window"] = wins[0]
                report_row[f"{side}_sma_long_window"]  = wins[1]
            if len(wins) >= 3:
                report_row[f"{side}_bb_window"]       = wins[2]
            # other params
            for pk, pv in params.items():
                report_row[f"{side}_{pk}"] = pv

    # ─── 8) build DataFrame, append to existing, and save ────────────────────
    df_new = pd.DataFrame([report_row])
    if os.path.exists(out_path):
        df_old = pd.read_excel(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(out_path, index=False)
