import os
import re
import pandas as pd
from datetime import datetime

def append_results(report_row: dict, filename="strategy_results.xlsx"):
    """
    Appends a new report row to an Excel file in ./reports/.
    - Adds/normalizes timestamp
    - Renames raw keys to snake_case
    - Flattens backtest_metrics dicts into numeric columns
    - Extracts basic hyper-parameters from code strings
    """

    # ─── 1) prepare output folder ────────────────────────────────────────────────
    base = os.path.dirname(__file__)
    reports_dir = os.path.join(base, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, filename)

    # ─── 2) timestamp ────────────────────────────────────────────────────────────
    ts = report_row.get("timestamp")
    if not ts:
        report_row["timestamp"] = datetime.now()
    elif isinstance(ts, str):
        report_row["timestamp"] = pd.to_datetime(ts)

    # ─── 3) rename raw fields ──────────────────────────────────────────────────
    rename_map = {
        "strategy_description":           "orig_desc",
        "strategy_code":                  "orig_code",
        "backtest_results":               "orig_metrics",
        "improved_strategy_description":  "improved_desc",
        "improved_strategy_code":         "improved_code",
        "improved_backtest_results":      "improved_metrics",
    }
    for old, new in rename_map.items():
        if old in report_row:
            report_row[new] = report_row.pop(old)

    # ─── 4) flatten metrics dict → individual columns ──────────────────────────
    def flatten_metrics(mdict, prefix):
        cols = {
            "Total Return (%)":   "return_pct",
            "Sharpe Ratio":       "sharpe",
            "Max Drawdown (%)":   "max_drawdown_pct",
            "Total Trades":       "n_trades",
            "Fee Cost":           "commission",
            "Initial Capital":    "initial_capital",
            "Final Capital":      "final_capital",
        }
        out = {}
        if isinstance(mdict, dict):
            for human, col in cols.items():
                if human in mdict:
                    val = mdict[human]
                    # strip trailing '%' if present
                    if isinstance(val, str) and val.endswith("%"):
                        val = float(val.strip("%"))
                    out[f"{prefix}{col}"] = val
        return out

    for prefix in ("orig_", "improved_"):
        key = f"{prefix}metrics"
        if key in report_row:
            metrics = report_row.pop(key)
            report_row.update(flatten_metrics(metrics, prefix))

    # ─── 5) extract hyper-parameters from code strings ───────────────────────────
    def parse_hyperparams(code: str):
        p = {}
        # find all rolling(window=XX)
        wins = re.findall(r"rolling\(window=(\d+)\)", code)
        if wins:
            # assume first is short-SMA, second is long-SMA (or BB window)
            p["sma_windows"] = list(map(int, wins))
        # entry threshold: pct_diff < -X
        m = re.search(r"pct_diff.*?<\s*-?([\d\.]+)", code)
        if m:
            p["entry_threshold_pct"] = float(m.group(1))
        # exit threshold: pct_diff > X
        m = re.search(r"pct_diff.*?>\s*([\d\.]+)", code)
        if m:
            p["exit_threshold_pct"] = float(m.group(1))
        # Bollinger std dev
        m = re.search(r"std_dev_bb\s*=\s*([\d\.]+)", code)
        if m:
            p["bb_std_dev"] = float(m.group(1))
        # position sizing factor "* 0.01"
        m = re.search(r"position_size\s*=.*\*\s*([\d\.]+)", code)
        if m:
            p["position_size_factor"] = float(m.group(1))
        return p

    for prefix in ("orig_", "improved_"):
        code_key = f"{prefix}code"
        if code_key in report_row:
            params = parse_hyperparams(report_row[code_key])
            for pk, pv in params.items():
                report_row[f"{prefix}{pk}"] = pv

    # ─── 6) build one-row DataFrame and append ─────────────────────────────────
    df_new = pd.DataFrame([report_row])
    if os.path.exists(out_path):
        df_old = pd.read_excel(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    # ─── 7) write back ───────────────────────────────────────────────────────────
    df.to_excel(out_path, index=False)
