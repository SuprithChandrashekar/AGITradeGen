"""
AI Strategy Supervising Analyst Agent
Ranks historical strategies and optionally feeds the best back for further improvement.
"""

from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ─── Configuration ────────────────────────────────
load_dotenv()

REPORT_FILE = "strategy_results.xlsx"
TOP_N       = 3
WEIGHTS     = {"return": 0.35, "sharpe": 0.35, "drawdown": 0.15, "trades": 0.15}

LLM = ChatOpenAI(
    model       = os.getenv("LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"),
    api_key     = os.getenv("OPENAI_API_KEY"),
    base_url    = "https://integrate.api.nvidia.com/v1",
    temperature = 0.4,
)

def get_logger(module_name):
    DEBUG_LEVEL = os.environ.get("DEBUG_LEVEL", "info").lower()
    level_priority = {"verbose": 0, "info": 1, "warn": 2, "error": 3}

    def log(level, tag, message, data=None, symbol=None, line=None):
        if level_priority[level] < level_priority.get(DEBUG_LEVEL, 1):
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        origin = f"[{module_name.upper()}]"
        symbol_str = f" - Called from {symbol}:{line}" if symbol and line else ""
        try:
            if data is not None:
                if isinstance(data, (dict, list)):
                    data_str = json.dumps(data, default=str)
                else:
                    data_str = str(data)
                message = f"{message} - {data_str}"
        except Exception:
            message = f"{message} - [ERROR SERIALIZING DATA]"
        print(f"[{timestamp}] {origin} {tag.upper()}{symbol_str} - {message}")
    return log

PROMPT = PromptTemplate.from_template("""
You are a senior quantitative strategist.

For each of the {top_n} strategy records below (JSON list), return:
  • `strategy_id`        – its timestamp  
  • `summary`            – ≤30 words on why it outperformed  
  • `tweak`              – one concrete improvement  
  • `sl_pct`, `tp_pct`   – stop-loss / take-profit in %  
  • `exp_return_3mo`     – expected 3-month return in %  
  • `confidence`         – 0–1 score  

DATA:
{data}
""")

# ─── Helpers ──────────────────────────────────────
def _report_path() -> Path:
    return (Path(__file__).parent / ".." / "reports" / REPORT_FILE).resolve()

def _compute_score(df: pd.DataFrame) -> pd.Series:
    pct = lambda s: s.rank(pct=True)
    return (
          WEIGHTS["return"]   * pct(df["improved_return_pct"])
        + WEIGHTS["sharpe"]   * pct(df["improved_sharpe"])
        + WEIGHTS["drawdown"] * (1 - pct(df["improved_max_drawdown_pct"]))
        + WEIGHTS["trades"]   * pct(df["improved_n_trades"])
    )

# ─── Public API ───────────────────────────────────
def analyse(top_n: int = TOP_N) -> str:
    """Generate LLM-driven suggestions for the top-N historical strategies."""
    df     = pd.read_excel(_report_path())
    df["score"] = _compute_score(df)
    df_top = df.nlargest(top_n, "score")

    prompt = PROMPT.format(
        top_n=top_n,
        data=json.dumps(df_top.to_dict("records"), indent=2, default=str),
    )
    resp = LLM.invoke(prompt)
    text = resp.get("text", resp) if isinstance(resp, dict) else str(resp)

    # Persist suggestions
    out_dir = _report_path().parent / "insights"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"insights_{stamp}.json").write_text(text)

    return text

def best_historical(keep_cols: list[str] | None = None) -> dict[str, str]:
    """
    Return the highest-scoring historical run as a dict.
    Defaults to strategy_description, strategy_code, improved_backtest_results.
    """
    logger = get_logger("supervision")
    logger("info", "MODULE IN", "best_historical()", {
        "report_path": _report_path()
    })
    df = pd.read_excel(_report_path())
    logger("verbose", "MODULE IN", "Loaded DataFrame", {
        "rows": len(df),
        "columns": list(df.columns)
    })
    df["score"] = _compute_score(df)
    row = df.loc[df["score"].idxmax()]

    champ_ctx = {
        "code": row["orig_code"],
        "description": row["orig_desc"],
        "metrics": {
            "sharpe": row["improved_sharpe"],
            "return": row["improved_return_pct"]
        }
    }
    logger("info", "MODULE OUT", "best_historical()", {
        "champ_ctx": champ_ctx
    })
    
    
    return champ_ctx

# ─── CLI ───────────────────────────────────────────
if __name__ == "__main__":
    print(analyse())
