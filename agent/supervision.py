"""
AI Strategy Supervising Analyst Agent
Ranks historical strategies and optionally feeds the best back for further improvement.
"""

from __future__ import annotations
import json, os
from datetime import datetime
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
        data=json.dumps(df_top.to_dict("records"), indent=2),
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
    df = pd.read_excel(_report_path())
    df["score"] = _compute_score(df)
    row = df.loc[df["score"].idxmax()]

    cols = keep_cols or [
        "strategy_description",
        "strategy_code",
        "improved_backtest_results",
    ]
    return {c: str(row[c]) for c in cols if c in row}

# ─── CLI ───────────────────────────────────────────
if __name__ == "__main__":
    print(analyse())
