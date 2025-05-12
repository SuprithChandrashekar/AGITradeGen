import openai
import pandas as pd
import numpy as np
import os 
import ast
import traceback

from backtesting import Strategy
from backtesting.test import SMA
from backtesting.lib import crossover

import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",  # Or another compatible Gemini model
    api_key=os.getenv("OPENAI_API_KEY"),  # Replace with your Gemini API key
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.5,
)

# Prompt to generate a backtesting.py-compatible strategy
strategy_prompt = PromptTemplate.from_template("""
### SYSTEM
You are a professional quant trading developer.  
You must return clean, valid, and runtime-safe Python code for a strategy function.

Specs:
• Define one function: `add_signal(df)`  
• df has columns: 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume' (5-min bars)  
• Use only pandas (pd) and numpy (np) — already imported  
• Modify df in-place by adding `df['signal']` with:
  → 1 = Buy 0 = Hold -1 = Sell  
• The code must be:
  - Deterministic (no randomness)
  - Syntax-safe (parseable by `ast.parse`)
  - Runtime-safe: **no variable may be referenced unless unconditionally defined**
  - One statement per line (no chaining with commas/semicolons)
  - No printing, I/O, or external libraries  
• Never end a line with a binary operator (e.g. `&`, `+`, `-`)  
  unless the full logical expression continues properly within parentheses.

Sanity Checks:
✅ Each temporary variable (`up_days`, `signals`, etc.) must be  
   assigned *before it is referenced*, unconditionally.  
✅ Mentally simulate: will this raise UnboundLocalError, KeyError, or NaN bugs?  
   If yes — fix before outputting.

Format:
1️⃣ One line: `Explanation: <describe strategy in ≤100 words>`  
2️⃣ Then a single fenced Python block with only the function  
3️⃣ No output or text after the code block.

### USER
Let’s step through designing a safe, effective intraday strategy for {ticker}.  
Explain in one line, then return the complete function inside a Python code block.
The function must be executable and deterministic using only Pandas/Numpy.
""")


chain = strategy_prompt | llm

def generate_strategy(df, user_prompt="Create a profitable strategy on TSLA"):
    if "on" in user_prompt:
        ticker = user_prompt.split("on")[-1].strip().upper()
    else:
        ticker = "SPY"

    try:
        response = (strategy_prompt | llm).invoke({"ticker": ticker})
        text = response["text"] if isinstance(response, dict) else str(response)

        # Extract description and code
        if "```" in text:
            parts = text.split("```")
            description = parts[0].strip()
            code_block = next((part for part in parts if "df" in part and "signal" in part), "")
            code = code_block.replace("python", "").strip()
            description = ast.literal_eval(f"'''{description}'''")
            code = ast.literal_eval(f"'''{code}'''")
            return code, description
        else:
            return "# No code block found", text.strip()

    except Exception as e:
        print(f"[ERROR] Strategy signal generation failed: {e}")
        return "# Error", "Strategy generation failed"
 


def execute_strategy(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    Executes the strategy code that modifies the DataFrame by adding a 'signal' column.
    Assumes the code defines a function `add_signal(df)` or similar.
    """
    local_env = {"df": df.copy(), "np": np, "pd": pd}
    
    try:
        exec(code, local_env)
        for val in local_env.values():
            if callable(val) and val.__name__.startswith("add_signal"):
                val(local_env["df"])
                break
        else:
            raise RuntimeError("No function named add_signal(...) found in strategy code.")
        
        return local_env["df"]

    except Exception as e:
        print(f"[ERROR] Failed to execute strategy code: {e}")
        raise

improvement_prompt = PromptTemplate.from_template("""
You are a quantitative trading strategist.

You are given:
- A Python strategy function that modifies a DataFrame to add a 'signal' column for trading (1=buy, 0=hold, -1=sell).
- A backtest performance summary of this strategy.
- You may assume the OHLCV data used is 5-minute bars for {ticker} over the last 5 trading days.

{input}                     

Your task:
1. Analyze the weaknesses in the strategy and explain what could be improved.
2. Then return a revised version of the Python strategy that:
   - Still adds a 'signal' column to the DataFrame.
   - Uses only numpy/pandas logic.
   - Avoids complex external dependencies or stateful classes.
   - Focuses on making the strategy more profitable, less volatile, or smarter.
                                                  
The strategy must be executable and deterministic using only Pandas/Numpy.
                                                  
The strategy must be executable and deterministic using only Pandas/Numpy.

You must return:
1. A brief explanation of your strategy
2. A function named `add_signal(df)` in a Python code block (```python) that modifies the DataFrame in-place to add the `signal` column.
                                               
IMPORTANT:
Ensure that any Series assigned to `df["signal"]` uses the same index as `df`. Example:
```python
df["signal"] = pd.Series(logic_array, index=df.index)""")

def improve_strategy(
    df,
    code: str,
    results_str: str,
    ticker: str = "TSLA",
    historical_context: str | None = None,          
):
    """
    Feed the original code & back‑test (plus optional historic champion)
    into the LLM and return (new_code, explanation).
    """
    llm_prompt = """You are an algorithmic‑trading Python expert.
Improve the following strategy while keeping it executable.

{input_block}

Return ONLY valid Python code that defines `add_signal(df)`.
"""

    # ╭─ optional champion block ─╮
    hist_block = (
        "\n\n# —— HISTORICAL TOP STRATEGY ——\n"
        f"{historical_context}\n"
        "# ————————————————————————————"
        if historical_context else ""
    )

    input_block = (
        f"STRATEGY CODE:\n```python\n{code}\n```\n\n"
        f"BACKTEST RESULTS:\n{results_str}{hist_block}"
    )

    prompt = llm_prompt.format(input_block=input_block)

    # ▶️  your own LLM call goes here  ◀️
    # chain = improvement_prompt | llm
    # response = chain.invoke({"ticker": ticker, "input": prompt})
    # text = response["text"] if isinstance(response, dict) else str(response)
    text = "```python\n# mock improved code\ndef add_signal(df):\n    df['signal']=0\n    return df\n```"  # placeholder

    # ── extract code block ─────────────────────────────────────────────
    if "```" in text:
        parts = text.split("```")
        code_block = next((p for p in parts if "def" in p or "df[" in p), "")
        improved_code = code_block.replace("python", "").strip()
        explanation   = parts[0].strip()
        return improved_code, explanation

    return "# No improved code returned", text.strip()