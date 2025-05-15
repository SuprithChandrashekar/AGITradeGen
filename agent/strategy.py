import openai
import pandas as pd
import numpy as np
import os 
import ast
import traceback
import re

from backtesting import Strategy
from backtesting.test import SMA
from backtesting.lib import crossover
from agent.supervision import best_historical

import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage

from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
import json
import traceback
import os

def get_logger(module_name):
    DEBUG_LEVEL = os.environ.get("DEBUG_LEVEL", "info").lower()
    level_priority = {"verbose": 0, "info": 1, "warn": 2, "error": 3}

    def log(level, tag, message, data=None, symbol=None, line=None):
        if level_priority[level] < level_priority.get(DEBUG_LEVEL, 1):
            return
        timestamp = datetime.utcnow().isoformat() + "Z"
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


llm = ChatOpenAI(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",  # Or another compatible Gemini model
    api_key=os.getenv("OPENAI_API_KEY"),  # Replace with your Gemini API key
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.1,
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
• Avoid combining too many strict filters (like RSI < 30 AND Close > ATR band) that rarely happen together.
• Your goal is to generate signals that lead to **at least 5-15 trades** on 5-day, 5-minute data.


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
        if not is_valid_python(code):
            raise SyntaxError("LLM-generated strategy code is invalid. Skipping execution.")
        exec(code, local_env)
        for val in local_env.values():
            if callable(val) and val.__name__.startswith("add_signal"):
                val(local_env["df"]) 
                break
        else:
            raise RuntimeError("No function named add_signal(...) found in strategy code.")
        
        result_df = local_env["df"]
        if 'signal' in result_df.columns:
            print("[DEBUG - Signals]", result_df['signal'].value_counts(dropna=False).to_dict())
        else:
            print("[ERROR] 'signal' column missing after execution.")


        return result_df

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
df["signal"] = pd.Series(logic_array, index=df.index)

⚠️ Avoid combining conditions that rarely co-occur (e.g., RSI < 30 AND SMA crossover AND ATR bands). These eliminate signals entirely.

✅ Ensure your strategy generates at least 5–15 trades on 5-minute TSLA data over 5 days.

💡 Prefer looser filters like:
- RSI > 50 or < 60
- MA5 vs MA20 crossover
- Volatility filters using rolling std or simple ATR bands

DO NOT generate strategies that result in df['signal'] being all 0.
If you're unsure, default to a basic RSI-based momentum strategy that triggers trades.
                                                                                                                                                     
Please output:

1. A brief explanation of your changes.
2. Your improved Python function wrapped in a fenced code block, e.g.:
                                                  

```python
def add_signal(df):
    # ...
    return df
                                                  """)

import re
import ast
from typing import Optional, Tuple
champ = best_historical()
print(champ)
hist_ctx = champ["code"] + "\n\n# Explanation:\n" + champ["description"]

def sanitize_code(
    code: str,
    results_str: str,
    ticker: str = "TSLA",
    historical_context=hist_ctx,
) -> Tuple[str, str]:
    """
    Calls the LLM to rewrite `code` based on `results_str` (and optional history).
    Returns (improved_code, explanation).
    """
    # Build optional history block
    hist_block = (
        "\n\n# —— HISTORICAL TOP STRATEGY ——\n"
        f"{historical_context}\n"
        "# ————————————————————————————"
        if historical_context else ""
    )

    prompt_input = (
        f"STRATEGY CODE:\n```python\n{code}\n```\n\n"
        f"BACKTEST RESULTS:\n{results_str}{hist_block}"
    )
    try:
        chain = improvement_prompt | llm
        response = chain.invoke({"ticker": ticker, "input": prompt_input})

        # extract text
        if isinstance(response, dict):
            text = response.get("text", "")
        elif isinstance(response, AIMessage):
            text = response.content
        else:
            text = str(response)

        if not text:
            return "# No improved code returned", "LLM returned empty response."

        if "```" in text:
            parts = text.split("```")
            explanation, raw_code = parts[0].strip(), parts[1].replace("python", "").strip()
        else:
            m = re.search(r"(def add_signal\(.*?return df)", text, re.S)
            if m:
                raw_code = m.group(1).strip()
                explanation = text[: m.start()].strip()
            else:
                return "# No improved code returned", text.strip()

        # literal‐eval to unescape
        try:
            explanation = ast.literal_eval(f"'''{explanation}'''")
        except:
            pass
        try:
            raw_code = ast.literal_eval(f"'''{raw_code}'''")
        except:
            pass

        # ensure a signal column exists
        if "df['signal']" not in raw_code:
            raw_code += "\n    df['signal'] = 0\n    return df"

        # sanitize: extract only the function and rebuild its body
        m2 = re.search(r"(def add_signal\(.*)", raw_code, re.S)
        func_text = m2.group(1) if m2 else "def add_signal(df):\n    pass"

        lines = func_text.splitlines()
        sig = lines[0]
        body = [l for l in lines[1:] if not l.strip().startswith("return")]
        # detect indent
        indent = "    "
        if body and body[0].startswith((" ", "\t")):
            indent = re.match(r"^(\s+)", body[0]).group(1)
        body.append(f"{indent}return df")

        improved_code = "\n".join([sig] + body)
        return improved_code, explanation

    except Exception as e:
        print(f"[ERROR] LLM-based improvement failed: {e}")
        import traceback; traceback.print_exc()
        fallback = "def add_signal(df):\n    df['signal'] = 0\n    return df"
        return fallback, f"LLM failed to improve: {e}"

def is_valid_python(code_str):
    try:
        ast.parse(code_str)
        return True
    except SyntaxError as e:
        print(f"[SYNTAX ERROR] Invalid generated code: {e}")
        return False

def improve_strategy(
    df,
    code: str,
    results_str: str,
    ticker: str = "TSLA",
    historical_context: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Wrapper that sanitizes the LLM output and then injects noise-threshold
    and cooldown filters before returning final (code, explanation).
    """
    # inside improve_strategy, after sanitize_code():
    # --- 🔁 NEW: Capture signal stats ---
    signal_counts = df["signal"].value_counts(dropna=False).to_dict() if "signal" in df.columns else {}
    signal_info_str = f"\n\nSignal Distribution (Original): {signal_counts}"
    results_str = results_str + signal_info_str

    # --- 🔁 Continue as before ---
    improved_code, explanation = sanitize_code(code, results_str, ticker, historical_context)

    # inject filters
    m = re.search(r"(def add_signal\([^:]+:.*?^)(.*?)(^return df)", improved_code,
                flags=re.S | re.M)
    if m:
        sig, body, ret = m.group(1), m.group(2), m.group(3)
        indent = re.match(r"^(\s*)return df", ret, flags=re.M).group(1)

        filters = (
            f"{indent}# Ensure pct_change exists\n"
            # f"{indent}df['pct_change'] = df['Close'].pct_change().fillna(0)\n\n"
            f"{indent}# Noise threshold (±0.1%)\n"
            f"{indent}threshold = 0.001\n"
            f"{indent}valid_up = df['pct_change'].shift(1) > threshold\n"
            f"{indent}valid_down = df['pct_change'].shift(1) < -threshold\n"
            f"{indent}df.loc[~valid_up & (df['signal']==1), 'signal'] = 0\n"
            f"{indent}df.loc[~valid_down & (df['signal']==-1), 'signal'] = 0\n\n"
            f"{indent}# Cooldown: no consecutive signals\n"
            f"{indent}prev = df['signal'].shift(1).fillna(0).astype(int)\n"
            f"{indent}df.loc[df['signal']==prev, 'signal'] = 0\n"
        )

        improved_code = (
            improved_code[: m.start(2)]
            + body
            + filters
            + ret
            + improved_code[m.end(3) :]
        )
        cleaned = []
        for line in improved_code.splitlines():
            if "apply(" in line and "lambda" in line:
                # skip the faulty ATR/RSI lines
                continue
            cleaned.append(line)
        improved_code = "\n".join(cleaned)

    return improved_code, explanation