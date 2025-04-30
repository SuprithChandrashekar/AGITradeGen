import openai
import pandas as pd
import numpy as np
import os 
import ast
import traceback

from backtesting import Strategy
from backtesting.test import SMA
from backtesting.lib import crossover


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="nvidia/llama-3.1-nemotron-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key= os.getenv("NVIDIA_API_KEY"),
    temperature=0.5,
)

# Prompt to generate a backtesting.py-compatible strategy
strategy_prompt = PromptTemplate.from_template("""
You are a quantitative trading developer.

Your task is to create a profitable intraday trading strategy for the U.S. stock {ticker}, using provided OHLCV data.

You will be given a Pandas DataFrame called `df` containing these columns:
- 'Datetime'
- 'Open', 'High', 'Low', 'Close', 'Volume'

Your goal is to analyze the data and populate a new column `signal` such that:
- `1` means "Buy"
- `0` means "Hold"
- `-1` means "Sell"
                                               
Before doing this, you must carefully analyze the given DataFrame create a strategy that is profitable and robust.              

The strategy must be executable and deterministic using only Pandas/Numpy.

You must return:
1. A brief explanation of your strategy
2. A function named `add_signal(df)` in a Python code block (```python) that modifies the DataFrame in-place to add the `signal` column.
                                               
IMPORTANT:
Ensure that any Series assigned to `df["signal"]` uses the same index as `df`. Example:
```python
df["signal"] = pd.Series(logic_array, index=df.index)""")


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

def improve_strategy(df, code: str, results_str: str, ticker="TSLA"):
    try:
        # Combine strategy code and backtest result
        prompt_input = {
            "ticker": ticker,
            "code": code,
            "results": results_str
        }
        full_input = (
            f"STRATEGY CODE:\n```python\n{code}\n```\n\n"
            f"BACKTEST RESULTS:\n{results_str}"
        )
        chain = improvement_prompt | llm
        response = chain.invoke({"ticker": ticker, "input": full_input})
        text = response["text"] if isinstance(response, dict) else str(response)

        if "```" in text:
            parts = text.split("```")
            explanation = parts[0].strip()
            code_block = next((p for p in parts if "def" in p or "df[" in p), "")
            improved_code = code_block.replace("python", "").strip()
            explanation = ast.literal_eval(f"'''{explanation}'''")
            improved_code = ast.literal_eval(f"'''{improved_code}'''")
            return improved_code, explanation
        else:
            return "# No improved code returned", text.strip()

    except Exception as e:
        print(f"[ERROR] Strategy improvement failed: {e}")
        return "# Error", "Improvement generation failed."