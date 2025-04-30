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

Your task is to create a profitable intraday trading strategy for the U.S. stock {ticker}, using 5-minute OHLCV data over the past 5 trading days.

You will be given a Pandas DataFrame called `df` containing these columns:
- 'Datetime'
- 'Open', 'High', 'Low', 'Close', 'Volume'

Your goal is to analyze the data and populate a new column `signal` such that:
- `1` means "Buy"
- `0` means "Hold"
- `-1` means "Sell"

The strategy must be executable and deterministic using only Pandas/Numpy.

You must return:
1. A brief explanation of your strategy
2. A Python code block (```python) that modifies the DataFrame in-place to add the `signal` column.""")


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