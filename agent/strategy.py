import openai
import pandas as pd
import os 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import ast

load_dotenv()

llm = ChatOpenAI(
    model="nvidia/llama-3.1-nemotron-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key= os.getenv("NVIDIA_API_KEY"),
    temperature=0.5,
)

# Prompt to generate a backtesting.py-compatible strategy
strategy_prompt = PromptTemplate.from_template("""
You are a highly skilled and competitive quantitative trading developer with a strong background in algorithmic trading and machine learning.

Your task is to create a **profitable intraday trading strategy** for the U.S. stock {ticker}, using 5-minute OHLCV data from the past 5 trading days.

You should first analyze the price action and identify exploitable short-term patterns (e.g., mean reversion, momentum, volatility breakouts).

Instructions:
1. Write a brief description of your trading logic and rationale.
2. Then write a full Python class that inherits from `backtesting.Strategy`, including:
   - `init(self)` to define and initialize indicators
   - `next(self)` to define trading logic and execution rules

⚠️ Do **not** use `.rolling()`, `.shift()`, or `pd.Series(...)` on `self.data.Close`. These operations will cause runtime errors.

✅ Instead, define all indicators using `self.I(...)`, for example:
```python
self.sma = self.I(SMA, self.data.Close, 20)
self.std = self.I(lambda x: pd.Series(x).rolling(20).std(), self.data.Close)
self.rsi = self.I(lambda x: 100 - (100 / (1 + x.pct_change().rolling(14).mean() / abs(x.pct_change().rolling(14).mean()))), self.data.Close)


Assume these imports are already provided:
```python
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
                                               
Return only:
A description of the strategy and the correspondingPython code inside a python ... block """)

chain = strategy_prompt | llm

def generate_strategy(df, user_prompt="Come up with a profitable strategy on SPY"): 
    # Extract ticker symbol from user prompt
    print(f"[INFO] Analyzing data for strategy generation...")
    if "on" in user_prompt:
        ticker = user_prompt.split("on")[-1].strip().upper()
    else:
        ticker = "SPY"

    try:
        # Run Langchain chain
        print(f"[INFO] Building strategy for {ticker}...")
        response = chain.invoke({"ticker": ticker})
        text = response["text"] if isinstance(response, dict) else str(response)

        if "```" in text:
            parts = text.split("```")
            description = parts[0].strip()
            code_block = next((part for part in parts if "class" in part), "# No valid strategy code returned")
            code = code_block.replace("python", "").strip()
            code = ast.literal_eval(f"'''{code}'''")
        else:
            description = text.strip()
            code = "# No code block returned by LLM."
        description = ast.literal_eval(f"'''{description
        }'''")
        print(f"[INFO] Strategy built!")
        return code, description

    except Exception as e:
        print(f"[ERROR] LLM strategy generation failed: {e}")
        return "# LLM generation error", "Strategy generation failed."