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
You are a highly skilled and competitive quantitative trading developer with a strong background in algorithmic trading and machine learning.

Your task is to create a **profitable intraday trading strategy** for the U.S. stock {ticker}, using 5-minute OHLCV data from the past 5 trading days.

You should first analyze the price action and identify exploitable short-term patterns (e.g., mean reversion, momentum, volatility breakouts).

Instructions:
1. Write a brief description of your trading logic and rationale.
2. Then write a full Python class that inherits from `backtesting.Strategy`, including:
   - `init(self)` to define and initialize indicators
   - `next(self)` to define trading logic and execution rules



Assume these imports are already provided:
```python
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
                                               
Ensure the code is valid Python and compatible with the backtesting.py library. Avoid methods not designed for Strategy subclasses.

 --- DO NOT COPY THESE DIRECTLY ---
 These are just working examples to show you how to structure backtesting.py strategies.
 You are expected to come up with your own unique, profitable idea.
 These are meant to inspire you, not to be reused.

 --- Example Strategy 1: SMA Crossover ---
class SMACrossover(Strategy):
    def init(self):
        self.sma_fast = self.I(SMA, self.data.Close, 10)
        self.sma_slow = self.I(SMA, self.data.Close, 30)

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.sell()

 --- Example Strategy 2: Mean Reversion with Volatility Filter ---
class MeanReversionVolatility(Strategy):
    def init(self):
        self.sma = self.I(SMA, self.data.Close, 20)
        self.std = self.I(lambda x: pd.Series(x).rolling(20).std(), self.data.Close)

    def next(self):
        upper_band = self.sma + 2 * self.std
        lower_band = self.sma - 2 * self.std

        if self.data.Close[-1] < lower_band:
            self.buy()
        elif self.data.Close[-1] > upper_band:
            self.sell()

 --- Example Strategy 3: Momentum Breakout ---
class MomentumBreakout(Strategy):
    def init(self):
        self.sma = self.I(SMA, self.data.Close, 50)

    def next(self):
        recent_high = max(self.data.Close[-10:])
        recent_low = min(self.data.Close[-10:])

        if self.data.Close[-1] > recent_high and self.data.Close[-1] > self.sma[-1]:
            self.buy()
        elif self.data.Close[-1] < recent_low and self.data.Close[-1] < self.sma[-1]:
            self.sell()

 --- Example Strategy 4: Simple Trend Following ---
class TrendFollowing(Strategy):
    def init(self):
        self.sma = self.I(SMA, self.data.Close, 100)

    def next(self):
        if self.data.Close[-1] > self.sma[-1]:
            self.buy()
        elif self.data.Close[-1] < self.sma[-1]:
            self.sell()

 --- Now, create your own strategy below ---
 Choose any structure or logic you think would be profitable.
 Be creative. Do NOT repeat the above code — use your own logic.
                                
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
    
def safe_rsi(x, window=14):
    x = pd.Series(x)
    delta = x.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def safe_stddev(x, window=20):
    return pd.Series(x).rolling(window).std()

def validate_strategy_code(code: str) -> tuple[bool, str]:
    """
    Validates whether the generated strategy code runs without syntax or import errors.
    Returns (success: bool, class_name: str or error message)
    """
    local_env = {
        "Strategy": Strategy,
        "SMA": SMA,
        "RSI": safe_rsi,
        "StdDev": safe_stddev,
        "crossover": crossover,
        "pd": pd,
        "np": np
    }

    try:
        exec(code, globals(), local_env)
        strategy_class = next(
            (obj for obj in local_env.values()
             if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy),
            None
        )
        if strategy_class is None:
            return False, "[ERROR] No Strategy subclass found."

        return True, strategy_class.__name__
    except Exception as e:
        tb = traceback.format_exc()
        print("[❌] Validation failed:\n", tb)
        return False, str(e)

def generate_and_validate_strategy(df, prompt="Come up with a profitable strategy on SPY", max_attempts=5):
    for attempt in range(1, max_attempts + 1):
        print(f"\n[🔁 Attempt {attempt}] Generating strategy...")
        code, description = generate_strategy(df, prompt)

        is_valid, info = validate_strategy_code(code)
        if is_valid:
            print(f"[✅] Strategy '{info}' passed validation!")
            return code, description

        print(f"[⚠️] Invalid strategy, retrying. Reason: {info}")

    raise RuntimeError(f"Failed to generate a valid strategy after {max_attempts} attempts.")