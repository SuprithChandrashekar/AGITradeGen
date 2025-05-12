# High‑Frequency Trading Strategy Generator – Final Report

**Course:** IE 421 – High‑Frequency Trading, Spring 2025
**Team:** Group 04
**Authors:** Aditya Ved, Suprith Chandra Shekar, James Cho, Adam Ahmed El Bahey
**Repository:** `ie421_hft_spring_2025_group_04_project`
**Date:** 12 May 2025

---

## Executive Summary

This project delivers an **AI‑driven research loop that discovers, tests, and iteratively improves intraday equity‑trading strategies** on 5‑minute bars.
Leveraging **large‑language models (LLMs)** for code synthesis, a vectorised **Python back‑tester** for rapid evaluation, and a lightweight **supervision agent** that mines historical runs to steer exploration, the system:

* Cuts human research time from days to minutes by automating signal‑engineering tasks.
* Achieves a **Sharpe ratio of ≈ 1.8** on out‑of‑sample tests for TSLA (improved strategy vs. 0.6 baseline).
* Provides a modular path to **live paper‑/real‑money execution via IBKR** with risk‑controlled order routing.
* Opens opportunities for **commercialisation as an AI quant‑research assistant** that scales to multi‑asset portfolios.

---

## 1  Introduction & Motivation

Algorithmic trading desks expend significant effort hand‑coding and validating new alpha. Recent breakthroughs in generative AI suggest that LLMs can draft syntactically correct, domain‑specific Python given tight constraints. Our hypothesis: *an LLM‑led research loop can accelerate strategy discovery while maintaining statistical rigour.*

The project objectives were therefore to:

1. **Generate deterministic trading logic** from plain‑English prompts using Gemini 1.5 Pro / GPT‑4o.
2. **Back‑test quickly** on cleaned intraday data to avoid data‑snooping biases.
3. **Close the feedback loop** by feeding performance diagnostics back into the LLM, spawning improved variants.
4. **Archive each generation** (code + results) for longitudinal analysis and supervisory guidance.
5. **Package the pipeline** for future integration with an Interactive Brokers execution layer.

---

## 2  System Architecture

```mermaid
flowchart TD
    subgraph LLM_Agent["LLM‑Based Strategy Agent"]
        P1(Prompt Template) -->|Ticker, Rules| G1>Gemini / GPT‑4]
        G1 -->|Python code<br/>add_signal(df)| Exec(Executor)
    end
    DF[Data Fetch & Clean] --> Exec
    Exec --> BT(Back‑tester)
    BT --> Log[(Results Excel)]
    Log --> Sup[Supervision Agent]
    Sup -->|Best Historic Variant| P1
```

**Key modules** (all under `agent/` unless noted):

| Layer         | File                            | Purpose                                                                            |
| ------------- | ------------------------------- | ---------------------------------------------------------------------------------- |
| Data          | `data_fetch.py`                 | Downloads 5‑min bars via *yfinance* with local caching.                            |
| Clean & Split | `data_split.py`                 | Performs RANSAC regression to drop outlier bars, supports LOO or fixed splits.     |
| Strategy LLM  | `strategy.py`                   | Crafts strict prompt → calls LLM → extracts safe `add_signal(df)` function.        |
| Executor      | `strategy.py::execute_strategy` | Injects LLM code via `exec`, appends `signal` to DataFrame.                        |
| Evaluator     | `eval.py`                       | Vectorised PnL engine with fees, drawdown, Sharpe, trade counts; Matplotlib plots. |
| Supervisor    | `supervison.py`                 | Scans Excel ledger (`reports/`) for best variants; informs next LLM prompt.        |
| Orchestrator  | `main.py`                       | End‑to‑end demo: fetch → clean → generate → back‑test → improve → log.             |

---

## 3  Data Pipeline

* **Universe:** Initial experiments focus on *Tesla (TSLA)*, but the code is parameterised for any Yahoo Finance ticker.
* **Granularity & Horizon:** 5‑minute OHLCV bars over the past 5 trading days (`period='5d'`).
* **Cleaning:** We observed micro‑structure noise and occasional zero‑volume prints. A **RANSAC regression filter** (Alg. 1) flags extreme residuals and returns a boolean `inliers` mask, preserving ±95 % of data while eliminating spikes.
* **Train/Test Split:** Default 80/20 chronological split ensures no look‑ahead bias. LOO CV is available for robustness checks.

---

## 4  Strategy Generation & Improvement

1. **Prompt Template**

   ```text
   You are a trading‑strategy generator…
     • Output clean Python with one function add_signal(df)
     • Use only pandas (pd) & numpy (np)
     • Deterministic, no I/O, no randomness
   ```
2. **LLM Call** – `ChatOpenAI` or `genai.GenerativeModel` constrained to 8192 tokens, temperature 0.3 for stability.
3. **Safety Checks** – Code is parsed via `ast.parse`; any exception triggers a retry.
4. **Execution** – Signals are added to the test set and back‑tested.
5. **Improvement Loop** – `improve_strategy()` feeds back the code‑block *and* performance string to the LLM, requesting a refined variant (Fig. 2).
6. **Logging** – Prompt, code, results, and commit hash are appended to `reports/results.xlsx` for future mining.

---

## 5  Back‑Testing Methodology

Our custom evaluator mirrors standard portfolio PnL maths:

```python
position = signal.shift().fillna(0)      # trade acts next bar
returns  = position * (df.Close.pct_change())
net      = returns - fee_per_trade * abs(signal.diff().fillna(0))
```

Metrics reported:

* **CAGR / Total Return**
* **Sharpe & Sortino Ratios (dailyised)**
* **Max Drawdown**
* **Win % & Trade Count**
* **Equity Curve** plotted for visual sanity.

Vectorisation keeps the 5‑day back‑test < 120 ms on a single core.

---

## 6  Experimental Results

> *Detailed numeric tables are embedded in `reports/2025‑05‑12_run001.xlsx`. Key highlights:*

| Variant                               |   Sharpe | Total Return | Trades |  Max DD |
| ------------------------------------- | -------: | -----------: | -----: | ------: |
| Baseline (LLM v1)                     |     0.63 |       +4.5 % |    24  |  ‑6.1 % |
| **Improved v2**                       | **1.82** |  **+14.7 %** |    31  |  ‑4.3 % |
| Supervisor‑selected (best of history) |     1.95 |      +16.2 % |    28  |  ‑4.0 % |

*Figure 1* shows the equity curves with 2‑σ confidence bands (see plot in repo). The **iterative loop clearly outperforms blind generation**, illustrating the value of performance‑aware prompting.

---

## 7  Business Value & Commercial Potential

* **Speed to Alpha:** Research cycles drop from *hours to minutes*, letting quants vet dozens of ideas per day.
* **Human‑in‑the‑Loop:** Analysts approve code before deployment, keeping compliance & risk teams happy.
* **Scalability:** Modular pipeline can parallelise across tickers / crypto / futures.
* **Cost Efficiency:** Runs on commodity cloud instances (< \$0.05 per strategy test).
* **IP Leverage:** The supervisor ledger becomes a proprietary knowledge base of profitable motifs—an asset in its own right.

---

## 8  Limitations & Risk Mitigation

| Category                  | Mitigation                                                                     |
| ------------------------- | ------------------------------------------------------------------------------ |
| **Data Snooping**         | Walk‑forward & LOO CV; hold‑out periods > 30‑days in future work.              |
| **LLM Hallucination**     | Strict `ast` validation; reject non‑deterministic constructs.                  |
| **Execution Slippage**    | Planned IBKR paper‑trading module with realistic tick‑level fill simulator.    |
| **Regulatory Compliance** | Manual sign‑off flow; audit trail via Excel + Git commits.                     |
| **Over‑fitting**          | Supervisor favours variants with higher out‑of‑sample Sharpe / lower turnover. |

---

## 9  Future Work

1. **Ensemble Allocator** – Combine four orthogonal strategies (volatility compression, fear‑factor fade, sentiment mismatch swing, rolling reversal) using a risk‑parity optimiser.
2. **Position Sizing Module** – Dynamic sizing via Kelly‑fraction approximations under max‑DD constraints.
3. **Paper → Live Execution** – `execution_broker.py` to route orders through **Interactive Brokers Gateway** with slippage models.
4. **Multi‑Asset Generalisation** – Auto‑select assets by liquidity & volatility screens; cluster task instances.
5. **Research Agent Autonomy** – Integrate Google **Agent Development Kit** for self‑directed hypothesis testing.

---

## 10  Conclusion

This project demonstrates that **generative AI can materially accelerate high‑frequency strategy research** while preserving quantitative discipline. By coupling prompt‑constrained code synthesis with automated back‑testing and a performance‑aware supervision layer, we achieved competitive Sharpe ratios on intraday TSLA data and laid the groundwork for production‑grade deployment.

The same architectural blueprint is transferrable to broader asset classes and offers a compelling **edge for both academic exploration and commercial trading desks**. Continued iteration along the outlined roadmap can turn this prototype into an end‑to‑end autonomous quant‑research platform.

---

## A Appendix: Key Parameters

| Symbol          | Default   | Description                                             |
| --------------- | --------- | ------------------------------------------------------- |
| `interval`      | `5m`      | Bar size requested from Yahoo Finance.                  |
| `period`        | `5d`      | Look‑back window for initial experiments.               |
| `fee_per_trade` | `0.001`   | Round‑trip cost fraction (approximates 10 bp each way). |
| `temperature`   | `0.3`     | LLM creativity; lower = safer code.                     |
| `capital`       | `$10 000` | Hypothetical starting equity in back‑tests.             |

---

*End of Report*
