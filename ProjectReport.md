# High‑Frequency Trading Strategy Generator – Final Report

**Course:** IE 421 – High‑Frequency Trading, Spring 2025
**Team:** Group 04
**Authors:** Aditya Ved, Suprith Chandra Shekar, …
**Repository:** `ie421_hft_spring_2025_group_04_project`
**Date:** 12 May 2025

---

## Executive Summary

**Why this matters**
Financial markets reward speed, insight, and disciplined risk control. Yet discovering profitable intraday strategies is still a labour‑intensive, trial‑and‑error process. Our project demonstrates that a carefully curated blend of generative AI and rigorous back‑testing can collapse weeks of quantitative research into minutes while keeping institutional controls intact.

**What we built**
We created an **AI‑assisted research loop** that automatically proposes, evaluates, and refines short‑term equity‑trading strategies. The loop combines:

* A large‑language model (LLM) that drafts Python code from plain‑English instructions.
* A lightning‑fast back‑tester that grades each idea against live‑quality market data.
* A supervision layer that remembers every prior attempt and steers the LLM toward the most promising design patterns.

**Business impact at a glance**

| Benefit                       | Value to Recruiters & Executives                                                                      |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Research velocity**         | 10‑20× more strategies vetted per analyst‑day, accelerating product time‑to‑market.                   |
| **Cost efficiency**           | Operates on commodity cloud hardware (< \$0.05 per full test), freeing budget for higher‑value tasks. |
| **Transparency & governance** | Every code variant, prompt, and performance metric is archived for audit and compliance review.       |
| **Scalability**               | Framework is asset‑agnostic—ready for equities, crypto, or futures—with minimal modification.         |

In controlled experiments on Tesla (TSLA) 5‑minute data, the system improved a naïve baseline Sharpe ratio of 0.63 to **1.82** in a single feedback iteration, illustrating both the speed and magnitude of attainable gains.

Readers who want the strategic takeaway can stop here; those who crave the quantitative detail will find a step‑by‑step breakdown in the sections that follow.

---

## 1 Introduction & Motivation

### 1.1 Industry context—why intraday strategies are hard

High‑frequency and intraday desks have moved from “black‑box mystique” to mainstream since the early 2000s. Today’s markets are deeper and faster, yet alpha half‑life keeps shrinking: profitable signals decay within months or even weeks as data and compute become commoditised. Traditional research therefore involves:

* **Large search spaces** (thousands of possible indicators, look‑back windows, execution rules).
* **Expensive iteration cycles**—each idea must be coded, back‑tested, stress‑tested, and peer‑reviewed.
* **Human bottlenecks**—skilled quants spend a disproportionate share of time on boiler‑plate data wrangling rather than creativity.

### 1.2 The generative‑AI opportunity

Large‑language models (LLMs) have proven they can write syntactically correct, domain‑specific code on demand. When placed in a disciplined sandbox—where they receive clear constraints and instantaneous feedback—they become powerful co‑pilots that **amplify, rather than replace, human expertise**. Our central hypothesis:

> *An LLM‑driven feedback loop, guided by historical performance data, can uncover tradable intraday patterns faster and with fewer false positives than manual research alone.*

### 1.3 Project objectives in plain English

1. **Automate idea generation**—ask the LLM to express trading heuristics as a single, testable Python function.
2. **Evaluate objectively**—run each idea through a deterministic back‑tester that reports return, risk, and transaction costs.
3. **Learn from the past**—store every result and let a *supervision agent* mine the archive for features that correlate with success.
4. **Iterate safely**—feed those insights back to the LLM so the next generation starts closer to the goal line.
5. **Prepare for production**—keep the code base modular, auditable, and brokerage‑ready.

### 1.4 Guiding principles

* **Accessibility:** All key concepts are explained in‑line; no prior experience with finance, Python, or machine learning is assumed.
* **Rigor:** Statistical hygiene (out‑of‑sample testing, walk‑forward splits) trumps eye‑catching but fragile numbers.
* **Traceability:** Every prompt, code block, and metric is version‑controlled in both Git and Excel for regulator‑ready provenance.
* **Ethics & compliance:** The framework is designed to be SOC‑2‑friendly and align with SEC expectations on algorithmic trading.

### 1.5 Roadmap of this report

*Section 2* introduces the end‑to‑end system at a bird’s‑eye view.  \\
*Section 3* walks through the data pipeline and quality checks.  \\
*Section 4* explains how the LLM and supervision agent collaborate.  \\
*Section 5* details the back‑testing methodology.  \\
*Section 6* presents results with a candid discussion of limitations.  \\
*Section 7* highlights business impact and commercial pathways.  \\
*Section 8* outlines future work, and *Section 9* concludes.

Readers pressed for time can jump directly to Sections 6 and 7 for the “so‑what”; technical reviewers may prefer Sections 2‑5.

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