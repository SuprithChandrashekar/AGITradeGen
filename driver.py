#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.stats.power import TTestPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# Path to your results file (adjust if needed)
RESULTS_XLSX = Path(__file__).parent / "reports" / "strategy_results.xlsx"

def run_main():
    """Call main.py and wait for it to finish."""
    print("→ Running main.py …")
    subprocess.run(
        ["python", str(Path(__file__).parent / "main.py")],
        check=True
    )

def compute_sample_sizes():
    """
    Load the latest improved_return_pct from the Excel, then
    compute required n for:
      1) detecting a mean return > 0 (80% power, α=0.05)
      2) detecting win-rate p>0.5 (80% power, α=0.05)
    """
    df = pd.read_excel(RESULTS_XLSX)
    # clean
    data = pd.to_numeric(df["improved_return_pct"], errors="coerce").dropna()
    if len(data) == 0:
        return float("inf"), float("inf")
    # mean test
    mean_ret = data.mean()
    std_ret  = data.std(ddof=1)
    # avoid division by zero
    if std_ret == 0:
        n_mean = 0.0
    else:
        d = mean_ret / std_ret  # Cohen's d
        tpow = TTestPower()
        n_mean = tpow.solve_power(
            effect_size=abs(d),
            alpha=0.05,
            power=0.8,
            alternative="two-sided"
        )
    # proportion test
    p0 = 0.5
    p1 = (data > 0).mean()  # empirical win-rate
    h = proportion_effectsize(p1, p0)  # Cohen’s h
    pown = NormalIndPower()
    n_prop = pown.solve_power(
        effect_size=abs(h),
        alpha=0.05,
        power=0.8,
        alternative="larger"
    )
    return n_mean, n_prop

def main_loop():
    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== Iteration #{iteration} ===")
        try:
            run_main()
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] main.py execution failed: {e}")
            time.sleep(2)
            continue

        try:
            n_mean, n_prop = compute_sample_sizes()
            print(f"→ Required runs to detect mean≠0 (80% power): {n_mean:.1f}")
            print(f"→ Required runs to detect win-rate>50% (80% power): {n_prop:.1f}")
        except Exception as e:
            print(f"[ERROR] Could not compute sample sizes: {e}")
            # If there’s an issue (such as file-lock), wait and try again.
            time.sleep(2)
            continue

        # stop when both sample-size requirements drop to (or below) 1 run
        if n_mean <= 1.0 and n_prop <= 1.0:
            print("\n✅ Both sample‐size requirements are ≤1; stopping loop.")
            break

        # Short pause to avoid file-lock issues
        time.sleep(1)

if __name__ == "__main__":
    # Start the animated plot in a separate process.
    anim_proc = subprocess.Popen(["python", str(Path(__file__).parent / "plot_animated_runs.py")])
    
    try:
        main_loop()
    finally:
        # When main_loop() ends, terminate the animation process.
        anim_proc.terminate()