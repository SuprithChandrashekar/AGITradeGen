# Code Snippet (Data Loading & Cleaning):

import pandas as pd
import numpy as np

# Define the path to the reports directory and file
from pathlib import Path
REPORT_FILE = "strategy_results.xlsx"
report_path = Path(__file__).parent / REPORT_FILE

# 1. Load the Excel file into a DataFrame
df = pd.read_excel(report_path)

# 2. Extract the improved strategy returns column and drop missing values
data = df['improved_return_pct'].dropna()
data = pd.to_numeric(data, errors='coerce').dropna()  # ensure numeric type

# 3. Compute summary statistics
n = len(data)
mean_return = data.mean()
std_return = data.std(ddof=1)  # sample standard deviation

# Code Snippet (T-Test):

import math
from scipy import stats

# 4. Perform a one-sample t-test on the mean of improved_return_pct
# H0: true mean = 0
res = stats.ttest_1samp(data, popmean=0.0)
t_statistic = res.statistic
p_value_two_sided = res.pvalue

# For a one-sided test (mean > 0), halve the two-sided p-value if mean_return is positive
if mean_return > 0:
    p_value_one_sided = p_value_two_sided / 2
else:
    p_value_one_sided = 1 - (1 - p_value_two_sided/2)  # essentially 1 - CDF for t if mean is negative

# Code Snippet (Proportion & CI):

# 5. Calculate proportion of positive returns
num_positive = (data > 0).sum()
prop_positive = num_positive / n

# 6. Compute 95% confidence interval for the proportion of positive returns
from math import sqrt
z = 1.96  # 95% confidence Z-score for normal approximation

# Wilson score interval (more accurate for proportions, especially with small n)
phat = prop_positive
center = (phat + z**2/(2*n)) / (1 + z**2/n)
margin = (z / (1 + z**2/n)) * math.sqrt(phat*(1-phat)/n + z**2/(4*n**2))
ci_lower = max(0.0, center - margin)
ci_upper = min(1.0, center + margin)

# Code Snippet (Power Analysis for Sample Size):
# 7. Calculate required sample size for detecting a 1% mean return with 80% power at 5% significance
effect = 1.0   # desired mean improvement to detect (1%)
alpha = 0.05   # 5% significance level (two-sided)
power = 0.80   # 80% power

# Using normal approximation formula:
z_alpha = 1.96    # z for 97.5th percentile (two-sided 5%)
z_beta = 0.84     # z for 80% power
sigma = std_return

n_required = ((z_alpha + z_beta) * sigma / effect) ** 2

# Round up to next whole number since sample size must be an integer
n_required = math.ceil(n_required)

# Final Python Module:

# reports/statistical_analysis.py

import math
from pathlib import Path
import pandas as pd
from scipy import stats

def main():
    # Load the strategy results Excel file
    report_path = Path(__file__).parent / "strategy_results.xlsx"
    if not report_path.exists():
        print("Error: strategy_results.xlsx not found in reports directory.")
        return
    df = pd.read_excel(report_path)
    
    # Clean and prepare data
    data = df['improved_return_pct'].dropna()
    data = pd.to_numeric(data, errors='coerce').dropna()
    n = len(data)
    if n == 0:
        print("No improved strategy results found for analysis.")
        return
    mean_return = data.mean()
    std_return = data.std(ddof=1)
    
    # One-sample t-test for mean return = 0
    t_stat, p_value_two_sided = stats.ttest_1samp(data, popmean=0.0)
    # Determine one-sided p-value for mean > 0 (if needed)
    if mean_return > 0:
        p_value_one_sided = p_value_two_sided / 2
    else:
        p_value_one_sided = 1 - (1 - p_value_two_sided/2)
    
    # 95% confidence interval for the mean
    dfree = n - 1
    t_crit = stats.t.ppf(0.975, dfree)  # two-sided 95% critical value
    ci_lower = mean_return - t_crit * (std_return / math.sqrt(n))
    ci_upper = mean_return + t_crit * (std_return / math.sqrt(n))
    
    # Proportion of positive returns and 95% CI (Wilson interval)
    num_positive = (data > 0).sum()
    prop_positive = num_positive / n
    z = 1.96  # 95% confidence level for normal approximation
    phat = prop_positive
    center = (phat + z**2/(2*n)) / (1 + z**2/n)
    margin = (z / (1 + z**2/n)) * math.sqrt(phat*(1-phat)/n + z**2/(4*n**2))
    ci_prop_lower = max(0.0, center - margin)
    ci_prop_upper = min(1.0, center + margin)
    
    # Sample size needed for 1% mean return detection (80% power, 5% alpha)
    effect = 1.0   # desired mean return to detect (in percent)
    alpha = 0.05
    power = 0.80
    # z for two-sided alpha and for power
    z_alpha = 1.96
    z_beta = 0.84
    sigma = std_return
    n_required = ((z_alpha + z_beta) * sigma / effect) ** 2
    n_required = math.ceil(n_required)  # round up
    
    # Print the statistical analysis results
    print(f"Total improved strategies analyzed: n = {n}")
    print(f"Mean improved return: {mean_return:.2f}% (std = {std_return:.2f}%)")
    print(f"95% CI for mean return: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print(f"One-sample t-test for mean return = 0: t({dfree}) = {t_stat:.2f}, two-sided p = {p_value_two_sided:.3f}")
    if mean_return > 0:
        print(f"One-sided p-value (H1: mean > 0) = {p_value_one_sided:.3f}")
    else:
        print(f"One-sided test (H1: mean > 0) p-value ≈ {p_value_one_sided:.3f} (mean is not positive)")
    # Interpret significance
    if p_value_two_sided < 0.05:
        direction = "positive" if mean_return > 0 else "negative"
        print(f"Result: The average improved return is statistically significant (p<0.05) and is {direction}.")
    else:
        print("Result: No statistically significant difference from zero average return (p>=0.05).")
    # Probability of positive return
    print(f"\nNumber of profitable (positive-return) strategies: {num_positive} out of {n} ({prop_positive*100:.1f}%).")
    print(f"Estimated probability of a strategy being profitable: {prop_positive*100:.1f}%")
    print(f"95% CI for this probability: [{ci_prop_lower*100:.1f}%, {ci_prop_upper*100:.1f}%]")
    # Sample size for 1% detection
    print(f"\nRequired sample size to detect a true mean return of 1% with 80% power: ~{n_required} strategies")
    print(f"(Using observed σ={std_return:.2f}%, α=5%, two-sided test).")
    
# Ensure the main function runs only when this script is executed directly
if __name__ == "__main__":
    main()
# ─────────────────────────────────────────────────────────────
from statsmodels.stats.power import TTestPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# ── 1) Sample‐size for mean return (one‐sample t‐test) ────────────────
# Suppose you want to detect a true average return of Δ = 0.5% (i.e. 0.5) 
# against H0: mean=0, with σ_est= observed std dev, α=0.05, power=0.8.

delta = 0.5               # target mean difference in percent
sigma_est = 1.26          # estimated σ from past data (in percent)
alpha = 0.05
power = 0.80

# Compute Cohen's d effect size:
effect_size = delta / sigma_est

# Instantiate the solver
ttest_power = TTestPower()
n_ttest = ttest_power.solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    alternative='two-sided'   # or 'larger' if you only care about positive mean
)

print(f"Need ~{n_ttest:.1f} runs to detect Δ={delta}% with 80% power (two‐sided).")

# ── 2) Sample‐size for proportion (one‐sample) ────────────────────────
# Suppose you expect to improve your win‐rate from p0=0.50 to p1=0.65 
# (i.e. the probability of a profitable run) and want 80% power at α=0.05.

p0 = 0.50
p1 = 0.65

# Compute the standardized effect size (Cohen’s h) for proportions
h = proportion_effectsize(p1, p0)

prop_power = NormalIndPower()
n_prop = prop_power.solve_power(
    effect_size=h,
    alpha=alpha,
    power=power,
    alternative='larger'     # one‐sided: testing p > p0
)

print(f"Need ~{n_prop:.1f} runs to detect p={p1:.2f} vs. p0={p0:.2f} with 80% power.")
# ─────────────────────────────────────────────────────────────