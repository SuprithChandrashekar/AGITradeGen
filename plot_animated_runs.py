import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from pathlib import Path
from statsmodels.stats.power import TTestPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

RESULTS_XLSX = Path(__file__).parent / "reports" / "strategy_results.xlsx"

def compute_sample_sizes():
    df = pd.read_excel(RESULTS_XLSX)
    data = pd.to_numeric(df["improved_return_pct"], errors="coerce").dropna()
    if len(data) == 0:
        return float("inf"), float("inf")
    mean_ret = data.mean()
    std_ret = data.std(ddof=1)

    if std_ret == 0:
        n_mean = 0.0
    else:
        d = mean_ret / std_ret
        n_mean = TTestPower().solve_power(
            effect_size=abs(d),
            alpha=0.05,
            power=0.8,
            alternative="two-sided"
        )

    p0 = 0.5
    p1 = (data > 0).mean()
    h = proportion_effectsize(p1, p0)
    n_prop = NormalIndPower().solve_power(
        effect_size=abs(h),
        alpha=0.05,
        power=0.8,
        alternative="larger"
    )
    return n_mean, n_prop

# Prepare data containers
iterations = []
n_mean_list = []
n_prop_list = []

# Set up plot
fig, ax = plt.subplots()
line_mean, = ax.plot([], [], color='blue', label='Mean Required Runs')
line_prop, = ax.plot([], [], color='orange', label='Win-Rate Required Runs')

ax.set_title("Required Runs over Iterations")
ax.set_xlabel("Iteration")
ax.set_ylabel("Required Runs")
ax.legend(loc="upper right")
ax.grid(True)

def init():
    line_mean.set_data([], [])
    line_prop.set_data([], [])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 50)
    return line_mean, line_prop

def update(frame):
    current_iter = frame + 1

    # Prevent over-extending lists
    if len(iterations) >= current_iter:
        return line_mean, line_prop

    n_mean, n_prop = compute_sample_sizes()
    iterations.append(current_iter)
    n_mean_list.append(n_mean)
    n_prop_list.append(n_prop)

    # Dynamically expand axes
    ax.set_xlim(0, max(10, current_iter + 5))
    y_max = max(max(n_mean_list, default=0), max(n_prop_list, default=0))
    ax.set_ylim(0, max(50, y_max + 5))

    line_mean.set_data(iterations, n_mean_list)
    line_prop.set_data(iterations, n_prop_list)
    return line_mean, line_prop

ani = animation.FuncAnimation(
    fig, update, init_func=init, frames=100, interval=2000, blit=True
)
plt.tight_layout()
plt.show()