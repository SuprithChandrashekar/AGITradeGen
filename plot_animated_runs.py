import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Set dark theme manually (instead of mplcyberpunk)
plt.style.use('dark_background')

# Simulate strategy evaluation data
iterations = np.arange(1, 21)
mean_runs = 40 + np.random.normal(0, 0.8, size=20)
winrate_runs = 16 + np.random.normal(0, 0.3, size=20)

# Generate ECG-style pulse signal
x_ecg = np.linspace(0, 6 * np.pi, 200)
ecg = np.sin(x_ecg) * np.exp(-0.1 * x_ecg) + 0.1 * np.random.randn(200)

# Create dashboard layout
fig = plt.figure(figsize=(14, 7), dpi=100)
gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.6, hspace=0.5)

# Line chart of required runs
ax1 = fig.add_subplot(gs[:2, :3])
ax1.plot(iterations, mean_runs, color='cyan', label='🧠 Mean Required Runs', linewidth=2)
ax1.plot(iterations, winrate_runs, color='orange', label='📈 Win-Rate Required Runs', linewidth=2)
ax1.set_title("Strategy Evaluation Over Iterations", fontsize=14)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Required Runs")
ax1.legend()
ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)

# ECG-style signal chart
ax2 = fig.add_subplot(gs[2, :4])
ax2.plot(ecg, color='lime', linewidth=2, label='Signal Strength Pulse')
ax2.set_title("Signal Confidence Pulse (ECG Inspired)", fontsize=14)
ax2.set_xlabel("Time Index")
ax2.set_ylabel("Pulse Strength")
ax2.set_ylim(-2, 2)
ax2.grid(True, color='gray', linestyle='--', linewidth=0.5)
ax2.legend()

# Metrics dashboard panel
ax3 = fig.add_subplot(gs[0, 3])
ax3.axis('off')
metrics = {
    "Last Iteration": f"{iterations[-1]}",
    "Avg Mean Runs": f"{mean_runs.mean():.2f}",
    "Avg Win-Rate": f"{winrate_runs.mean():.2f}",
    "Best Win-Rate": f"{winrate_runs.min():.2f}",
    "Sharpe Ratio": "1.82",
    "Drawdown": "-2.3%",
    "Strategies": f"{len(iterations)}"
}
dashboard_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
ax3.text(0, 1, f"📊 DASHBOARD\n\n{dashboard_text}",
         fontsize=11, va='top', ha='left', fontfamily='monospace', color='deepskyblue')

plt.tight_layout()
plt.show()
