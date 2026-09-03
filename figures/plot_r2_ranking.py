"""Figure 5: Ranking of all 25 models by R² on the test set."""
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch

df = pd.read_csv("data/model_metrics_table_B1.csv").sort_values("R2", ascending=False)
d = df.iloc[::-1]
col = ["#f1c40f" if m == "ANN-ReLU" else ("#2471a3" if r > .95 else ("#c0392b" if r < 0 else "0.7"))
       for m, r in zip(d.Model, d.R2)]
fig, ax = plt.subplots(figsize=(8.6, 7.2))
ax.barh(d.Model, d.R2, color=col)
for m, r in zip(d.Model, d.R2):
    ax.text(r + .012 if r > 0 else r - .012, m, f"{r:.3f}",
            va="center", ha="left" if r > 0 else "right", fontsize=7.5)
ax.axvline(0, color="0.4", lw=.8); ax.set_xlim(-.25, 1.12)
ax.set_xlabel("R² (test set)")
ax.set_title("Coefficient of determination for all 25 models,\nCacak monthly NASA POWER series", fontsize=11)
ax.legend(handles=[Patch(color="#2471a3", label="Two leading models"),
                   Patch(color="#f1c40f", label="ANN-ReLU baseline (added in revision)"),
                   Patch(color="0.7", label="Remaining converged models"),
                   Patch(color="#c0392b", label="Divergent TCN-recurrent hybrids")],
          loc="lower right", fontsize=8)
ax.grid(axis="x", alpha=.25); fig.tight_layout()
fig.savefig("figures/fig05_r2_ranking.png", dpi=170)
print("figures/fig05_r2_ranking.png")
