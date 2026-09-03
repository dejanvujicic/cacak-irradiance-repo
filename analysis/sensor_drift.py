"""
Table 4 and the accompanying figure: decline in sensor response during 2025, with panel voltage as a secondary channel (Section 3.7).
"""
import pandas as pd, numpy as np, matplotlib
from scipy import stats
matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_csv("data/MeteoData.csv")
df["dattime"] = pd.to_datetime(df["dattime"]); df["h"] = df.dattime.dt.hour
d = df[(df.awsid == "AWS1") & (df.dattime < "2025-11-01")]
q = d[(d.light > 50) & (d.voltage > 5) & (d.h.between(10, 14))].copy()
q["ym"] = q.dattime.dt.to_period("M")

g = q.groupby("ym").agg(n=("id", "size"),
                        light_p95=("light", lambda x: x.quantile(.95)),
                        volt_p95=("voltage", lambda x: x.quantile(.95)))
g = g[g.n > 200]
print("Table 4:"); print(g.round(1).to_string())

t = np.arange(len(g))
for col in ["light_p95", "volt_p95"]:
    sl, _, r, p, _ = stats.linregress(t, g[col])
    print(f"{col:>10}: nagib {sl:+.2f}/mesec  p={p:.4f}  r2={r**2:.2f}")

# Comparison of the same calendar months
qc = d[(d.light > 5) & (d.light <= 1100)].copy(); qc["m"] = qc.dattime.dt.month
a = qc[(qc.dattime < "2024-12-01") & qc.m.isin([8, 9, 10])].light.mean()
b = qc[(qc.dattime > "2025-01-01") & qc.m.isin([8, 9, 10])].light.mean()
print(f"\naug-oct 2024: {a:.1f} W/m2 | aug-oct 2025: {b:.1f} W/m2 | pad {100*(1-b/a):.1f} %")

x = g.index.to_timestamp()
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(x, g.light_p95 / g.light_p95.iloc[0], "o-", color="#c0392b", label="Irradiance channel (BH1750)")
ax.plot(x, g.volt_p95 / g.volt_p95.iloc[0], "s-", color="#2471a3", label="PV panel output")
ax.axhline(1, color="0.7", lw=.8, ls="--"); ax.set_ylim(.5, 1.15)
ax.set_ylabel("Midday p95, normalised to Aug 2024")
ax.set_title("Co-located decline of both radiometric channels at AWS1-M")
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.25)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig("figures/fig_drift_two_channels.png", dpi=160)
print("figures/fig_drift_two_channels.png")
