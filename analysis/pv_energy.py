"""Table 2 and Figure 2: Solar harvesting conditions based on panel voltage."""
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

NOM = 619.0
df = pd.read_csv("data/MeteoData.csv")
df["dattime"] = pd.to_datetime(df["dattime"]); df["day"] = df.dattime.dt.date
d = df[(df.awsid == "AWS1") & (df.dattime < "2025-11-01")].copy()

print("identity of channel 'voltage':")
print("  illumination correlation:", round(d.voltage.corr(d.light), 3))
print("  wind speed correlation:", round(d.voltage.corr(d.windspeed), 3))

g = d.groupby("day").agg(vsum=("voltage", "sum"), n=("id", "size"),
                         vmax=("voltage", "max"),
                         on=("voltage", lambda x: (x > 0).sum() * NOM / 3600))
g = g[g.n >= 60]; g.index = pd.to_datetime(g.index)
g["Vh"] = g.vsum * NOM / 3600
st = g[(g.index >= "2024-08-01") & (g.index <= "2025-05-31")]
print("\nTable 2:")
print(st.groupby(st.index.month).agg(prod_hours=("on", "median"),
                                     Vh=("Vh", "median"), days=("n", "size")).round(1).to_string())

full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(full.index, full.vmax, lw=1.0, color="#c0392b")
ax.set_ylabel("Daily max panel voltage (V)", color="#c0392b"); ax.set_ylim(0, 85)
ax2 = ax.twinx(); ax2.plot(full.index, full.on, lw=1.0, color="#2471a3", alpha=.85)
ax2.set_ylabel("Productive hours per day", color="#2471a3"); ax2.set_ylim(0, 14)
ax.axvspan(pd.Timestamp("2024-12-10"), pd.Timestamp("2025-03-25"), color="0.85", zorder=0)
ax.text(pd.Timestamp("2025-01-25"), 42, "outage\n(98 d)", ha="center", fontsize=9, color="0.35")
ax.set_title("AWS1-M photovoltaic harvesting conditions, Aug 2024 - Oct 2025")
ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig("figures/fig02_pv_harvesting.png", dpi=160)
print("\nfigures/fig02_pv_harvesting.png")
