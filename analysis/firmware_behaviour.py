"""
Verification of firmware behavior from the archive (Sections 3.2, 3.3, 3.5):

- whether nighttime transmissions are suppressed
- whether there is an adaptive cadence based on panel voltage
- whether batch transmission is supported
- whether the node buffers records when the connection is interrupted
"""
import pandas as pd

df = pd.read_csv("data/MeteoData.csv")
df["dattime"] = pd.to_datetime(df["dattime"]); df["h"] = df.dattime.dt.hour
d = df[(df.awsid == "AWS1") & (df.dattime >= "2024-08-15") &
       (df.dattime <= "2024-10-31")].sort_values("dattime")
d["dt"] = d.dattime.diff().dt.total_seconds()
w = d[(d.dt > 0) & (d.dt < 5000)]

h = d.groupby("h").size()
print("Nighttime/daytime ratio of records per hour:",
      round(h.loc[0:4].mean() / h.loc[10:14].mean(), 3), "(1.0 = nema potiskivanja)")
print("Nighttime interval:", round(w[w.h.between(0, 4)].dt.median(), 1), "s")
print("Daytime interval:", round(w[w.h.between(10, 14)].dt.median(), 1), "s")
print("Interval with voltage=0 :", round(w[w.voltage == 0].dt.median(), 1), "s")
print("Interval with voltage>40:", round(w[w.voltage > 40].dt.median(), 1), "s")
print("Ratio of intervals <60 s   :", round(100 * (w.dt < 60).mean(), 2), "% (batch)")

a = df[df.awsid.isin(["AWS1", "AWS2"])].sort_values("id")
back = a.groupby("awsid").dattime.diff().dt.total_seconds()
print("Records with timestamps older than the previous record by ID:", int((back < 0).sum()),
      "-> 0 means there is no buffer")
