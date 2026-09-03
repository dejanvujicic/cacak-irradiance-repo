"""
Quality control (Section 3.8) and descriptive statistics (Table 9), including the division of the mountain range into stable and degraded periods.
"""
import pandas as pd

df = pd.read_csv("data/MeteoData.csv")
df["dattime"] = pd.to_datetime(df["dattime"])

def qc(d, night_thr=5):
    return d[(d.light > night_thr) & (d.light <= 1100) &
             (d.temperature.between(-30, 50)) & (d.humidity.between(0, 100))]

mnt = qc(df[(df.awsid == "AWS1") & (df.dattime < "2025-11-01")])
rur = qc(df[df.awsid == "AWS2"])
urb = qc(df[(df.awsid == "AWS1") & (df.dattime >= "2026-03-01")], night_thr=0)

periods = {
    "AWS1-M full period":            (mnt, None, None),
    "AWS1-M stable (Aug24-May25)":   (mnt, "2024-08-01", "2025-05-31"),
    "AWS1-M decline (Jun-Sep25)":    (mnt, "2025-06-01", "2025-09-30"),
    "AWS2-R Trnava":                 (rur, None, None),
    "AWS1-U Cacak":                  (urb, None, None),
}
print(f"{'Period':<32}{'N':>7}{'Mean':>8}{'Median':>8}{'Max':>7}{'Std':>8}{'%NASA':>8}")
for k, (d, a, b) in periods.items():
    w = d if a is None else d[(d.dattime >= a) & (d.dattime <= b + " 23:59:59")]
    print(f"{k:<32}{len(w):>7}{w.light.mean():>8.1f}{w.light.median():>8.1f}"
          f"{w.light.max():>7.0f}{w.light.std():>8.1f}{100*w.light.mean()/329.5:>7.0f}%")
