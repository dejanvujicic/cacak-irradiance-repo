"""Table 1: Connection reliability by location, based on the measurement database."""
import pandas as pd

NOM = 619.0                     
EXP_DAY = 86400 / NOM        

SITES = [("AWS1-M (mountain)", "AWS1", "2024-08-11", "2025-10-31"),
         ("AWS2-R (rural)",    "AWS2", "2024-05-14", "2024-08-18"),
         ("AWS1-U (urban)",    "AWS1", "2026-03-01", "2026-05-31")]

df = pd.read_csv("data/MeteoData.csv")
df["dattime"] = pd.to_datetime(df["dattime"])

rows = []
for name, aws, s, e in SITES:
    d = df[(df.awsid == aws) & (df.dattime >= s) &
           (df.dattime <= e + " 23:59:59")].sort_values("dattime")
    d = d.assign(day=d.dattime.dt.date)
    per_day = d.groupby("day").size()
    op = per_day[per_day >= 10]                 # working days
    dt = d.dattime.diff().dt.total_seconds().dropna()
    rows.append(dict(Site=name,
                     CalDays=(pd.Timestamp(e) - pd.Timestamp(s)).days + 1,
                     OpDays=len(op), Records=len(d),
                     MedPerDay=int(per_day.median()),
                     Delivery=round(100 * op.mean() / EXP_DAY, 1),
                     Gaps1h=int((dt > 3600).sum()),
                     MaxGapDays=round(dt.max() / 86400, 1)))
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nexpected logs/day with {NOM} s: {EXP_DAY:.1f}")
