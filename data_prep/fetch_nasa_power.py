"""
Downloads daily NASA POWER data for the Čačak grid cell and generates the monthly dataset used by all models in this study.

Output: data/cacak-monthly.xlsx (303 rows, January 2001 – March 2026)

NASA POWER data are not redistributed with this repository; the script downloads them directly from the source.
"""
import requests, pandas as pd

LAT, LON = 43.89, 20.35
PARS = ["ALLSKY_KT", "ALLSKY_SRF_ALB", "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"]
CUTOFF = (2026, 3)          # arhiva u trenutku analize; kasnije se produzila

def fetch_daily(start=2001, end=2026):
    r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point",
                     params={"parameters": ",".join(PARS), "community": "RE",
                             "longitude": LON, "latitude": LAT,
                             "start": f"{start}0101", "end": f"{end}1231",
                             "format": "JSON"}, timeout=120)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    df = pd.DataFrame(p)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df[(df > -900).all(axis=1)]

def to_monthly(df):
    m = df.groupby([df.index.year, df.index.month]).mean()
    m.index.names = ["YEAR", "MO"]
    m = m.reset_index()
    return m[~((m.YEAR > CUTOFF[0]) | ((m.YEAR == CUTOFF[0]) & (m.MO > CUTOFF[1])))]

if __name__ == "__main__":
    m = to_monthly(fetch_daily())
    m.to_excel("data/cacak-monthly.xlsx", index=False)
    print(len(m), "months |", int(m.iloc[0].YEAR), int(m.iloc[0].MO),
          "->", int(m.iloc[-1].YEAR), int(m.iloc[-1].MO))
    assert len(m) == 303, "expected 303 months"
