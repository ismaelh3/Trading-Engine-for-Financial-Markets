from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os
import requests

load_dotenv()


# ============ FRED API ============= #
FRED_API_KEY = os.getenv("FRED_API")
START_RAW = "2014-01-01"
END_RAW   = "2025-12-31"

FRED_SERIES = {
    "CPI": "CPIAUCSL",              # monthly
    "UNRATE": "UNRATE",             # monthly
    "DGS10": "DGS10",               # daily
    "DGS2": "DGS2",                 # daily
    "VIX": "VIXCLS",                # daily
    "CREDIT_SPREAD": "BAA10Y",      # daily
}

def download_fred_series(series_id: str, start: str, end: str, api_key: str) -> pd.Series:
    """
    Official FRED API download.
    """
    if not api_key:
        raise ValueError("Set FRED_API_KEY in your environment before running.")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()['observations']
    # print(data)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"]
    s.name = series_id
    return s



# for series_id in FRED_SERIES.values():
#     download_fred_series(series_id, START_RAW, END_RAW, FRED_API_KEY)


cpi_df = download_fred_series(FRED_SERIES["CPI"], START_RAW, END_RAW, FRED_API_KEY)
print(cpi_df.dtypes)