"""
Fetch the daily series this project needs from LSEG/Refinitiv Workspace into
data/raw_prices/ (one CSV per RIC). Proprietary vendor data is NOT shipped with the
repo — run this with an active LSEG Workspace session to reproduce the study.

    python scripts/fetch_data.py --start 2010-01-01 --end today
"""
import argparse
from datetime import date
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw_prices"
RAW.mkdir(parents=True, exist_ok=True)

# RIC -> output filename stem (index RICs stored with a leading underscore, matching src/data.py)
RICS = {
    ".COR1M": "_COR1M", ".COR3M": "_COR3M", ".COR6M": "_COR6M", ".DSPX": "_DSPX",
    ".VIX": "_VIX", ".VIX3M": "_VIX3M", "SPY": "SPY",
    "XLK": "XLK", "XLF": "XLF", "XLE": "XLE", "XLV": "XLV", "XLY": "XLY",
    "XLP": "XLP", "XLI": "XLI", "XLU": "XLU", "XLB": "XLB",
}
FALLBACK = ["TRDPRC_1", "CLOSE", "CF_CLOSE", "TR.PriceClose"]


def fetch_one(ld, ric, field, start, end):
    if field.startswith("TR."):
        df = ld.get_data(universe=ric, fields=[field],
                         parameters={"SDate": start, "EDate": end, "Frq": "D"})
    else:
        df = ld.get_history(universe=ric, fields=[field], start=start, end=end, interval="daily")
    if df is None or df.empty:
        return None
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0:
        return None
    s = num.iloc[:, 0].dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index() if len(s) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="today")
    args = ap.parse_args()
    end = date.today().strftime("%Y-%m-%d") if args.end == "today" else args.end

    import lseg.data as ld
    ld.open_session()
    print(f"Fetching {len(RICS)} RICs {args.start} -> {end}")
    for ric, stem in RICS.items():
        s = None
        for f in FALLBACK:
            s = fetch_one(ld, ric, f, args.start, end)
            if s is not None and len(s) >= 50:
                break
        if s is None:
            print(f"  {ric:8} FAILED — check entitlement")
            continue
        s.to_csv(RAW / f"{stem}.csv", header=[s.name or "value"])
        print(f"  {ric:8} -> {stem}.csv  ({len(s)} obs)")
    ld.close_session()
    print("Done.")


if __name__ == "__main__":
    main()
