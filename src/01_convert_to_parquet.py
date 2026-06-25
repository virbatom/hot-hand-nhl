# 01_convert_to_parquet.py

"""
Convert raw NHL CSVs to Parquet. Run once.

Why Parquet: columnar, compressed, preserves dtypes.
8M-row CSV takes ~60-90s to read every time; same as Parquet ~3-5s.
"""
from pathlib import Path
import time
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
INTERIM = PROJECT_ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

# Map shorthand name -> raw filename. Adjust filenames if yours differ.
FILES = {
    "events":       "NHL_EventData.csv",
    "players":      "NHL_Players.csv",
    "schedule":     "NHL_Schedule.csv",
    "shifts":       "NHL_Shifts.csv",
    "results_odds": "results-and-odds.csv",
}

def convert(name: str, fname: str) -> None:
    src = RAW / fname
    dst = INTERIM / f"{name}.parquet"
    if not src.exists():
        print(f"SKIP  {fname}  (not found in {RAW})")
        return
    t0 = time.time()
    # infer_schema_length=100_000 -> use more rows for type inference than polars' default
    # (defaults sometimes misread columns that are empty in early rows then populated later).
    # ignore_errors=True -> coerce malformed rows to null instead of crashing; we'll audit after.
    df = pl.read_csv(src, infer_schema_length=100_000, ignore_errors=True, null_values=["", "NA", "NaN", r"\N"])
    # Force xG columns to Float64 (inference fails when most rows are null)
    for c in ["xG_F", "xG_S"]:
        if c in df.columns and df.schema[c] == pl.Utf8:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    df.write_parquet(dst, compression="snappy")
    dt = time.time() - t0
    size_mb = dst.stat().st_size / 1e6
    print(f"OK    {name:13s}  shape={df.shape!s:25s}  {size_mb:7.1f} MB  ({dt:5.1f}s)")

if __name__ == "__main__":
    print(f"Raw dir:     {RAW}")
    print(f"Output dir:  {INTERIM}\n")
    for name, fname in FILES.items():
        convert(name, fname)
    print("\nDone.")