# 02_sanity_check.py

"""
Initial sanity checks on the parquet data. Print-only, no writes.
Goal: catch data-quality issues before we build the panel.
"""
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"

events   = pl.read_parquet(INTERIM / "events.parquet")
schedule = pl.read_parquet(INTERIM / "schedule.parquet")
shifts   = pl.read_parquet(INTERIM / "shifts.parquet")
players  = pl.read_parquet(INTERIM / "players.parquet")

def header(s: str) -> None:
    print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)

header("SHAPES & MEMORY")
for name, df in [("events", events), ("schedule", schedule),
                 ("shifts", shifts), ("players", players)]:
    print(f"{name:10s}  shape={df.shape!s:22s}  mem={df.estimated_size('mb'):6.1f} MB")

header("EVENTS: SEASON COVERAGE")
print(
    events.group_by("Season")
    .agg(pl.len().alias("rows"),
         pl.n_unique("GameID").alias("games"))
    .sort("Season")
)

header("EVENTS: EVENT TYPE FREQUENCIES")
print(events.group_by("Event").agg(pl.len().alias("n")).sort("n", descending=True))

header("EVENTS: STRENGTH STATE CODES  (need to identify 5v5, 6v5, 5v6)")
print(
    events.filter(pl.col("Event").is_in(["goal", "shot-on-goal", "missed-shot", "blocked-shot"]))
          .group_by("StrengthState")
          .agg(pl.len().alias("n"))
          .sort("n", descending=True)
)

header("EVENTS: STRENGTH STATE × EVENT  (cross-tab for shots/goals)")
print(
    events.filter(pl.col("Event").is_in(["goal", "shot-on-goal"]))
          .group_by(["StrengthState", "Event"])
          .agg(pl.len().alias("n"))
          .sort(["StrengthState", "Event"])
    .head(40)
)

header("EVENTS: COLUMN NULL RATES FOR MODERN SEASONS (>=20102011)")
modern = events.filter(pl.col("Season") >= 20102011)
nulls = {c: modern[c].null_count() / len(modern) for c in modern.columns}
for col, frac in sorted(nulls.items(), key=lambda kv: -kv[1]):
    if frac > 0.01:
        print(f"  {col:25s}  {frac*100:5.1f}% null")

header("SCHEDULE: SEASONS COVERED")
print(
    schedule.group_by("Season")
    .agg(pl.len().alias("games"))
    .sort("Season")
)

header("KEY DTYPES (important for joins)")
print("events Player1_ID dtype:    ", events.schema["Player1_ID"])
print("events GameID dtype:        ", events.schema["GameID"])
print("schedule GameID dtype:      ", schedule.schema["GameID"])
print("players PlayerID dtype:     ", players.schema["PlayerID"])

header("JOIN SANITY: do event GameIDs match schedule GameIDs (post-2010)?")
ev_games = events.filter(pl.col("Season") >= 20102011).select("GameID").unique()
sc_games = schedule.select("GameID").unique()
both = ev_games.join(sc_games, on="GameID", how="inner")
print(f"  events games (post-2010):  {len(ev_games):,}")
print(f"  schedule games:             {len(sc_games):,}")
print(f"  intersection:               {len(both):,}")
print(f"  in events, not schedule:    {len(ev_games) - len(both):,}")
