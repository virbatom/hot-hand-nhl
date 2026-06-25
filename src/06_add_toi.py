# 06_add_toi.py

"""
Compute Time on Ice (TOI) per (player, game) from the shifts data.

For each player-game in the analysis panel, compute:
  - toi_5v5_total:  total 5v5 ice time in seconds
  - toi_5v5_pre:    5v5 ice time BEFORE the split point (first goal or placebo)
  - toi_5v5_post:   5v5 ice time AFTER the split point

Shifts that straddle the split time are proportionally allocated.

Output: updated data/processed/analysis_panel.parquet (adds 3 columns)
"""
from pathlib import Path
import polars as pl
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM   = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"

t0 = time.time()

# 1. Load shifts and filter to modern 5v5
print("Loading shifts...")
shifts = pl.read_parquet(INTERIM / "shifts.parquet")
print(f"  Raw shifts: {len(shifts):,}")

# Clean \N values in strength state columns
for c in ["Home_StrengthState", "Away_StrengthState"]:
    if shifts.schema[c] == pl.Utf8:
        shifts = shifts.with_columns(
            pl.when(pl.col(c) == r"\N").then(None).otherwise(pl.col(c)).alias(c)
        )

# We need to know which seasons these shifts belong to.
# Join with schedule to get Season.
schedule = pl.read_parquet(INTERIM / "schedule.parquet")
sched_min = schedule.select("GameID", "Season", "SeasonState")

shifts = shifts.join(sched_min, on="GameID", how="inner")

# Filter to modern regular season, 5v5
shifts_5v5 = shifts.filter(
    (pl.col("Season") >= 20102011) &
    (pl.col("SeasonState") == "regular") &
    (pl.col("Home_StrengthState") == "5v5") &
    (pl.col("Away_StrengthState") == "5v5")
)
print(f"  5v5 modern shifts: {len(shifts_5v5):,}")

# 2. Explode roster columns to get (PlayerID, GameID, Start, End)
print("\nExploding roster columns to individual players...")

# Clean \N in roster columns
ROSTER_COLS = ["Home_Forwards_ID", "Home_Defenders_ID", "Home_Goalie_ID",
               "Away_Forwards_ID", "Away_Defenders_ID", "Away_Goalie_ID"]

for c in ROSTER_COLS:
    if shifts_5v5.schema[c] == pl.Utf8:
        shifts_5v5 = shifts_5v5.with_columns(
            pl.when(pl.col(c) == r"\N").then(None).otherwise(pl.col(c)).alias(c)
        )

def explode_roster_str(df, col):
    """Explode a space-separated string player ID column."""
    return (df
        .filter(pl.col(col).is_not_null() & (pl.col(col).str.len_chars() > 0))
        .select("GameID", "Start", "End", "Duration", pl.col(col).alias("raw"))
        .with_columns(pl.col("raw").str.split(" ").alias("ids"))
        .explode("ids")
        .filter(pl.col("ids").str.len_chars() > 0)  # drop empty strings from double spaces
        .with_columns(
            pl.col("ids").str.strip_chars()
              .cast(pl.Int64, strict=False).alias("PlayerID")
        )
        .filter(pl.col("PlayerID").is_not_null() & (pl.col("PlayerID") > 0))
        .select("PlayerID", "GameID", "Start", "End", "Duration")
    )

def explode_roster_int(df, col):
    """Handle single-integer player ID column (goalies)."""
    return (df
        .filter(pl.col(col).is_not_null() & (pl.col(col) > 0))
        .select(pl.col(col).alias("PlayerID"), "GameID", "Start", "End", "Duration")
    )

parts = []
STR_COLS = ["Home_Forwards_ID", "Home_Defenders_ID",
            "Away_Forwards_ID", "Away_Defenders_ID"]
INT_COLS = ["Home_Goalie_ID", "Away_Goalie_ID"]

for col in STR_COLS:
    parts.append(explode_roster_str(shifts_5v5, col))
    print(f"  {col}: done")

for col in INT_COLS:
    parts.append(explode_roster_int(shifts_5v5, col))
    print(f"  {col}: done")

player_shifts = pl.concat(parts).unique()
print(f"\n  Player-shift rows (deduplicated): {len(player_shifts):,}")

# 3. Compute total 5v5 TOI per (player, game)
print("\nComputing total TOI per player-game...")

toi_total = (player_shifts
    .group_by(["PlayerID", "GameID"])
    .agg(pl.col("Duration").sum().alias("toi_5v5_total"))
)
print(f"  Player-game pairs with TOI: {len(toi_total):,}")

# Sanity: mean TOI should be ~12-14 min for skaters
mean_toi_min = toi_total["toi_5v5_total"].mean() / 60
print(f"  Mean 5v5 TOI: {mean_toi_min:.1f} min (expect ~12-14)")

# 4. Compute pre/post split TOI
print("\nComputing pre/post split TOI...")

# Load the analysis panel to get split_time per (PlayerID, GameID)
panel = pl.read_parquet(PROCESSED / "analysis_panel.parquet")
# Drop any existing TOI columns from previous runs
drop_cols = [c for c in panel.columns if "toi_5v5" in c or "corsi_rate_post_toi" in c or "corsi_rate_pre_toi" in c]
if drop_cols:
    panel = panel.drop(drop_cols)
    print(f"  Dropped old columns: {drop_cols}")
split_times = panel.select("PlayerID", "GameID", "split_time").unique()

# Join split_time onto player_shifts
ps_with_split = player_shifts.join(split_times, on=["PlayerID", "GameID"], how="inner")

# For each shift, compute seconds before and after split_time
# A shift from Start to End that straddles split_time gets split:
#   pre  = max(0, min(End, split_time) - Start)
#   post = max(0, End - max(Start, split_time))
ps_with_split = ps_with_split.with_columns([
    (pl.min_horizontal("End", "split_time") - pl.col("Start"))
      .clip(lower_bound=0)
      .alias("dur_pre"),
    (pl.col("End") - pl.max_horizontal("Start", "split_time"))
      .clip(lower_bound=0)
      .alias("dur_post"),
])

toi_split = (ps_with_split
    .group_by(["PlayerID", "GameID"])
    .agg([
        pl.col("dur_pre").sum().alias("toi_5v5_pre"),
        pl.col("dur_post").sum().alias("toi_5v5_post"),
    ])
)
print(f"  Player-game pairs with split TOI: {len(toi_split):,}")

# 5. Join TOI onto analysis panel and save
print("\nJoining TOI onto analysis panel...")

panel = panel.join(toi_total, on=["PlayerID", "GameID"], how="left")
panel = panel.join(toi_split, on=["PlayerID", "GameID"], how="left")

# Fill nulls (players not found in shifts = 0 TOI, shouldn't happen often)
null_toi = panel.filter(pl.col("toi_5v5_total").is_null()).height
print(f"  Rows with null TOI: {null_toi:,} ({null_toi/len(panel)*100:.1f}%)")

panel = panel.with_columns([
    pl.col("toi_5v5_total").fill_null(0),
    pl.col("toi_5v5_pre").fill_null(0),
    pl.col("toi_5v5_post").fill_null(0),
])

# Compute rate per 60 with actual TOI
panel = panel.with_columns([
    pl.when(pl.col("toi_5v5_post") > 0)
      .then(pl.col("corsi_post_split") / pl.col("toi_5v5_post") * 3600)
      .otherwise(None)
      .alias("corsi_rate_post_toi60"),
    pl.when(pl.col("toi_5v5_pre") > 0)
      .then(pl.col("corsi_pre_split") / pl.col("toi_5v5_pre") * 3600)
      .otherwise(None)
      .alias("corsi_rate_pre_toi60"),
])

OUT = PROCESSED / "analysis_panel.parquet"
panel.write_parquet(OUT)
print(f"\nSaved -> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
print(f"Shape:  {panel.shape}")
print(f"New columns: toi_5v5_total, toi_5v5_pre, toi_5v5_post, corsi_rate_post_toi60, corsi_rate_pre_toi60")

# Sanity checks
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

print("\ntoi_5v5_total (minutes) distribution:")
print(panel.select((pl.col("toi_5v5_total") / 60).alias("toi_min")).describe())

print("\ntoi_5v5_pre vs toi_5v5_post (minutes, scorers only):")
scorers = panel.filter(pl.col("scored_first_goal") == 1)
print(f"  Pre:  mean={scorers['toi_5v5_pre'].mean()/60:.1f} min")
print(f"  Post: mean={scorers['toi_5v5_post'].mean()/60:.1f} min")

print("\nCorsi rate with actual TOI vs game-time exposure (full sample):")
for col in ["corsi_rate_post_60", "corsi_rate_post_toi60"]:
    vals = panel[col].drop_nulls()
    print(f"  {col}: mean={vals.mean():.2f}, median={vals.median():.2f}")

print(f"\nTotal time: {time.time()-t0:.0f}s")