# 12_shots_by_scorestate.py

"""
12_shots_by_scorestate.py

Enrich the analysis panel with per-score-state shot counts.
For each (player, game), count post-split AND pre-split Corsi shot attempts,
broken down by the score state AT THE TIME OF EACH SHOT:
  - leading  (ScoreState > 0)   team is ahead
  - tied     (ScoreState == 0)
  - trailing (ScoreState < 0)   team is behind
Score state is from the shooting player's team perspective.

Output: data/processed/analysis_panel_ss.parquet
  (original analysis_panel.parquet is left untouched)
"""
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM   = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# 1. Load events, filter to 5v5 Corsi shot attempts (modern regular)
print("Loading events...")
events = pl.read_parquet(INTERIM / "events.parquet")
print(f"  Raw events: {len(events):,}")

shots = events.filter(
    (pl.col("Corsi") == 1) &
    (pl.col("Season") >= 20102011) &
    (pl.col("SeasonState") == "regular") &
    (pl.col("StrengthState") == "5v5")
).select(
    pl.col("Player1_ID").alias("PlayerID"),
    "GameID",
    "GameTime",
    pl.col("ScoreState").alias("ss_raw"),
)
print(f"  5v5 Corsi attempts: {len(shots):,}")

# Clean score state -> integer, nulls = 0 (tied)
shots = shots.with_columns(
    pl.col("ss_raw").cast(pl.Int64, strict=False).fill_null(0).alias("score_state")
).filter(pl.col("PlayerID").is_not_null())

# 2. Join split_time from the panel
print("Joining split_time from panel...")
panel = pl.read_parquet(PROCESSED / "analysis_panel.parquet")
split = panel.select("PlayerID", "GameID", "split_time").unique()

shots = shots.join(split, on=["PlayerID", "GameID"], how="inner")
print(f"  Shots matched to panel split_time: {len(shots):,}")

# 3. Tag pre/post and score-state bucket
shots = shots.with_columns([
    (pl.col("GameTime") > pl.col("split_time")).alias("is_post"),
    pl.when(pl.col("score_state") > 0).then(pl.lit("leading"))
      .when(pl.col("score_state") < 0).then(pl.lit("trailing"))
      .otherwise(pl.lit("tied")).alias("ss_bucket"),
])

# 4. Aggregate counts per (player, game)
def count_bucket(df, is_post, bucket, name):
    return (df
        .filter((pl.col("is_post") == is_post) & (pl.col("ss_bucket") == bucket))
        .group_by(["PlayerID", "GameID"])
        .agg(pl.len().alias(name))
    )

agg = split.select("PlayerID", "GameID")
specs = [
    (True,  "leading",  "shots_post_leading"),
    (True,  "tied",     "shots_post_tied"),
    (True,  "trailing", "shots_post_trailing"),
    (False, "leading",  "shots_pre_leading"),
    (False, "tied",     "shots_pre_tied"),
    (False, "trailing", "shots_pre_trailing"),
]
for is_post, bucket, name in specs:
    agg = agg.join(count_bucket(shots, is_post, bucket, name),
                   on=["PlayerID", "GameID"], how="left")

fill_cols = [s[2] for s in specs]
agg = agg.with_columns([pl.col(c).fill_null(0) for c in fill_cols])

# 5. Join onto panel, validate, save
print("Joining onto panel and validating...")
drop_existing = [c for c in panel.columns if c in fill_cols]
if drop_existing:
    panel = panel.drop(drop_existing)

panel = panel.join(agg, on=["PlayerID", "GameID"], how="left")
panel = panel.with_columns([pl.col(c).fill_null(0) for c in fill_cols])

# validation: post buckets should sum to corsi_post_split
panel = panel.with_columns(
    (pl.col("shots_post_leading") + pl.col("shots_post_tied") + pl.col("shots_post_trailing"))
      .alias("_post_sum")
)
mismatch = panel.filter(pl.col("_post_sum") != pl.col("corsi_post_split")).height
print(f"  Rows where post breakdown != corsi_post_split: {mismatch:,} "
      f"({mismatch/len(panel)*100:.2f}%)")
if mismatch > 0:
    print("  Sample mismatches:")
    print(panel.filter(pl.col("_post_sum") != pl.col("corsi_post_split"))
          .select("PlayerID","GameID","split_time","corsi_post_split","_post_sum",
                  "shots_post_leading","shots_post_tied","shots_post_trailing").head(10))
panel = panel.drop("_post_sum")

OUT = PROCESSED / "analysis_panel_ss.parquet"
panel.write_parquet(OUT)
print(f"\nSaved -> {OUT}")
print(f"Shape: {panel.shape}")
print(f"New columns: {fill_cols}")

# Sanity checks
print("\n" + "="*60)
print("SANITY CHECKS")
print("="*60)
print("\nTotal post-split shots by score state:")
print(panel.select(["shots_post_leading","shots_post_tied","shots_post_trailing"]).sum())

print("\nMean post-split shots by score state, SCORERS:")
print(panel.filter(pl.col("scored_first_goal")==1).select([
    pl.col("shots_post_leading").mean().alias("leading"),
    pl.col("shots_post_tied").mean().alias("tied"),
    pl.col("shots_post_trailing").mean().alias("trailing"),
]))
print("Mean post-split shots by score state, NON-SCORERS:")
print(panel.filter(pl.col("scored_first_goal")==0).select([
    pl.col("shots_post_leading").mean().alias("leading"),
    pl.col("shots_post_tied").mean().alias("tied"),
    pl.col("shots_post_trailing").mean().alias("trailing"),
]))