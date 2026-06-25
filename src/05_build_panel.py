# 05_build_panel.py

"""
Build the analysis panel: one row per (player × game) for seasons >= 2010-11.

Key design decisions (documented for thesis methodology section):
  - Shot attempts = Corsi events (SOG + missed + blocked + goals)
  - Primary filter: StrengthState == "5v5"
  - Treatment goal: first 5v5 goal OR ENF goal (scorer pulled own goalie)
  - Excluded: ENA goals (opposing goalie pulled = empty-net goal)
  - Exposure: ice time approximated via shift data where available,
    else time-remaining-in-game after first goal
  - Non-scorers get a "placebo split" at the median first-goal time
    of scorers in that season (for comparable pre/post windows)

Output: data/processed/analysis_panel.parquet
"""
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM   = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

events   = pl.read_parquet(INTERIM / "events.parquet")
history  = pl.read_parquet(INTERIM / "player_game_history.parquet")
schedule = pl.read_parquet(INTERIM / "schedule.parquet")
players  = pl.read_parquet(INTERIM / "players.parquet")

print(f"Events:   {events.shape}")
print(f"History:  {history.shape}")
print(f"Schedule: {schedule.shape}")
print(f"Players:  {players.shape}")

# 0. Constants
MODERN_SEASON = 20102011         # first season for analysis
REG_GAME_SECS = 3 * 20 * 60     # 3600 seconds in regulation (3 × 20 min)

# Strength states to include for shot counting
SHOT_STRENGTH = ["5v5"]

# Strength states where a GOAL counts as treatment (first-goal trigger)
# 5v5 = normal play; ENF = scorer's team pulled own goalie (extra attacker)
GOAL_STRENGTH = ["5v5", "ENF"]

# Strength states to EXCLUDE for goals (opposing goalie pulled)
# ENA = Empty Net Against = opposing team's net is empty
GOAL_EXCLUDE  = ["ENA"]

# 1. Filter to modern seasons, regular season
print("\n[1/7] Filtering to modern seasons...")
ev = events.filter(
    (pl.col("Season") >= MODERN_SEASON) &
    (pl.col("SeasonState") == "regular")
)
print(f"  Events after filter: {len(ev):,}")

hist = history.filter(
    (pl.col("Season") >= MODERN_SEASON) &
    (pl.col("SeasonState") == "regular")
)
print(f"  Player-game rows:   {len(hist):,}")

# 2. Identify shot attempts (Corsi) at 5v5 per player per game
print("\n[2/7] Extracting 5v5 shot attempts...")

# Corsi events: shot-on-goal, missed-shot, blocked-shot, goal
# The dataset has a 'Corsi' column (1/0) — use it
shots_5v5 = (ev
    .filter(
        (pl.col("Corsi") == 1) &
        (pl.col("StrengthState").is_in(SHOT_STRENGTH)) &
        (pl.col("Player1_ID").is_not_null()) &
        (pl.col("Player1_ID") > 0)
    )
    .select(
        pl.col("Player1_ID").alias("PlayerID"),
        "GameID", "GameTime", "Event",
        pl.col("Goal").cast(pl.Int8).alias("Goal"),
        pl.col("Shot").cast(pl.Int8).alias("Shot"),
        pl.col("Fenwick").cast(pl.Int8).alias("Fenwick"),
        pl.col("Corsi").cast(pl.Int8).alias("Corsi"),
        "ShotType", "ShotDistance", "ShotAngle",
        "ScoreState", "StrengthState",
        pl.col("xG_F").cast(pl.Float64).alias("xG_F"),
    )
)
print(f"  5v5 Corsi events: {len(shots_5v5):,}")
print(f"  Breakdown:")
print(shots_5v5.group_by("Event").agg(pl.len().alias("n")).sort("n", descending=True))

# 3. Identify valid first goals (treatment)
print("\n[3/7] Identifying first goals (treatment)...")

# All goals at valid strength states (5v5 + ENF), excluding ENA
goals_valid = (ev
    .filter(
        (pl.col("Goal") == 1) &
        (pl.col("StrengthState").is_in(GOAL_STRENGTH)) &
        (pl.col("Player1_ID").is_not_null()) &
        (pl.col("Player1_ID") > 0)
    )
    .select(
        pl.col("Player1_ID").alias("PlayerID"),
        "GameID", "GameTime", "StrengthState", "ScoreState",
    )
    .sort(["PlayerID", "GameID", "GameTime"])
)
print(f"  Valid goals (5v5 + ENF): {len(goals_valid):,}")

# First goal per (player, game)
first_goals = (goals_valid
    .group_by(["PlayerID", "GameID"])
    .agg(
        pl.col("GameTime").min().alias("first_goal_time"),
        pl.col("StrengthState").first().alias("first_goal_strength"),
        pl.col("ScoreState").first().alias("score_state_at_first_goal"),
        pl.len().alias("total_goals_valid"),
    )
)
print(f"  Player-games with >=1 valid goal: {len(first_goals):,}")

# Also count ALL goals (including 2nd, 3rd) for secondary analysis
all_goals_count = (goals_valid
    .group_by(["PlayerID", "GameID"])
    .agg(pl.len().alias("goals_in_game"))
)

# 4. Aggregate shots per (player, game) into pre/post windows
print("\n[4/7] Aggregating shot counts with pre/post split...")

# Join first-goal time onto shots
shots_with_goal = shots_5v5.join(
    first_goals.select("PlayerID", "GameID", "first_goal_time"),
    on=["PlayerID", "GameID"],
    how="left"
)

# For each (player, game), compute:
#   - total shots (Corsi, Fenwick, SOG)
#   - shots BEFORE first goal
#   - shots AFTER first goal (excluding the goal itself)
#   - mean xG before/after
panel_shots = (shots_with_goal
    .group_by(["PlayerID", "GameID"])
    .agg([
        # Total counts
        pl.col("Corsi").sum().alias("corsi_total"),
        pl.col("Fenwick").sum().alias("fenwick_total"),
        pl.col("Shot").sum().alias("sog_total"),
        pl.col("Goal").sum().alias("goals_5v5"),  # 5v5 goals only (from shot filter)
        pl.col("xG_F").mean().alias("xg_per_shot_total"),
        pl.col("xG_F").sum().alias("xg_total"),

        # Pre-first-goal (strictly before)
        pl.col("Corsi").filter(
            pl.col("GameTime") < pl.col("first_goal_time")
        ).sum().alias("corsi_pre"),

        pl.col("xG_F").filter(
            pl.col("GameTime") < pl.col("first_goal_time")
        ).mean().alias("xg_per_shot_pre"),

        # Post-first-goal (strictly after the goal event)
        pl.col("Corsi").filter(
            pl.col("GameTime") > pl.col("first_goal_time")
        ).sum().alias("corsi_post"),

        pl.col("xG_F").filter(
            pl.col("GameTime") > pl.col("first_goal_time")
        ).mean().alias("xg_per_shot_post"),

        # Time of first and last shot (for exposure estimation)
        pl.col("GameTime").min().alias("first_shot_time"),
        pl.col("GameTime").max().alias("last_shot_time"),

        # First goal time (take first non-null)
        pl.col("first_goal_time").first().alias("first_goal_time"),
    ])
)
print(f"  Player-game rows with shots: {len(panel_shots):,}")

# 5. Compute exposure (time remaining after first goal)
print("\n[5/7] Computing exposure and placebo split...")

# For scorers: time_remaining = REG_GAME_SECS - first_goal_time
# For non-scorers: use season-median first-goal time as placebo split
season_map = (ev
    .select("GameID", "Season")
    .unique()
)
panel_shots = panel_shots.join(season_map, on="GameID", how="left")

# Compute season-level median first-goal time among scorers
season_median_goal_time = (first_goals
    .join(season_map, on="GameID", how="left")
    .group_by("Season")
    .agg(pl.col("first_goal_time").median().alias("median_first_goal_time"))
)
print("  Season-level median first-goal times:")
print(season_median_goal_time.sort("Season").head(20))

panel_shots = panel_shots.join(season_median_goal_time, on="Season", how="left")

# Treatment indicator
panel_shots = panel_shots.with_columns(
    pl.when(pl.col("first_goal_time").is_not_null())
      .then(pl.lit(1))
      .otherwise(pl.lit(0))
      .cast(pl.Int8)
      .alias("scored_first_goal")
)

# Split time: actual goal time for scorers, median for non-scorers
panel_shots = panel_shots.with_columns(
    pl.coalesce(["first_goal_time", "median_first_goal_time"])
      .alias("split_time")
)

# Time remaining after split (seconds); cap at regulation length
panel_shots = panel_shots.with_columns(
    (pl.lit(REG_GAME_SECS) - pl.col("split_time"))
      .clip(lower_bound=0)
      .alias("time_after_split"),
    pl.col("split_time")
      .clip(upper_bound=REG_GAME_SECS)
      .alias("time_before_split"),
)

# For non-scorers, recompute pre/post using placebo split
# (corsi_pre/post were null for non-scorers since first_goal_time was null)
shots_with_split = shots_5v5.join(
    panel_shots.select("PlayerID", "GameID", "split_time"),
    on=["PlayerID", "GameID"],
    how="inner"
)

panel_shots_v2 = (shots_with_split
    .group_by(["PlayerID", "GameID"])
    .agg([
        pl.col("Corsi").filter(
            pl.col("GameTime") < pl.col("split_time")
        ).sum().alias("corsi_pre_split"),

        pl.col("Corsi").filter(
            pl.col("GameTime") > pl.col("split_time")
        ).sum().alias("corsi_post_split"),

        pl.col("Fenwick").filter(
            pl.col("GameTime") > pl.col("split_time")
        ).sum().alias("fenwick_post_split"),

        pl.col("Shot").filter(
            pl.col("GameTime") > pl.col("split_time")
        ).sum().alias("sog_post_split"),

        pl.col("xG_F").filter(
            pl.col("GameTime") > pl.col("split_time")
        ).mean().alias("xg_per_shot_post_split"),

        pl.col("xG_F").filter(
            pl.col("GameTime") > pl.col("split_time")
        ).sum().alias("xg_post_split"),
    ])
)

panel_shots = panel_shots.join(panel_shots_v2, on=["PlayerID", "GameID"], how="left")

# 6. Join player history + player info
print("\n[6/7] Joining player history and metadata...")

# Player history
panel = panel_shots.join(
    hist.select("PlayerID", "GameID", "Team", "Date",
                "career_games_pre", "career_season_num",
                "played_prev_team_game", "scored_prev_team_game", "games_missed"),
    on=["PlayerID", "GameID"],
    how="inner"
)

# Player metadata (position, handedness)
panel = panel.join(
    players.select("PlayerID", "Position", "ShootsCatches"),
    left_on="PlayerID", right_on="PlayerID",
    how="left"
)

# All goals count (including 2nd, 3rd for secondary flag)
panel = panel.join(
    all_goals_count,
    on=["PlayerID", "GameID"],
    how="left"
).with_columns(
    pl.col("goals_in_game").fill_null(0).cast(pl.Int32),
    # Multi-goal game flag (scored 2+ valid goals)
    (pl.col("goals_in_game").fill_null(0) >= 2)
      .cast(pl.Int8)
      .alias("multi_goal_game"),
)

# 7. Exclude goalies, final cleanup, save
print("\n[7/7] Final cleanup...")

# Drop goalies (they don't take offensive shots in our context)
pre_drop = len(panel)
panel = panel.filter(pl.col("Position") != "G")
print(f"  Dropped {pre_drop - len(panel):,} goalie rows")

# Shot rate (per 60 min of remaining time)
panel = panel.with_columns(
    pl.when(pl.col("time_after_split") > 0)
      .then(pl.col("corsi_post_split") / pl.col("time_after_split") * 3600)
      .otherwise(None)
      .alias("corsi_rate_post_60"),
    pl.when(pl.col("time_before_split") > 0)
      .then(pl.col("corsi_pre_split") / pl.col("time_before_split") * 3600)
      .otherwise(None)
      .alias("corsi_rate_pre_60"),
)

# Sort
panel = panel.sort(["PlayerID", "Date", "GameID"])

OUT = PROCESSED / "analysis_panel.parquet"
panel.write_parquet(OUT)
print(f"\nSaved -> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
print(f"Shape:  {panel.shape}")
print(f"Columns: {panel.columns}")

# SANITY CHECKS
print("\n" + "=" * 72)
print("SANITY CHECKS")
print("=" * 72)

print(f"\nPanel shape: {panel.shape}")
print(f"Unique players: {panel['PlayerID'].n_unique():,}")
print(f"Unique games:   {panel['GameID'].n_unique():,}")
print(f"Seasons:        {sorted(panel['Season'].unique().to_list())}")

# Treatment rate
scored = panel.filter(pl.col("scored_first_goal") == 1)
print(f"\nTreatment rate: {len(scored)/len(panel)*100:.1f}% of player-games have >=1 valid goal")

# Shot distributions
print("\nCorsi total (all player-games):")
print(panel.select("corsi_total").describe())

print("\nCorsi post-split (scorers vs non-scorers):")
for label, filt in [("Scorers", pl.col("scored_first_goal") == 1),
                     ("Non-scorers", pl.col("scored_first_goal") == 0)]:
    sub = panel.filter(filt)
    stats = sub.select("corsi_post_split").describe()
    print(f"\n  {label} (n={len(sub):,}):")
    print(stats)

# Time distributions
print("\nFirst goal time distribution (scorers only, minutes):")
fg_mins = scored.select((pl.col("first_goal_time") / 60).alias("fg_min"))
print(fg_mins.describe())

# Position breakdown
print("\nPosition breakdown:")
print(panel.group_by("Position").agg(
    pl.len().alias("n"),
    pl.col("scored_first_goal").mean().alias("goal_rate"),
    pl.col("corsi_total").mean().alias("avg_corsi"),
).sort("n", descending=True))

# Quick peek at the hot-hand signal (raw, unadjusted)
print("\n--- RAW HOT-HAND SIGNAL (unadjusted) ---")
print("Mean corsi_rate_post_60 by treatment:")
print(panel.group_by("scored_first_goal").agg(
    pl.len().alias("n"),
    pl.col("corsi_rate_post_60").mean().alias("mean_rate_post"),
    pl.col("corsi_rate_pre_60").mean().alias("mean_rate_pre"),
    pl.col("corsi_total").mean().alias("mean_corsi_total"),
))