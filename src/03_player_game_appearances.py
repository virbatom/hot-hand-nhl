# 03_player_game_appearances.py


r"""
Build player-game appearance log (corrected version).

Sources (all yield (PlayerID, GameID, Team) rows):
  A. Player1_ID with EventTeam       -> reliable, covers all seasons
  B. Goalie_ID on shot/goal events   -> opposing team via schedule join
  C. Roster columns (post-\N clean)  -> modern seasons only

Combine, dedupe on (PlayerID, GameID), resolve team conflicts by preferring
the Player1-based assignment (most reliable source).

Output: data/interim/player_game_appearances.parquet
"""
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"

events   = pl.read_parquet(INTERIM / "events.parquet")
schedule = pl.read_parquet(INTERIM / "schedule.parquet")

# Clean \N -> null on roster columns (MySQL-style nulls in CSV)
ROSTER_COLS = ["Home_Forwards_ID", "Home_Defenders_ID", "Home_Goalie_ID",
               "Away_Forwards_ID", "Away_Defenders_ID", "Away_Goalie_ID"]

events = events.with_columns([
    pl.when(pl.col(c) == r"\N").then(None).otherwise(pl.col(c)).alias(c)
    for c in ROSTER_COLS
])

print("Roster-populated row rate by season (spot checks):")
for s in [19171918, 19671968, 20072008, 20102011, 20202021, 20242025]:
    ev_s = events.filter(pl.col("Season") == s)
    if len(ev_s) == 0:
        print(f"  {s}: no data"); continue
    frac = ev_s.filter(pl.col("Home_Forwards_ID").is_not_null()).height / len(ev_s)
    print(f"  {s}: {frac*100:5.1f}% rows have Home_Forwards_ID")

sched_min = schedule.select("GameID", "HomeTeam", "AwayTeam")

# SOURCE A: Player1 -> on EventTeam
src_a = (events
    .filter(
        pl.col("Player1_ID").is_not_null() & (pl.col("Player1_ID") > 0) &
        pl.col("EventTeam").is_not_null()
    )
    .select(pl.col("Player1_ID").alias("PlayerID"), "GameID",
            pl.col("EventTeam").alias("Team"))
    .unique()
)
print(f"\nSource A (Player1):  {len(src_a):,} rows")

# SOURCE B: Goalie on shot/goal events -> opposing team
src_b = (events
    .filter(
        pl.col("Goalie_ID").is_not_null() & (pl.col("Goalie_ID") > 0) &
        pl.col("Event").is_in(["goal", "shot-on-goal", "missed-shot", "blocked-shot"]) &
        pl.col("EventTeam").is_not_null()
    )
    .select(pl.col("Goalie_ID").alias("PlayerID"), "GameID", "EventTeam")
    .unique()
    .join(sched_min, on="GameID", how="inner")
    .with_columns(
        pl.when(pl.col("EventTeam") == pl.col("HomeTeam"))
          .then(pl.col("AwayTeam"))
          .otherwise(pl.col("HomeTeam"))
          .alias("Team")
    )
    .select("PlayerID", "GameID", "Team")
    .unique()
)
print(f"Source B (Goalie):   {len(src_b):,} rows")

# SOURCE C: Rosters (6 columns, explode comma-separated IDs)
def roster_rows(col: str, team_col: str) -> pl.DataFrame:
    return (events
        .filter(pl.col(col).is_not_null() & (pl.col(col).str.len_chars() > 0))
        .select("GameID", pl.col(col).alias("raw"))
        .unique()
        .with_columns(pl.col("raw").str.split(",").alias("ids"))
        .explode("ids")
        .with_columns(
            pl.col("ids").str.strip_chars()
              .cast(pl.Int64, strict=False).alias("PlayerID")
        )
        .filter(pl.col("PlayerID").is_not_null() & (pl.col("PlayerID") > 0))
        .join(sched_min, on="GameID", how="inner")
        .with_columns(pl.col(team_col).alias("Team"))
        .select("PlayerID", "GameID", "Team")
        .unique()
    )

roster_parts = [
    roster_rows("Home_Forwards_ID",  "HomeTeam"),
    roster_rows("Home_Defenders_ID", "HomeTeam"),
    roster_rows("Home_Goalie_ID",    "HomeTeam"),
    roster_rows("Away_Forwards_ID",  "AwayTeam"),
    roster_rows("Away_Defenders_ID", "AwayTeam"),
    roster_rows("Away_Goalie_ID",    "AwayTeam"),
]
src_c = pl.concat(roster_parts).unique()
print(f"Source C (Rosters):  {len(src_c):,} rows")

# Combine; detect + resolve team conflicts (prefer source A)
combined = pl.concat([src_a, src_b, src_c]).unique()
print(f"\nCombined (before conflict resolution): {len(combined):,}")

conflicts = (combined
    .group_by(["PlayerID", "GameID"])
    .agg(pl.col("Team").n_unique().alias("n_teams"))
    .filter(pl.col("n_teams") > 1)
)
print(f"(PlayerID, GameID) pairs with >1 team assignments: {len(conflicts):,}")

# Prefer Player1-derived team; fall back otherwise
pref = src_a.rename({"Team": "TeamPref"})
resolved = (combined
    .join(pref, on=["PlayerID", "GameID"], how="left")
    .with_columns(pl.coalesce(["TeamPref", "Team"]).alias("Team"))
    .select("PlayerID", "GameID", "Team")
    .unique(subset=["PlayerID", "GameID"], keep="first")
)
print(f"After resolution: {len(resolved):,}")

# Enrich with schedule fields
sched_full = schedule.select("GameID", "Season", "SeasonState", "Date")
appearances = (resolved
    .join(sched_full, on="GameID", how="inner")
    .select("PlayerID", "GameID", "Team", "Season", "SeasonState", "Date")
    .unique()
    .sort(["PlayerID", "Date", "GameID"])
)

print(f"\nFinal shape: {appearances.shape}")
print(f"Unique players: {appearances['PlayerID'].n_unique():,}")
print(f"Unique games:   {appearances['GameID'].n_unique():,}")
print(f"Date range:     {appearances['Date'].min()} -> {appearances['Date'].max()}")

OUT = INTERIM / "player_game_appearances.parquet"
appearances.write_parquet(OUT)
print(f"\nSaved -> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")

# Sanity: compare to known real career-game totals
print("\nReal career-game benchmarks (approx, regular + playoffs):")
for name, g in [("Patrick Marleau", 1779), ("Zdeno Chara", 1680),
                ("Jaromir Jagr (NHL)", 1733), ("Joe Thornton", 1714),
                ("Alex Ovechkin (active)", 1491), ("Sidney Crosby (active)", 1324)]:
    print(f"  {name:28s} ~{g}")

players = pl.read_parquet(INTERIM / "players.parquet").select("PlayerID", "Player")
top = (appearances
    .group_by("PlayerID")
    .agg(pl.len().alias("games"))
    .join(players, on="PlayerID", how="left")
    .sort("games", descending=True)
    .head(20))
print("\nTop 20 by games in our corrected data:")
print(top)