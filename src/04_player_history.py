# 04_player_history

"""
Build player-level history features for each (player, game) appearance.

Output columns:
  PlayerID, GameID, Team, Season, SeasonState, Date,
  career_games_pre      - total games played before this game (full history 1917+)
  career_season_num     - count of distinct seasons with >=1 game, up to & including current
  played_prev_team_game - 1 if player appeared in their team's prior scheduled game, 0 if not, null if first team game
  scored_prev_team_game - 1 if player scored in prev team game (only if played_prev_team_game==1), else 0/null
  games_missed          - count of team games missed since last appearance (0 = consecutive)

Output: data/interim/player_game_history.parquet
"""
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"

appearances = pl.read_parquet(INTERIM / "player_game_appearances.parquet")
events      = pl.read_parquet(INTERIM / "events.parquet")
schedule    = pl.read_parquet(INTERIM / "schedule.parquet")

print(f"Appearances: {appearances.shape}")
print(f"Events:      {events.shape}")
print(f"Schedule:    {schedule.shape}")

# 1. career_games_pre  (cumulative count of prior games)
# Sort by date, then assign a running count per player.
# career_games_pre = row_number - 1 (0 for first game ever)
print("\n[1/5] Computing career_games_pre...")
app = appearances.sort(["PlayerID", "Date", "GameID"])
app = app.with_columns(
    (pl.col("GameID").cum_count().over("PlayerID") - 1)
    .cast(pl.Int32)
    .alias("career_games_pre")
)
# Quick check
print("  Ovechkin career_games_pre in latest game:",
      app.filter(pl.col("PlayerID") == 8471214)
         .sort("Date", descending=True)
         .head(1)["career_games_pre"].to_list())

# 2. career_season_num  (distinct seasons played, including current)
# For each player-game, count how many unique seasons that player has
# appeared in, up to and including the current season.
print("\n[2/5] Computing career_season_num...")

# Build a mapping: for each (PlayerID, Season), what is the season rank?
player_seasons = (app
    .select("PlayerID", "Season")
    .unique()
    .sort(["PlayerID", "Season"])
    .with_columns(
        pl.col("Season").cum_count().over("PlayerID")
          .cast(pl.Int32)
          .alias("career_season_num")
    )
)
app = app.join(player_seasons, on=["PlayerID", "Season"], how="left")

# Quick check
ovi_seasons = (player_seasons
    .filter(pl.col("PlayerID") == 8471214)
    .sort("Season"))
print(f"  Ovechkin seasons played: {len(ovi_seasons)}")
print(f"  Latest career_season_num: {ovi_seasons['career_season_num'].max()}")

# 3. Team schedule chain (for prev-team-game logic)
# For each team + game, find the team's PREVIOUS game by date.
# We need to handle: a team plays in GameID X, their prior game was GameID Y.
print("\n[3/5] Building team schedule chain...")

# Melt schedule to (GameID, Team, Date) — each game appears twice (home + away)
team_games = pl.concat([
    schedule.select(
        "GameID", "Season", "SeasonState", "Date",
        pl.col("HomeTeam").alias("Team")),
    schedule.select(
        "GameID", "Season", "SeasonState", "Date",
        pl.col("AwayTeam").alias("Team")),
]).sort(["Team", "Date", "GameID"])

# For each team game, find the previous game (shift within team partition)
team_games = team_games.with_columns(
    pl.col("GameID").shift(1).over("Team").alias("prev_team_GameID"),
    pl.col("Date").shift(1).over("Team").alias("prev_team_Date"),
)

# Count team-game sequence number (for games_missed calc)
team_games = team_games.with_columns(
    pl.col("GameID").cum_count().over("Team").alias("team_game_seq")
)

team_chain = team_games.select("GameID", "Team", "prev_team_GameID", "team_game_seq")
print(f"  Team-game rows: {len(team_chain):,}")

# 4. played_prev_team_game + scored_prev_team_game + games_missed
print("\n[4/5] Computing prev-team-game flags...")

# Join: for each (player, game, team), get that team's previous GameID
app = app.join(team_chain, on=["GameID", "Team"], how="left")

# Build a set of (PlayerID, GameID) pairs for quick lookup
app_set = app.select("PlayerID", "GameID").unique()

# Check if player appeared in prev_team_GameID
app = app.join(
    app_set.rename({"GameID": "prev_team_GameID"})
           .with_columns(pl.lit(1).cast(pl.Int8).alias("played_prev_team_game")),
    on=["PlayerID", "prev_team_GameID"],
    how="left"
).with_columns(
    pl.when(pl.col("prev_team_GameID").is_null())
      .then(None)  # first game for this team ever
      .otherwise(pl.col("played_prev_team_game").fill_null(0))
      .alias("played_prev_team_game")
)

# games_missed: how many team games between last appearance and this game?
# For each player, get their team_game_seq at previous appearance
player_prev_app = (app
    .select("PlayerID", "Team", "GameID", "team_game_seq", "Date")
    .sort(["PlayerID", "Team", "Date", "GameID"])
    .with_columns(
        pl.col("team_game_seq").shift(1).over(["PlayerID", "Team"]).alias("prev_app_seq")
    )
)
app = app.join(
    player_prev_app.select("PlayerID", "GameID", "prev_app_seq"),
    on=["PlayerID", "GameID"],
    how="left"
)
app = app.with_columns(
    (pl.col("team_game_seq") - pl.col("prev_app_seq") - 1)
    .cast(pl.Int32)
    .alias("games_missed")
)

# scored_prev_team_game: did player score in prev team game?
print("\n[5/5] Computing scored_prev_team_game...")
goals = (events
    .filter(pl.col("Goal") == 1)
    .select(pl.col("Player1_ID").alias("PlayerID"), "GameID")
    .unique()
    .with_columns(pl.lit(1).cast(pl.Int8).alias("_scored"))
)

app = app.join(
    goals.rename({"GameID": "prev_team_GameID"}),
    on=["PlayerID", "prev_team_GameID"],
    how="left"
).with_columns(
    pl.when(pl.col("played_prev_team_game") == 1)
      .then(pl.col("_scored").fill_null(0))
      .otherwise(None)
      .cast(pl.Int8)
      .alias("scored_prev_team_game")
)

# Clean up and save
OUT_COLS = [
    "PlayerID", "GameID", "Team", "Season", "SeasonState", "Date",
    "career_games_pre", "career_season_num",
    "played_prev_team_game", "scored_prev_team_game", "games_missed",
]
result = app.select(OUT_COLS).sort(["PlayerID", "Date", "GameID"])

OUT = INTERIM / "player_game_history.parquet"
result.write_parquet(OUT)
print(f"\nSaved -> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
print(f"Shape: {result.shape}")

# Sanity checks
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# A. Distribution of career_games_pre for 2010-11 starters
modern_start = result.filter(pl.col("Season") == 20102011)
print("\ncareer_games_pre at start of 2010-11 season:")
print(modern_start.group_by("PlayerID").agg(
    pl.col("career_games_pre").min().alias("first_game_cgp")
).select("first_game_cgp").describe())

# B. games_missed distribution
print("\ngames_missed distribution (all data):")
print(result.group_by("games_missed").agg(pl.len().alias("n")).sort("games_missed").head(15))

# C. scored_prev_team_game rate
played = result.filter(pl.col("played_prev_team_game") == 1)
scored_rate = played.filter(pl.col("scored_prev_team_game") == 1).height / played.height
print(f"\nscored_prev_team_game rate (among consecutive appearances): {scored_rate*100:.1f}%")

# D. Spot-check a known player
print("\nSpot-check: Ovechkin last 10 games:")
ovi = result.filter(pl.col("PlayerID") == 8471214).sort("Date", descending=True).head(10)
print(ovi.select(["GameID", "Date", "Team", "career_games_pre", "career_season_num",
                   "played_prev_team_game", "scored_prev_team_game", "games_missed"]))