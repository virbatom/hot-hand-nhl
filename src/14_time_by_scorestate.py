# 14_time_by_scorestate.py

"""
14_time_by_scorestate.py

For each (player, game), compute the share of post-split time in each
score state, TWO WAYS:
  TEAM (game-clock):  pct_time_leading / tied / trailing
                      (over [split_time, split_time+time_after_split])
  PLAYER (on-ice):    pct_onice_leading / tied / trailing
                      (over the player's 5v5 shifts within that window)

Score state = team perspective, changes at EVERY goal (any strength).

Adds to analysis_panel_ss.parquet:
  secs_post_leading/tied/trailing,  pct_time_leading/tied/trailing
  secs_onice_leading/tied/trailing, pct_onice_leading/tied/trailing

RUN ORDER: 12_shots_by_scorestate.py -> 14_time_by_scorestate.py
"""
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM   = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# 1. Score-differential timeline (home perspective) per game
print("Building score timeline from goals...")
events   = pl.read_parquet(INTERIM / "events.parquet")
schedule = pl.read_parquet(INTERIM / "schedule.parquet")
sched    = schedule.select("GameID", "HomeTeam")

goals = (events.filter(pl.col("Goal") == 1)
         .select("GameID", "GameTime", "EventTeam")
         .filter(pl.col("EventTeam").is_not_null())
         .join(sched, on="GameID", how="inner")
         .with_columns((pl.col("EventTeam") == pl.col("HomeTeam")).cast(pl.Int64).alias("hg"))
         .sort(["GameID", "GameTime"]))
goals = (goals.with_columns([
            pl.col("hg").cum_sum().over("GameID").alias("hs"),
            (1 - pl.col("hg")).cum_sum().over("GameID").alias("as_")])
         .with_columns((pl.col("hs") - pl.col("as_")).alias("home_diff"))
         .select("GameID",
                 pl.col("GameTime").cast(pl.Int64),
                 pl.col("home_diff").cast(pl.Int64)))

starts = (goals.select("GameID").unique().with_columns([
            pl.lit(0, dtype=pl.Int64).alias("GameTime"),
            pl.lit(0, dtype=pl.Int64).alias("home_diff")]))

timeline = (pl.concat([starts, goals]).sort(["GameID", "GameTime"])
            .with_columns(pl.col("GameTime").shift(-1).over("GameID").alias("seg_end"))
            .with_columns(pl.col("seg_end").fill_null(10_000_000))
            .rename({"GameTime": "seg_start"}))
print(f"  Timeline segments: {len(timeline):,}")

# 2. Player windows (post-split)
print("Preparing player windows...")
panel = pl.read_parquet(PROCESSED / "analysis_panel_ss.parquet")
appr  = (pl.read_parquet(INTERIM / "player_game_appearances.parquet")
         .select("PlayerID", "GameID", "Team").unique())

windows = (panel.select("PlayerID", "GameID", "Season", "split_time", "time_after_split").unique()
           .join(appr, on=["PlayerID", "GameID"], how="left")
           .join(sched, on="GameID", how="left")
           .with_columns((pl.col("Team") == pl.col("HomeTeam")).alias("is_home"))
           .with_columns((pl.col("split_time") + pl.col("time_after_split")).alias("win_end"))
           .rename({"split_time": "win_start"}))
print(f"  Windows: {len(windows):,}  unknown home/away: "
      f"{windows.filter(pl.col('is_home').is_null()).height:,}")

def classify(df, diff_col):
    return (df.with_columns(
                pl.when(pl.col("is_home")).then(pl.col(diff_col))
                  .otherwise(-pl.col(diff_col)).alias("td"))
              .with_columns(
                pl.when(pl.col("td") > 0).then(pl.lit("leading"))
                  .when(pl.col("td") < 0).then(pl.lit("trailing"))
                  .otherwise(pl.lit("tied")).alias("state")))

def pivot_secs(df, prefix):
    out = (df.group_by(["PlayerID", "GameID", "state"])
             .agg(pl.col("ov").sum().alias("secs"))
             .pivot(values="secs", index=["PlayerID", "GameID"],
                    on="state", aggregate_function="sum").fill_null(0))
    for s in ["leading", "tied", "trailing"]:
        if s not in out.columns:
            out = out.with_columns(pl.lit(0).alias(s))
    return out.rename({"leading": f"secs_{prefix}_leading",
                       "tied":    f"secs_{prefix}_tied",
                       "trailing":f"secs_{prefix}_trailing"})

# 3a. TEAM-level (game-clock): window x timeline
print("Team-level: intersecting windows with score segments...")
team = (windows.join(timeline, on="GameID", how="inner")
        .with_columns(
            (pl.min_horizontal("win_end", "seg_end") - pl.max_horizontal("win_start", "seg_start"))
            .clip(lower_bound=0).alias("ov"))
        .filter(pl.col("ov") > 0))
team = pivot_secs(classify(team, "home_diff"), "post")   # -> secs_post_*

# 3b. PLAYER-level (on-ice): explode 5v5 shifts, clip, intersect
#     Done SEASON-BY-SEASON to keep memory bounded.
print("Player-level: exploding 5v5 shifts (season-chunked)...")
shifts = pl.read_parquet(INTERIM / "shifts.parquet")
for c in ["Home_StrengthState", "Away_StrengthState"]:
    if shifts.schema[c] == pl.Utf8:
        shifts = shifts.with_columns(
            pl.when(pl.col(c) == r"\N").then(None).otherwise(pl.col(c)).alias(c))
shifts = shifts.join(schedule.select("GameID", "Season", "SeasonState"), on="GameID", how="inner")
shifts5 = shifts.filter(
    (pl.col("Season") >= 20102011) & (pl.col("SeasonState") == "regular") &
    (pl.col("Home_StrengthState") == "5v5") & (pl.col("Away_StrengthState") == "5v5"))

STR_COLS = ["Home_Forwards_ID", "Home_Defenders_ID", "Away_Forwards_ID", "Away_Defenders_ID"]
INT_COLS = ["Home_Goalie_ID", "Away_Goalie_ID"]
for c in STR_COLS:
    if shifts5.schema[c] == pl.Utf8:
        shifts5 = shifts5.with_columns(
            pl.when(pl.col(c) == r"\N").then(None).otherwise(pl.col(c)).alias(c))

def expl_str(df, col):
    return (df.filter(pl.col(col).is_not_null() & (pl.col(col).str.len_chars() > 0))
            .select("GameID", "Start", "End", pl.col(col).alias("raw"))
            .with_columns(pl.col("raw").str.split(" ").alias("ids")).explode("ids")
            .filter(pl.col("ids").str.len_chars() > 0)
            .with_columns(pl.col("ids").str.strip_chars().cast(pl.Int64, strict=False).alias("PlayerID"))
            .filter(pl.col("PlayerID").is_not_null() & (pl.col("PlayerID") > 0))
            .select("PlayerID", "GameID", "Start", "End"))
def expl_int(df, col):
    return (df.filter(pl.col(col).is_not_null() & (pl.col(col) > 0))
            .select(pl.col(col).alias("PlayerID"), "GameID", "Start", "End"))

tl_small = timeline.select("GameID", "seg_start", "seg_end", "home_diff")
seasons = sorted(windows.select("Season").unique().to_series().to_list())
onice_parts = []
for s in seasons:
    w_s  = windows.filter(pl.col("Season") == s) \
                  .select("PlayerID", "GameID", "win_start", "win_end", "is_home")
    sh_s = shifts5.filter(pl.col("Season") == s)
    psh_s = pl.concat([expl_str(sh_s, c) for c in STR_COLS] +
                      [expl_int(sh_s, c) for c in INT_COLS]).unique()
    ps_s = (psh_s.join(w_s, on=["PlayerID", "GameID"], how="inner")
            .with_columns([pl.max_horizontal("Start", "win_start").alias("on_start"),
                           pl.min_horizontal("End", "win_end").alias("on_end")])
            .filter(pl.col("on_end") > pl.col("on_start"))
            .select("PlayerID", "GameID", "on_start", "on_end", "is_home"))
    pj_s = (ps_s.join(tl_small, on="GameID", how="inner")
            .with_columns(
                (pl.min_horizontal("on_end", "seg_end") - pl.max_horizontal("on_start", "seg_start"))
                .clip(lower_bound=0).alias("ov"))
            .filter(pl.col("ov") > 0)
            .with_columns(
                pl.when(pl.col("is_home")).then(pl.col("home_diff"))
                  .otherwise(-pl.col("home_diff")).alias("td"))
            .with_columns(
                pl.when(pl.col("td") > 0).then(pl.lit("leading"))
                  .when(pl.col("td") < 0).then(pl.lit("trailing"))
                  .otherwise(pl.lit("tied")).alias("state")))
    onice_parts.append(pj_s.group_by(["PlayerID", "GameID", "state"])
                            .agg(pl.col("ov").sum().alias("secs")))
    print(f"  season {s}: ok")

onice = (pl.concat(onice_parts)
         .pivot(values="secs", index=["PlayerID", "GameID"],
                on="state", aggregate_function="sum").fill_null(0))
for st in ["leading", "tied", "trailing"]:
    if st not in onice.columns:
        onice = onice.with_columns(pl.lit(0).alias(st))
onice = onice.rename({"leading": "secs_onice_leading",
                      "tied": "secs_onice_tied",
                      "trailing": "secs_onice_trailing"})

# 4. Join both, compute %, validate, save
print("Joining, computing percentages, validating...")
new_cols = ["secs_post_leading","secs_post_tied","secs_post_trailing",
            "pct_time_leading","pct_time_tied","pct_time_trailing",
            "secs_onice_leading","secs_onice_tied","secs_onice_trailing",
            "pct_onice_leading","pct_onice_tied","pct_onice_trailing"]
panel = panel.drop([c for c in panel.columns if c in new_cols])
panel = (panel.join(team,  on=["PlayerID","GameID"], how="left")
              .join(onice, on=["PlayerID","GameID"], how="left"))
for c in ["secs_post_leading","secs_post_tied","secs_post_trailing",
          "secs_onice_leading","secs_onice_tied","secs_onice_trailing"]:
    panel = panel.with_columns(pl.col(c).fill_null(0))

panel = panel.with_columns([
    (pl.col("secs_post_leading")+pl.col("secs_post_tied")+pl.col("secs_post_trailing")).alias("_tt"),
    (pl.col("secs_onice_leading")+pl.col("secs_onice_tied")+pl.col("secs_onice_trailing")).alias("_to"),
])
for st in ["leading","tied","trailing"]:
    panel = panel.with_columns([
        pl.when(pl.col("_tt") > 0).then(pl.col(f"secs_post_{st}")/pl.col("_tt"))
          .otherwise(0.0).alias(f"pct_time_{st}"),
        pl.when(pl.col("_to") > 0).then(pl.col(f"secs_onice_{st}")/pl.col("_to"))
          .otherwise(0.0).alias(f"pct_onice_{st}"),
    ])

# validation
team_bad  = panel.with_columns((pl.col("_tt")-pl.col("time_after_split")).abs().alias("d")) \
                 .filter(pl.col("d") > 2).height
onice_bad = panel.with_columns((pl.col("_to")-pl.col("toi_5v5_post")).abs().alias("d")) \
                  .filter(pl.col("d") > 2).height
print(f"  TEAM  time-sum != time_after_split (>2s): {team_bad:,} ({team_bad/len(panel)*100:.2f}%)")
print(f"  ONICE time-sum != toi_5v5_post     (>2s): {onice_bad:,} ({onice_bad/len(panel)*100:.2f}%)")
panel = panel.drop(["_tt", "_to"])

OUT = PROCESSED / "analysis_panel_ss.parquet"
panel.write_parquet(OUT)
print(f"\nSaved -> {OUT}\nShape: {panel.shape}")

# Sanity checks (verify home/away orientation)
print("\n" + "=" * 60 + "\nSANITY CHECKS\n" + "=" * 60)
for grp, name in [(1, "SCORERS (should lean LEADING)"), (0, "NON-SCORERS")]:
    sub = panel.filter(pl.col("scored_first_goal") == grp)
    print(f"\n{name}:")
    print("  TEAM :", sub.select([
        pl.col("pct_time_leading").mean().round(3).alias("lead"),
        pl.col("pct_time_tied").mean().round(3).alias("tied"),
        pl.col("pct_time_trailing").mean().round(3).alias("trail")]).to_dicts()[0])
    print("  ONICE:", sub.select([
        pl.col("pct_onice_leading").mean().round(3).alias("lead"),
        pl.col("pct_onice_tied").mean().round(3).alias("tied"),
        pl.col("pct_onice_trailing").mean().round(3).alias("trail")]).to_dicts()[0])