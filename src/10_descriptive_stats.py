# 10_descriptive_stats.py

"""
Descriptive statistics and visualizations for thesis Chapter 3 (Data).

Sections:
  D1:  Dataset overview (table of all datasets)
  D2:  Event type breakdown (what happens on the ice)
  D3:  Shots and goals per game — league trends over seasons
  D4:  Shot type breakdown (wrist, slap, snap, etc.)
  D5:  Shot location heatmap (ice rink visualization)
  D6:  Goal distribution per player per game
  D7:  Player-level shooting distribution
  D8:  Strength state breakdown (5v5, PP, etc.)
  D9:  Period-level patterns (when do goals happen?)
  D10: xG distribution and calibration
  D11: Score-state and shot volume relationship
  D12: Panel dataset summary statistics table
  D13: Correlation matrix of key variables
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Arc
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM   = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES   = PROJECT_ROOT / "output" / "figures"
TABLES    = PROJECT_ROOT / "output" / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

# Load data
events   = pl.read_parquet(INTERIM / "events.parquet")
schedule = pl.read_parquet(INTERIM / "schedule.parquet")
players  = pl.read_parquet(INTERIM / "players.parquet")
panel    = pl.read_parquet(PROCESSED / "analysis_panel.parquet").to_pandas()

# Filter to modern era
ev = events.filter(
    (pl.col("Season") >= 20102011) &
    (pl.col("SeasonState") == "regular")
)

print(f"Modern events: {len(ev):,}")
print(f"Panel rows:    {len(panel):,}")

# D1: Dataset overview table
print("\n" + "=" * 72)
print("D1: DATASET OVERVIEW")
print("=" * 72)

datasets_info = {
    "NHL_EventData": {"rows": len(events), "cols": events.width,
                      "desc": "Play-by-play events (every on-ice action)"},
    "NHL_Players":   {"rows": len(players), "cols": players.width,
                      "desc": "Player biographical data"},
    "NHL_Schedule":  {"rows": len(schedule), "cols": schedule.width,
                      "desc": "Game schedule with scores"},
    "Analysis Panel": {"rows": len(panel), "cols": panel.shape[1],
                       "desc": "Player × game observations (2010–2025, 5v5)"},
}

overview_df = pd.DataFrame(datasets_info).T
overview_df.to_csv(TABLES / "D1_dataset_overview.csv")
print(overview_df)

# Key numbers for the text
sched_modern = schedule.filter(
    (pl.col("Season") >= 20102011) & (pl.col("SeasonState") == "regular"))
print(f"\nModern era (2010-11 to 2024-25):")
print(f"  Seasons:        15")
print(f"  Regular games:  {len(sched_modern):,}")
print(f"  Unique players: {panel['PlayerID'].nunique():,}")
print(f"  Total events:   {len(ev):,}")

# D2: Event type breakdown
print("\n" + "=" * 72)
print("D2: EVENT TYPE BREAKDOWN")
print("=" * 72)

event_counts = (ev
    .group_by("Event")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .to_pandas()
)
event_counts["pct"] = event_counts["count"] / event_counts["count"].sum() * 100
print(event_counts.to_string(index=False))
event_counts.to_csv(TABLES / "D2_event_types.csv", index=False)

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))
top_events = event_counts.head(10)
colors = ["#2563eb" if e in ["shot-on-goal", "goal", "missed-shot", "blocked-shot"]
          else "#94a3b8" for e in top_events["Event"]]
bars = ax.barh(top_events["Event"][::-1], top_events["count"][::-1],
               color=colors[::-1], edgecolor="white")
ax.set_xlabel("Number of Events (2010–2025, Regular Season)")
ax.set_title("Event Type Frequency in NHL Play-by-Play Data")

# Add count labels
for bar, count in zip(bars, top_events["count"][::-1]):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
            f"{count:,}", va="center", fontsize=9)

# Add legend note
ax.text(0.95, 0.05, "Blue = shot-related events (Corsi)",
        transform=ax.transAxes, ha="right", fontsize=9, color="#2563eb")
fig.tight_layout()
fig.savefig(FIGURES / "D2_event_types.png", bbox_inches="tight")
plt.close()
print("D2 figure saved")

# D3: League trends — shots and goals per game over seasons
print("\n" + "=" * 72)
print("D3: LEAGUE TRENDS")
print("=" * 72)

season_stats = (ev
    .filter(pl.col("StrengthState") == "5v5")
    .group_by("Season")
    .agg(
        pl.col("Corsi").sum().alias("total_corsi"),
        pl.col("Goal").sum().alias("total_goals"),
        pl.col("Shot").sum().alias("total_sog"),
        pl.n_unique("GameID").alias("games"),
    )
    .sort("Season")
    .to_pandas()
)
season_stats["corsi_per_game"] = season_stats["total_corsi"] / season_stats["games"]
season_stats["goals_per_game"] = season_stats["total_goals"] / season_stats["games"]
season_stats["sog_per_game"] = season_stats["total_sog"] / season_stats["games"]
season_stats["season_label"] = season_stats["Season"].astype(str).str[:4] + "-" + season_stats["Season"].astype(str).str[4:6]

print(season_stats[["season_label", "games", "corsi_per_game", "sog_per_game", "goals_per_game"]].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.plot(season_stats["season_label"], season_stats["corsi_per_game"],
        "o-", color="#2563eb", linewidth=2, markersize=5)
ax.set_title("Corsi (Shot Attempts) per Game\nat 5v5")
ax.set_ylabel("Corsi per Game")
ax.tick_params(axis="x", rotation=45)

ax = axes[1]
ax.plot(season_stats["season_label"], season_stats["sog_per_game"],
        "o-", color="#16a34a", linewidth=2, markersize=5)
ax.set_title("Shots on Goal per Game\nat 5v5")
ax.set_ylabel("SOG per Game")
ax.tick_params(axis="x", rotation=45)

ax = axes[2]
ax.plot(season_stats["season_label"], season_stats["goals_per_game"],
        "o-", color="#dc2626", linewidth=2, markersize=5)
ax.set_title("Goals per Game\nat 5v5")
ax.set_ylabel("Goals per Game")
ax.tick_params(axis="x", rotation=45)

fig.suptitle("NHL Shooting Trends (2010–2025, 5v5 Even Strength, Regular Season)",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "D3_league_trends.png", bbox_inches="tight")
plt.close()
print("D3 figure saved")

# D4: Shot type breakdown
print("\n" + "=" * 72)
print("D4: SHOT TYPE BREAKDOWN")
print("=" * 72)

shot_types = (ev
    .filter(
        (pl.col("Corsi") == 1) &
        (pl.col("StrengthState") == "5v5") &
        pl.col("ShotType").is_not_null()
    )
    .group_by("ShotType")
    .agg(
        pl.len().alias("attempts"),
        pl.col("Goal").sum().alias("goals"),
    )
    .sort("attempts", descending=True)
    .to_pandas()
)
shot_types["pct_of_attempts"] = shot_types["attempts"] / shot_types["attempts"].sum() * 100
shot_types["conversion_rate"] = shot_types["goals"] / shot_types["attempts"] * 100
print(shot_types.to_string(index=False))
shot_types.to_csv(TABLES / "D4_shot_types.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: volume
ax = axes[0]
ax.barh(shot_types["ShotType"][::-1], shot_types["attempts"][::-1],
        color="#2563eb", alpha=0.8, edgecolor="white")
ax.set_xlabel("Number of Attempts")
ax.set_title("Shot Attempts by Type (5v5)")
for i, (v, g) in enumerate(zip(shot_types["attempts"][::-1], shot_types["goals"][::-1])):
    ax.text(v + 1000, i, f"{v:,}", va="center", fontsize=9)

# Right: conversion rate
ax = axes[1]
ax.barh(shot_types["ShotType"][::-1], shot_types["conversion_rate"][::-1],
        color="#dc2626", alpha=0.8, edgecolor="white")
ax.set_xlabel("Goal Conversion Rate (%)")
ax.set_title("Scoring Efficiency by Shot Type (5v5)")
for i, v in enumerate(shot_types["conversion_rate"][::-1]):
    ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)

fig.tight_layout()
fig.savefig(FIGURES / "D4_shot_types.png", bbox_inches="tight")
plt.close()
print("D4 figure saved")

# D5: Shot location heatmap on ice rink
print("\n" + "=" * 72)
print("D5: SHOT LOCATION HEATMAP")
print("=" * 72)

shot_locs = (ev
    .filter(
        (pl.col("Corsi") == 1) &
        (pl.col("StrengthState") == "5v5") &
        pl.col("x").is_not_null() &
        pl.col("y").is_not_null()
    )
    .select(
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("Goal").cast(pl.Int8),
    )
    .to_pandas()
)
print(f"  Shots with coordinates: {len(shot_locs):,}")
print(f"  x range: {shot_locs['x'].min():.0f} to {shot_locs['x'].max():.0f}")
print(f"  y range: {shot_locs['y'].min():.0f} to {shot_locs['y'].max():.0f}")

# Normalize: shots from both ends → offensive zone (positive x)
shot_locs["x_abs"] = shot_locs["x"].abs()
shot_locs["y_adj"] = np.where(shot_locs["x"] >= 0, shot_locs["y"], -shot_locs["y"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: all shot attempts
ax = axes[0]
h = ax.hist2d(shot_locs["x_abs"], shot_locs["y_adj"],
              bins=[50, 50], range=[[0, 100], [-42, 42]],
              cmap="Blues", cmin=1)
plt.colorbar(h[3], ax=ax, label="Shot attempts")
# Draw goal crease approximation
circle = plt.Circle((89, 0), 6, fill=False, color="red", linewidth=2)
ax.add_patch(circle)
ax.set_xlabel("Distance from Center Ice (feet)")
ax.set_ylabel("Lateral Position (feet)")
ax.set_title("All Shot Attempts (5v5)")
ax.set_aspect("equal")

# Right: goals only
ax = axes[1]
goals_only = shot_locs[shot_locs["Goal"] == 1]
h2 = ax.hist2d(goals_only["x_abs"], goals_only["y_adj"],
               bins=[50, 50], range=[[0, 100], [-42, 42]],
               cmap="Reds", cmin=1)
plt.colorbar(h2[3], ax=ax, label="Goals")
circle2 = plt.Circle((89, 0), 6, fill=False, color="black", linewidth=2)
ax.add_patch(circle2)
ax.set_xlabel("Distance from Center Ice (feet)")
ax.set_ylabel("Lateral Position (feet)")
ax.set_title("Goals Only (5v5)")
ax.set_aspect("equal")

fig.suptitle("Shot Locations on the Ice (Offensive Half-Rink, 2010–2025)",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "D5_shot_locations.png", bbox_inches="tight")
plt.close()
print("D5 figure saved")

# D6: Distribution of goals per player per game
print("\n" + "=" * 72)
print("D6: GOALS PER PLAYER PER GAME")
print("=" * 72)

panel_clean = panel[panel["time_after_split"] > 0].copy()

goals_dist = panel_clean.groupby("goals_in_game").size().reset_index(name="count")
goals_dist["pct"] = goals_dist["count"] / goals_dist["count"].sum() * 100
print(goals_dist.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(goals_dist["goals_in_game"], goals_dist["pct"],
       color="#2563eb", alpha=0.8, edgecolor="white")
ax.set_xlabel("Goals Scored in Game (5v5 + ENF)")
ax.set_ylabel("Percentage of Player-Games")
ax.set_title("Distribution of Goals per Player per Game\n(2010–2025, Regular Season)")
for i, row in goals_dist.iterrows():
    ax.text(row["goals_in_game"], row["pct"] + 0.5,
            f"{row['pct']:.1f}%\n(n={row['count']:,})",
            ha="center", fontsize=9)
ax.set_xticks(goals_dist["goals_in_game"])
fig.tight_layout()
fig.savefig(FIGURES / "D6_goals_distribution.png", bbox_inches="tight")
plt.close()
print("D6 figure saved")

# D7: Player-level shooting distribution (shots per game)
print("\n" + "=" * 72)
print("D7: PLAYER-LEVEL SHOOTING DISTRIBUTION")
print("=" * 72)

player_avg = (panel_clean
    .groupby("PlayerID")
    .agg(
        games=("GameID", "nunique"),
        total_corsi=("corsi_total", "sum"),
        total_goals=("goals_in_game", "sum"),
        position=("Position", "first"),
    )
)
player_avg["corsi_per_game"] = player_avg["total_corsi"] / player_avg["games"]
player_avg["goals_per_game"] = player_avg["total_goals"] / player_avg["games"]

# Filter to players with 50+ games for stable averages
player_reg = player_avg[player_avg["games"] >= 50]
print(f"  Players with 50+ games: {len(player_reg):,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for pos, color, label in [("F", "#2563eb", "Forwards"), ("D", "#dc2626", "Defensemen")]:
    sub = player_reg[player_reg["position"] == pos]
    ax.hist(sub["corsi_per_game"], bins=30, alpha=0.6, color=color, label=label, edgecolor="white")
ax.set_xlabel("Average Corsi per Game (5v5)")
ax.set_ylabel("Number of Players")
ax.set_title("Distribution of Player Shooting Rates\n(Players with 50+ games)")
ax.legend()
ax.axvline(player_reg["corsi_per_game"].median(), color="gray", linestyle="--",
           label=f"Median: {player_reg['corsi_per_game'].median():.2f}")

ax = axes[1]
for pos, color, label in [("F", "#2563eb", "Forwards"), ("D", "#dc2626", "Defensemen")]:
    sub = player_reg[player_reg["position"] == pos]
    ax.hist(sub["goals_per_game"], bins=30, alpha=0.6, color=color, label=label, edgecolor="white")
ax.set_xlabel("Average Goals per Game (5v5 + ENF)")
ax.set_ylabel("Number of Players")
ax.set_title("Distribution of Player Scoring Rates\n(Players with 50+ games)")
ax.legend()

fig.tight_layout()
fig.savefig(FIGURES / "D7_player_distributions.png", bbox_inches="tight")
plt.close()
print("D7 figure saved")

# D8: Strength state breakdown
print("\n" + "=" * 72)
print("D8: STRENGTH STATE BREAKDOWN")
print("=" * 72)

strength = (ev
    .filter(pl.col("Corsi") == 1)
    .group_by("StrengthState")
    .agg(
        pl.len().alias("attempts"),
        pl.col("Goal").sum().alias("goals"),
    )
    .sort("attempts", descending=True)
    .to_pandas()
)
strength["pct_attempts"] = strength["attempts"] / strength["attempts"].sum() * 100
strength["conv_rate"] = strength["goals"] / strength["attempts"] * 100
print(strength.to_string(index=False))
strength.to_csv(TABLES / "D8_strength_states.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
top_str = strength.head(8)
colors_str = ["#2563eb" if s == "5v5" else "#f59e0b" if "5v4" in str(s) or "5v3" in str(s)
              else "#94a3b8" for s in top_str["StrengthState"]]
ax.barh(top_str["StrengthState"][::-1], top_str["pct_attempts"][::-1],
        color=colors_str[::-1], edgecolor="white")
ax.set_xlabel("% of All Shot Attempts")
ax.set_title("Shot Attempts by Strength State\n(2010–2025, Regular Season)")
for i, (v, n) in enumerate(zip(top_str["pct_attempts"][::-1], top_str["attempts"][::-1])):
    ax.text(v + 0.3, i, f"{v:.1f}% ({n:,})", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "D8_strength_states.png", bbox_inches="tight")
plt.close()
print("D8 figure saved")

# D9: When do goals happen? (by game minute)
print("\n" + "=" * 72)
print("D9: GOAL TIMING DISTRIBUTION")
print("=" * 72)

goal_timing = (ev
    .filter(
        (pl.col("Goal") == 1) &
        (pl.col("StrengthState") == "5v5") &
        pl.col("GameTime").is_not_null()
    )
    .select((pl.col("GameTime") / 60).alias("minute"))
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(goal_timing["minute"], bins=60, range=(0, 65),
        color="#dc2626", alpha=0.7, edgecolor="white", linewidth=0.5)
for p in [20, 40]:
    ax.axvline(p, color="gray", linestyle=":", linewidth=2)
    ax.text(p + 0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 100,
            f"Period {p // 20 + 1}", fontsize=10, color="gray")
ax.set_xlabel("Game Minute")
ax.set_ylabel("Number of Goals")
ax.set_title("When Do Goals Happen? (5v5, 2010–2025)")
fig.tight_layout()
fig.savefig(FIGURES / "D9_goal_timing.png", bbox_inches="tight")
plt.close()
print("D9 figure saved")

# D10: Expected Goals (xG) distribution
print("\n" + "=" * 72)
print("D10: xG DISTRIBUTION")
print("=" * 72)

xg_data = (ev
    .filter(
        (pl.col("Corsi") == 1) &
        (pl.col("StrengthState") == "5v5") &
        pl.col("xG_F").is_not_null()
    )
    .select(
        pl.col("xG_F").cast(pl.Float64),
        pl.col("Goal").cast(pl.Int8),
    )
    .to_pandas()
)
print(f"  Shots with xG: {len(xg_data):,}")
print(f"  xG mean: {xg_data['xG_F'].mean():.4f}")
print(f"  xG median: {xg_data['xG_F'].median():.4f}")
print(f"  Actual goal rate: {xg_data['Goal'].mean():.4f}")
print(f"  Calibration (xG mean / actual rate): {xg_data['xG_F'].mean() / xg_data['Goal'].mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: xG distribution
ax = axes[0]
ax.hist(xg_data["xG_F"], bins=100, range=(0, 0.5), color="#2563eb", alpha=0.7,
        edgecolor="white", linewidth=0.3)
ax.axvline(xg_data["xG_F"].mean(), color="#dc2626", linestyle="--",
           label=f"Mean xG: {xg_data['xG_F'].mean():.4f}")
ax.axvline(xg_data["xG_F"].median(), color="#16a34a", linestyle="--",
           label=f"Median xG: {xg_data['xG_F'].median():.4f}")
ax.set_xlabel("Expected Goals (xG)")
ax.set_ylabel("Count")
ax.set_title("Distribution of xG per Shot Attempt (5v5)")
ax.legend()

# Right: xG calibration — binned actual vs expected
ax = axes[1]
xg_data["xg_bin"] = pd.cut(xg_data["xG_F"], bins=20, labels=False)
calib = xg_data.groupby("xg_bin").agg(
    mean_xg=("xG_F", "mean"),
    actual_rate=("Goal", "mean"),
    count=("Goal", "count"),
).dropna()
ax.scatter(calib["mean_xg"], calib["actual_rate"], s=calib["count"]/100,
           color="#2563eb", alpha=0.7, edgecolors="white")
ax.plot([0, 0.5], [0, 0.5], "k--", alpha=0.5, label="Perfect calibration")
ax.set_xlabel("Mean Predicted xG (binned)")
ax.set_ylabel("Actual Goal Rate")
ax.set_title("xG Calibration Plot\n(closer to diagonal = better model)")
ax.legend()
ax.set_xlim(0, 0.45)
ax.set_ylim(0, 0.45)

fig.tight_layout()
fig.savefig(FIGURES / "D10_xG_distribution.png", bbox_inches="tight")
plt.close()
print("D10 figure saved")

# D11: Score state and shot volume
print("\n" + "=" * 72)
print("D11: SCORE STATE AND SHOT VOLUME")
print("=" * 72)

score_shots = (ev
    .filter(
        (pl.col("Corsi") == 1) &
        (pl.col("StrengthState") == "5v5") &
        pl.col("ScoreState").is_not_null()
    )
    .select(
        pl.col("ScoreState").cast(pl.Int64, strict=False).alias("score_state"),
        pl.col("Corsi").cast(pl.Int8),
        "GameID", "Period",
    )
    .drop_nulls("score_state")
    .to_pandas()
)

# Aggregate: mean Corsi per period by score state
ss_agg = (score_shots
    .groupby("score_state")
    .agg(
        attempts=("Corsi", "sum"),
        periods=("GameID", "count"),
    )
)
ss_agg["rate"] = ss_agg["attempts"] / ss_agg["periods"]
ss_agg = ss_agg[(ss_agg.index >= -3) & (ss_agg.index <= 3)].sort_index()

fig, ax = plt.subplots(figsize=(9, 5))
colors_ss = ["#dc2626" if i < 0 else "#16a34a" if i > 0 else "#94a3b8"
             for i in ss_agg.index]
ax.bar(ss_agg.index, ss_agg["rate"], color=colors_ss, edgecolor="white", width=0.7)
ax.set_xlabel("Score State (team's perspective: +1 = leading by 1, −1 = trailing by 1)")
ax.set_ylabel("Shot Attempts per Event Window")
ax.set_title("Shot Volume by Score State (5v5)\nTrailing Teams Shoot More — Leading Teams Protect Leads")
ax.set_xticks(range(-3, 4))
ax.set_xticklabels([f"{i:+d}" if i != 0 else "Tied" for i in range(-3, 4)])
fig.tight_layout()
fig.savefig(FIGURES / "D11_score_state_shots.png", bbox_inches="tight")
plt.close()
print("D11 figure saved")

# D12: Panel summary statistics table
print("\n" + "=" * 72)
print("D12: PANEL SUMMARY STATISTICS")
print("=" * 72)

desc_cols = [
    "corsi_total", "corsi_pre_split", "corsi_post_split",
    "sog_total", "goals_5v5", "goals_in_game",
    "first_goal_time", "time_after_split", "time_before_split",
    "xg_per_shot_total", "xg_total",
    "career_games_pre", "career_season_num",
    "scored_first_goal", "scored_prev_team_game",
    "games_missed",
]

panel_desc = panel[desc_cols].describe().T
panel_desc["non_null"] = panel[desc_cols].notna().sum()
panel_desc = panel_desc[["non_null", "mean", "std", "min", "25%", "50%", "75%", "max"]]
panel_desc.columns = ["N", "Mean", "Std", "Min", "P25", "Median", "P75", "Max"]
print(panel_desc.round(3).to_string())
panel_desc.round(3).to_csv(TABLES / "D12_summary_statistics.csv")
print("\nSaved to tables/D12_summary_statistics.csv")

# D13: Correlation matrix of key variables
print("\n" + "=" * 72)
print("D13: CORRELATION MATRIX")
print("=" * 72)

corr_cols = ["corsi_total", "corsi_post_split", "goals_in_game",
             "scored_first_goal", "career_games_pre", "career_season_num",
             "xg_per_shot_total", "scored_prev_team_game"]
corr_labels = ["Corsi Total", "Corsi Post", "Goals", "Scored 1st",
               "Career GP", "Career Szn", "xG/Shot", "Scored Prev"]

corr_matrix = panel[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_labels)))
ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha="right")
ax.set_yticklabels(corr_labels)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = corr_matrix.values[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")
ax.set_title("Correlation Matrix of Key Analysis Variables")
fig.tight_layout()
fig.savefig(FIGURES / "D13_correlation_matrix.png", bbox_inches="tight")
plt.close()
print("D13 figure saved")

# D14: Treatment vs control group comparison table
print("\n" + "=" * 72)
print("D14: TREATMENT vs CONTROL COMPARISON")
print("=" * 72)

compare_cols = ["corsi_total", "sog_total", "xg_per_shot_total", "xg_total",
                "career_games_pre", "career_season_num",
                "corsi_pre_split", "corsi_post_split"]

scorers = panel[panel["scored_first_goal"] == 1]
non_scorers = panel[panel["scored_first_goal"] == 0]

comp = pd.DataFrame({
    "Variable": compare_cols,
    "Scorers_mean": [scorers[c].mean() for c in compare_cols],
    "Scorers_std": [scorers[c].std() for c in compare_cols],
    "NonScorers_mean": [non_scorers[c].mean() for c in compare_cols],
    "NonScorers_std": [non_scorers[c].std() for c in compare_cols],
    "Diff": [scorers[c].mean() - non_scorers[c].mean() for c in compare_cols],
})
print(comp.round(3).to_string(index=False))
comp.round(3).to_csv(TABLES / "D14_treatment_control.csv", index=False)

print(f"\n{'='*72}")
print("ALL DESCRIPTIVE STATS COMPLETE")
print(f"{'='*72}")
print(f"Figures saved to: {FIGURES}")
print(f"Tables saved to:  {TABLES}")
print(f"Total figures: {len(list(FIGURES.glob('*.png')))}")
print(f"Total tables:  {len(list(TABLES.glob('*.csv')))}")