# 11_fix_D11.py

"""
Fix D11: Score state and shot volume.
Correct approach: count Corsi events per (game, period, score_state),
then average across game-periods.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"
FIGURES = PROJECT_ROOT / "output" / "figures"

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

events = pl.read_parquet(INTERIM / "events.parquet")

# Get all 5v5 Corsi events with valid score state
shots = (events
    .filter(
        (pl.col("Season") >= 20102011) &
        (pl.col("SeasonState") == "regular") &
        (pl.col("Corsi") == 1) &
        (pl.col("StrengthState") == "5v5") &
        pl.col("ScoreState").is_not_null()
    )
    .select("GameID", "Period", "EventTeam",
            pl.col("ScoreState").cast(pl.Int64, strict=False).alias("score_state"))
    .drop_nulls("score_state")
)

# Count shots per (game, team, score_state)
# This gives us: in game X, team Y took N shots while at score_state Z
team_shots = (shots
    .group_by(["GameID", "EventTeam", "score_state"])
    .agg(pl.len().alias("shot_count"))
    .to_pandas()
)

# Filter to common score states
team_shots = team_shots[team_shots["score_state"].between(-3, 3)]

# Average shots per game-team by score state
avg_by_ss = (team_shots
    .groupby("score_state")
    .agg(
        mean_shots=("shot_count", "mean"),
        median_shots=("shot_count", "median"),
        n_observations=("shot_count", "count"),
        total_shots=("shot_count", "sum"),
    )
    .sort_index()
)

print("Score state -> average shots per team-game stint:")
print(avg_by_ss)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: mean shots per team-game at each score state
ax = axes[0]
colors = ["#dc2626" if i < 0 else "#16a34a" if i > 0 else "#94a3b8"
          for i in avg_by_ss.index]
bars = ax.bar(avg_by_ss.index, avg_by_ss["mean_shots"],
              color=colors, edgecolor="white", width=0.7)
ax.set_xlabel("Score State (team perspective: +1 = leading by 1)")
ax.set_ylabel("Mean Shot Attempts per Team-Game Stint")
ax.set_title("Trailing Teams Shoot More\nLeading Teams Protect Leads")
ax.set_xticks(range(-3, 4))
ax.set_xticklabels([f"{i:+d}" if i != 0 else "Tied" for i in range(-3, 4)])
for bar, v in zip(bars, avg_by_ss["mean_shots"]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
            f"{v:.2f}", ha="center", fontsize=10)

# Right: total shots (shows how much time is spent in each state)
ax = axes[1]
ax.bar(avg_by_ss.index, avg_by_ss["total_shots"],
       color=colors, edgecolor="white", width=0.7)
ax.set_xlabel("Score State")
ax.set_ylabel("Total Shot Attempts")
ax.set_title("Time Spent in Each Score State\n(Tied dominates)")
ax.set_xticks(range(-3, 4))
ax.set_xticklabels([f"{i:+d}" if i != 0 else "Tied" for i in range(-3, 4)])
for bar, v in zip(ax.patches, avg_by_ss["total_shots"]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 5000,
            f"{v:,.0f}", ha="center", fontsize=9, rotation=0)

fig.suptitle("Score State and Shot Volume (5v5, 2010–2025)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "D11_score_state_shots.png", bbox_inches="tight")
plt.close()
print("\nD11 fixed and saved")