# 09_figures.py

"""
Thesis figures for Hot Hand analysis.

F1: Distribution of first-goal timing
F2: Corsi rate pre vs post split (scorers vs non-scorers)
F3: Model comparison — IRR forest plot
F4: Robustness summary forest plot
F5: Score-state decomposition (the key finding)
F6: Shot rate by game minute (scorers vs non-scorers)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES   = PROJECT_ROOT / "output" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Style
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

panel = pl.read_parquet(PROCESSED / "analysis_panel.parquet").to_pandas()
panel = panel[panel["time_after_split"] > 0].copy()

# F1: Distribution of first-goal timing
fig, ax = plt.subplots(figsize=(10, 5))
scorers = panel[panel["scored_first_goal"] == 1]
goal_mins = scorers["first_goal_time"] / 60

ax.hist(goal_mins, bins=60, range=(0, 65), color="#2563eb", alpha=0.7,
        edgecolor="white", linewidth=0.5)
ax.axvline(goal_mins.median(), color="#dc2626", linestyle="--", linewidth=2,
           label=f"Median: {goal_mins.median():.1f} min")
for p in [20, 40]:
    ax.axvline(p, color="gray", linestyle=":", alpha=0.5)
    ax.text(p + 0.5, ax.get_ylim()[1] * 0.95, f"Period {p // 20 + 1}",
            fontsize=9, color="gray")
ax.set_xlabel("Game Time (minutes)")
ax.set_ylabel("Count of Player-Games")
ax.set_title("Distribution of First Goal Timing (5v5 + ENF, 2010–2025)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "F1_first_goal_timing.png", bbox_inches="tight")
plt.close()
print("F1 saved")

# F2: Pre vs Post shot rate comparison
panel["corsi_rate_pre_60"] = np.where(
    panel["time_before_split"] > 0,
    panel["corsi_pre_split"] / panel["time_before_split"] * 3600, np.nan)
panel["corsi_rate_post_60"] = np.where(
    panel["time_after_split"] > 0,
    panel["corsi_post_split"] / panel["time_after_split"] * 3600, np.nan)

fig, ax = plt.subplots(figsize=(8, 5))
groups = ["Non-Scorers", "Scorers"]
pre_means = [
    panel[panel["scored_first_goal"] == 0]["corsi_rate_pre_60"].mean(),
    panel[panel["scored_first_goal"] == 1]["corsi_rate_pre_60"].mean(),
]
post_means = [
    panel[panel["scored_first_goal"] == 0]["corsi_rate_post_60"].mean(),
    panel[panel["scored_first_goal"] == 1]["corsi_rate_post_60"].mean(),
]

x = np.arange(len(groups))
w = 0.3
bars1 = ax.bar(x - w/2, pre_means, w, label="Pre-split rate", color="#2563eb", alpha=0.8)
bars2 = ax.bar(x + w/2, post_means, w, label="Post-split rate", color="#dc2626", alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Corsi Attempts per 60 min (5v5)")
ax.set_title("Shot Rate Before vs After Split Point")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()
ax.set_ylim(0, max(pre_means + post_means) * 1.15)
fig.tight_layout()
fig.savefig(FIGURES / "F2_pre_post_rates.png", bbox_inches="tight")
plt.close()
print("F2 saved")

# F3: Model comparison forest plot (IRR)
models = {
    "M1. Poisson":         (0.9383, 0.0037),
    "M2. Neg. Binomial":   (0.9374, 0.0039),
    "M3. NB + Season FE":  (0.9370, 0.0039),
    "M5. Linear FE":       (None, None),  # skip, different scale
}

# With score state
models_ss = {
    "M3 + Score State":    (1.0061, 0.0083),
}

fig, ax = plt.subplots(figsize=(9, 5))
labels = []
irrs = []
cis_low = []
cis_high = []

for name, (irr, se) in {**models, **models_ss}.items():
    if irr is None:
        continue
    labels.append(name)
    irrs.append(irr)
    ci_l = np.exp(np.log(irr) - 1.96 * se)
    ci_h = np.exp(np.log(irr) + 1.96 * se)
    cis_low.append(ci_l)
    cis_high.append(ci_h)

y_pos = np.arange(len(labels))
colors = ["#2563eb"] * 3 + ["#16a34a"]

ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, zorder=0)
for i in range(len(labels)):
    ax.errorbar(irrs[i], y_pos[i],
                xerr=[[irrs[i] - cis_low[i]], [cis_high[i] - irrs[i]]],
                fmt="o", color=colors[i], markersize=8, capsize=4, linewidth=2)
    ax.text(cis_high[i] + 0.003, y_pos[i], f"{irrs[i]:.4f}",
            va="center", fontsize=10)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Incidence Rate Ratio (IRR)")
ax.set_title("Effect of Scoring on Subsequent Shot Rate\n(IRR with 95% CI)")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIGURES / "F3_model_comparison.png", bbox_inches="tight")
plt.close()
print("F3 saved")

# F4: Robustness forest plot
rob_results = {
    "R0. Baseline + SS":        (1.0061, 0.0083, 561051),
    "R2. Forwards":             (1.0081, 0.0043, 371210),
    "R3. Defensemen":           (0.9957, 0.0095, 189841),
    "R4a. Placebo (s=296)":     (0.9977, 0.0037, 561051),
    "R4b. Placebo (s=185)":     (0.9945, 0.0037, 561051),
    "R4c. Placebo (s=142)":     (0.9909, 0.0037, 561051),
    "R5. Early goals":          (1.0101, 0.0041, 540707),
    "R7. Single-goal":          (0.9454, 0.0042, 556590),
    "R8. Consecutive":          (1.0068, 0.0039, 528547),
    "R9. Experienced":          (1.0238, 0.0045, 371721),
    "R10. Recent (2018+)":      (1.0117, 0.0039, 268636),
}

fig, ax = plt.subplots(figsize=(10, 7))
labels_r = list(rob_results.keys())
y_pos_r = np.arange(len(labels_r))

ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, zorder=0)
ax.axvspan(0.99, 1.01, alpha=0.1, color="green", zorder=0)

for i, (name, (irr, se, n)) in enumerate(rob_results.items()):
    ci_l = np.exp(np.log(irr) - 1.96 * se)
    ci_h = np.exp(np.log(irr) + 1.96 * se)
    color = "#dc2626" if (ci_l > 1 or ci_h < 1) else "#2563eb"
    ax.errorbar(irr, y_pos_r[i],
                xerr=[[irr - ci_l], [ci_h - irr]],
                fmt="o", color=color, markersize=7, capsize=3, linewidth=1.5)
    ax.text(ci_h + 0.003, y_pos_r[i], f"{irr:.4f} (n={n:,})",
            va="center", fontsize=9)

ax.set_yticks(y_pos_r)
ax.set_yticklabels(labels_r)
ax.set_xlabel("IRR (with 95% CI)")
ax.set_title("Robustness Checks — All with Score-State Controls\n(Green band = ±1% of null)")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIGURES / "F4_robustness_forest.png", bbox_inches="tight")
plt.close()
print("F4 saved")

# F5: Score-state decomposition
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: without score-state control
ax = axes[0]
vals = [0.9370, 1.0]
labels_d = ["Without\nScore-State", "Null (IRR=1)"]
colors_d = ["#dc2626", "#9ca3af"]
bars = ax.bar(labels_d, vals, color=colors_d, width=0.5, edgecolor="white")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_ylim(0.9, 1.05)
ax.set_ylabel("IRR")
ax.set_title("Model 3: Apparent Cold Hand\n(IRR = 0.937, p < 0.001)")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.4f}",
            ha="center", fontsize=11, fontweight="bold")

# Right: with score-state control
ax = axes[1]
vals2 = [1.0061, 0.8943, 1.0581]
labels_d2 = ["Scored\n(treatment)", "Leading\n(after goal)", "Trailing"]
colors_d2 = ["#16a34a", "#2563eb", "#f59e0b"]
bars2 = ax.bar(labels_d2, vals2, color=colors_d2, width=0.5, edgecolor="white")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_ylim(0.85, 1.1)
ax.set_ylabel("IRR")
ax.set_title("R1: After Score-State Control\n(Scored: IRR = 1.006, p = 0.46)")
for bar, v in zip(bars2, vals2):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.4f}",
            ha="center", fontsize=11, fontweight="bold")

fig.suptitle("The Score-State Decomposition: Key Finding", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "F5_score_state_decomposition.png", bbox_inches="tight")
plt.close()
print("F5 saved")

# F6: Shot rate by period (scorers vs non-scorers)
events = pl.read_parquet(PROJECT_ROOT / "data" / "interim" / "events.parquet")
shots_by_period = (events
    .filter(
        (pl.col("Season") >= 20102011) &
        (pl.col("SeasonState") == "regular") &
        (pl.col("StrengthState") == "5v5") &
        (pl.col("Corsi") == 1) &
        (pl.col("Player1_ID").is_not_null()) &
        (pl.col("Player1_ID") > 0)
    )
    .select(pl.col("Player1_ID").alias("PlayerID"), "GameID", "Period",
            pl.col("Corsi").cast(pl.Int8))
    .group_by(["PlayerID", "GameID", "Period"])
    .agg(pl.col("Corsi").sum().alias("corsi"))
    .to_pandas()
)

# Merge scorer flag
scorer_flag = panel[["PlayerID", "GameID", "scored_first_goal"]].drop_duplicates()
shots_by_period = shots_by_period.merge(scorer_flag, on=["PlayerID", "GameID"], how="inner")
shots_by_period = shots_by_period[shots_by_period["Period"].isin([1, 2, 3])]

fig, ax = plt.subplots(figsize=(8, 5))
for label, val, color in [("Non-Scorers", 0, "#2563eb"), ("Scorers", 1, "#dc2626")]:
    sub = shots_by_period[shots_by_period["scored_first_goal"] == val]
    means = sub.groupby("Period")["corsi"].mean()
    ax.plot(means.index, means.values, "o-", color=color, label=label,
            linewidth=2, markersize=8)

ax.set_xlabel("Period")
ax.set_ylabel("Mean Corsi per Player-Game-Period (5v5)")
ax.set_title("Shot Attempts by Period: Scorers vs Non-Scorers")
ax.set_xticks([1, 2, 3])
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "F6_shots_by_period.png", bbox_inches="tight")
plt.close()
print("F6 saved")

print(f"\nAll figures saved to {FIGURES}")