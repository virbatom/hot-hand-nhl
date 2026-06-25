# 09b_figures_toi.py

"""
Updated thesis figures incorporating TOI-based results.

F1:  First goal timing distribution (unchanged)
F2:  Pre vs Post shot rate — TOI-based
F3:  Model comparison forest plot — both exposures
F4:  Robustness forest plot — TOI exposure
F5:  Score-state decomposition — both exposures
F6:  Shots by period (unchanged)
F7:  NEW: Exposure comparison (game-clock vs TOI)
F8:  NEW: TOI distribution by scorer status
F9:  NEW: Combined robustness — game-clock vs TOI side by side
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES   = PROJECT_ROOT / "output" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

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
panel = panel[panel["toi_5v5_post"] > 0].copy()

# F2b: Pre vs Post shot rate — TOI-based
print("F2b: Pre vs Post shot rate (TOI)...")

panel["rate_pre_toi"] = np.where(
    panel["toi_5v5_pre"] > 0,
    panel["corsi_pre_split"] / panel["toi_5v5_pre"] * 3600, np.nan)
panel["rate_post_toi"] = np.where(
    panel["toi_5v5_post"] > 0,
    panel["corsi_post_split"] / panel["toi_5v5_post"] * 3600, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Game-clock rates
ax = axes[0]
panel["rate_pre_gc"] = np.where(
    panel["time_before_split"] > 0,
    panel["corsi_pre_split"] / panel["time_before_split"] * 3600, np.nan)
panel["rate_post_gc"] = np.where(
    panel["time_after_split"] > 0,
    panel["corsi_post_split"] / panel["time_after_split"] * 3600, np.nan)

groups = ["Non-Scorers", "Scorers"]
for i, (label, filt) in enumerate([(0, panel["scored_first_goal"]==0), (1, panel["scored_first_goal"]==1)]):
    sub = panel[filt]
    pre = sub["rate_pre_gc"].mean()
    post = sub["rate_post_gc"].mean()
    x_pos = i
    b1 = ax.bar(x_pos - 0.15, pre, 0.28, color="#2563eb", alpha=0.8,
                label="Pre-split" if i==0 else "")
    b2 = ax.bar(x_pos + 0.15, post, 0.28, color="#dc2626", alpha=0.8,
                label="Post-split" if i==0 else "")
    ax.text(x_pos - 0.15, pre + 0.03, f"{pre:.2f}", ha="center", fontsize=10)
    ax.text(x_pos + 0.15, post + 0.03, f"{post:.2f}", ha="center", fontsize=10)

ax.set_ylabel("Corsi per 60 min")
ax.set_title("Game-Clock Exposure\n(per 60 min game time)")
ax.set_xticks([0, 1])
ax.set_xticklabels(groups)
ax.legend()
ax.set_ylim(0, 3.6)

# Right: TOI rates
ax = axes[1]
for i, (label, filt) in enumerate([(0, panel["scored_first_goal"]==0), (1, panel["scored_first_goal"]==1)]):
    sub = panel[filt]
    pre = sub["rate_pre_toi"].dropna().mean()
    post = sub["rate_post_toi"].dropna().mean()
    x_pos = i
    b1 = ax.bar(x_pos - 0.15, pre, 0.28, color="#2563eb", alpha=0.8,
                label="Pre-split" if i==0 else "")
    b2 = ax.bar(x_pos + 0.15, post, 0.28, color="#dc2626", alpha=0.8,
                label="Post-split" if i==0 else "")
    ax.text(x_pos - 0.15, pre + 0.15, f"{pre:.1f}", ha="center", fontsize=10)
    ax.text(x_pos + 0.15, post + 0.15, f"{post:.1f}", ha="center", fontsize=10)

ax.set_ylabel("Corsi per 60 min")
ax.set_title("TOI Exposure\n(per 60 min actual ice time)")
ax.set_xticks([0, 1])
ax.set_xticklabels(groups)
ax.legend()

fig.suptitle("Shot Rate Before vs After Split Point: Two Exposure Definitions", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "F2b_pre_post_rates_toi.png", bbox_inches="tight")
plt.close()
print("  saved")

# F3b: Model comparison forest plot — both exposures
print("F3b: Model comparison forest plot (both exposures)...")

models_data = [
    # (label, IRR, SE, color, group)
    ("M1. Poisson (game-clock)", 0.9383, 0.0037, "#94a3b8", "Game-clock"),
    ("M2. NB (game-clock)", 0.9374, 0.0039, "#94a3b8", "Game-clock"),
    ("M3. NB+Season (game-clock)", 0.9370, 0.0039, "#94a3b8", "Game-clock"),
    ("M3+SS (game-clock)", 1.0061, 0.0083, "#16a34a", "Game-clock"),
    ("", None, None, None, "spacer"),
    ("M1. Poisson (TOI)", 0.9059, 0.0037, "#60a5fa", "TOI"),
    ("M2. NB (TOI)", 0.9055, 0.0038, "#60a5fa", "TOI"),
    ("M3. NB+Season (TOI)", 0.9050, 0.0038, "#60a5fa", "TOI"),
    ("M3+SS (TOI)", 0.9444, 0.0080, "#dc2626", "TOI"),
]

fig, ax = plt.subplots(figsize=(11, 7))
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, zorder=0)

# assign decreasing y so list order reads top->bottom; spacer leaves a gap
y = len(models_data) - 1
ticks, ticklabels = [], []
for label, irr, se, color, group in models_data:
    if group == "spacer":
        y -= 1
        continue
    ci_l = np.exp(np.log(irr) - 1.96 * se)
    ci_h = np.exp(np.log(irr) + 1.96 * se)
    ax.errorbar(irr, y, xerr=[[irr - ci_l], [ci_h - irr]],
                fmt="o", color=color, markersize=8, capsize=4, linewidth=2)
    ax.text(ci_h + 0.004, y, f"{irr:.4f}", va="center", fontsize=10, color=color)
    ticks.append(y)
    ticklabels.append(label)
    y -= 1

ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels)
ax.set_xlabel("Incidence Rate Ratio (IRR)")
ax.set_title("Effect of Scoring on Subsequent Shot Rate\nGame-Clock vs TOI Exposure (IRR with 95% CI)")

# Add legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#94a3b8", label="Game-clock (no score state)"),
    Patch(facecolor="#16a34a", label="Game-clock + score state"),
    Patch(facecolor="#60a5fa", label="TOI (no score state)"),
    Patch(facecolor="#dc2626", label="TOI + score state"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "F3b_model_comparison_both.png", bbox_inches="tight")
plt.close()
print("  saved")

# F4b: Robustness forest plot — TOI exposure
print("F4b: Robustness forest plot (TOI)...")

rob_toi = {
    "R0. Baseline + SS":     (0.9444, 0.0080, 557355),
    "R2. Forwards":          (0.9431, 0.0043, 368304),
    "R3. Defensemen":        (0.9505, 0.0095, 189051),
    "R4a. Placebo (s=296)":  (1.0048, 0.0038, 557355),
    "R4b. Placebo (s=185)":  (1.0007, 0.0038, 557355),
    "R4c. Placebo (s=142)":  (0.9964, 0.0038, 557355),
    "R5. Early goals":       (0.9528, 0.0041, 539187),
    "R7. Single-goal":       (0.8894, 0.0042, 552899),
    "R8. Consecutive":       (0.9469, 0.0039, 525043),
    "R9. Experienced":       (0.9712, 0.0045, 369191),
    "R10. Recent (2018+)":   (0.9524, 0.0039, 266927),
    "R11. No score state":   (0.9050, 0.0038, 557355),
}

fig, ax = plt.subplots(figsize=(11, 8))
labels_r = list(rob_toi.keys())
y_pos_r = np.arange(len(labels_r))

ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, zorder=0)
ax.axvspan(0.94, 0.95, alpha=0.08, color="red", zorder=0)

for i, (name, (irr, se, n)) in enumerate(rob_toi.items()):
    ci_l = np.exp(np.log(irr) - 1.96 * se)
    ci_h = np.exp(np.log(irr) + 1.96 * se)
    if "Placebo" in name:
        color = "#16a34a"
    elif "No score" in name:
        color = "#f59e0b"
    elif ci_h < 1.0:
        color = "#dc2626"
    else:
        color = "#2563eb"
    ax.errorbar(irr, y_pos_r[i],
                xerr=[[irr - ci_l], [ci_h - irr]],
                fmt="o", color=color, markersize=7, capsize=3, linewidth=1.5)
    ax.text(ci_h + 0.004, y_pos_r[i], f"{irr:.4f} (n={n:,})",
            va="center", fontsize=9)

ax.set_yticks(y_pos_r)
ax.set_yticklabels(labels_r)
ax.set_xlabel("IRR (with 95% CI)")
ax.set_title("Robustness Checks with TOI Exposure\n(Red shading = baseline cold hand range)")
ax.invert_yaxis()

from matplotlib.patches import Patch
legend_el = [
    Patch(facecolor="#dc2626", label="Significant cold hand (p<0.05)"),
    Patch(facecolor="#16a34a", label="Placebo (should be ~1.0)"),
    Patch(facecolor="#f59e0b", label="Without score-state control"),
    Patch(facecolor="#2563eb", label="Other checks"),
]
ax.legend(handles=legend_el, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "F4b_robustness_toi.png", bbox_inches="tight")
plt.close()
print("  saved")

# F5b: Score-state decomposition — both exposures
print("F5b: Score-state decomposition (both exposures)...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Game-clock without SS
ax = axes[0]
vals = [0.9370, 1.0]
labels_d = ["Without\nScore-State", "Null"]
colors_d = ["#dc2626", "#e5e7eb"]
bars = ax.bar(labels_d, vals, color=colors_d, width=0.5, edgecolor="white")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_ylim(0.88, 1.05)
ax.set_ylabel("IRR")
ax.set_title("Game-Clock, No SS\n(IRR=0.937, p<0.001)")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.4f}",
            ha="center", fontsize=11, fontweight="bold")

# Middle: Game-clock with SS
ax = axes[1]
vals2 = [1.0061, 0.8943, 1.0581]
labels_d2 = ["Scored", "Leading", "Trailing"]
colors_d2 = ["#16a34a", "#2563eb", "#f59e0b"]
bars2 = ax.bar(labels_d2, vals2, color=colors_d2, width=0.5, edgecolor="white")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_ylim(0.88, 1.10)
ax.set_ylabel("IRR")
ax.set_title("Game-Clock + SS\n(Scored: IRR=1.006, p=0.46)")
for bar, v in zip(bars2, vals2):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.4f}",
            ha="center", fontsize=11, fontweight="bold")

# Right: TOI with SS
ax = axes[2]
vals3 = [0.9444, 0.9325, 1.0530]
labels_d3 = ["Scored", "Leading", "Trailing"]
colors_d3 = ["#dc2626", "#2563eb", "#f59e0b"]
bars3 = ax.bar(labels_d3, vals3, color=colors_d3, width=0.5, edgecolor="white")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_ylim(0.88, 1.10)
ax.set_ylabel("IRR")
ax.set_title("TOI + SS\n(Scored: IRR=0.944, p<0.001)")
for bar, v in zip(bars3, vals3):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.4f}",
            ha="center", fontsize=11, fontweight="bold")

fig.suptitle("Score-State Decomposition: The Role of Exposure Definition", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "F5b_decomposition_both.png", bbox_inches="tight")
plt.close()
print("  saved")

# F7: Exposure comparison visualization
print("F7: Exposure comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Distribution of exposure values
ax = axes[0]
scorers = panel[panel["scored_first_goal"] == 1]
non_scorers = panel[panel["scored_first_goal"] == 0]

ax.hist(non_scorers["toi_5v5_post"] / 60, bins=50, alpha=0.5, color="#2563eb",
        label=f"Non-scorers (n={len(non_scorers):,})", density=True, edgecolor="white")
ax.hist(scorers["toi_5v5_post"] / 60, bins=50, alpha=0.5, color="#dc2626",
        label=f"Scorers (n={len(scorers):,})", density=True, edgecolor="white")
ax.set_xlabel("Post-Split Ice Time at 5v5 (minutes)")
ax.set_ylabel("Density")
ax.set_title("Distribution of Actual Ice Time After Split\n(Shift-Based TOI)")
ax.legend()

# Right: Game-clock vs TOI scatter (sample)
ax = axes[1]
sample = panel.sample(n=min(5000, len(panel)), random_state=296)
ax.scatter(sample["time_after_split"] / 60, sample["toi_5v5_post"] / 60,
           alpha=0.15, s=5, color="#2563eb")
ax.plot([0, 35], [0, 35], "k--", alpha=0.3, label="45° line (if equal)")
ax.set_xlabel("Game-Clock Time Remaining (min)")
ax.set_ylabel("Actual 5v5 Ice Time After Split (min)")
ax.set_title("Game-Clock vs Actual Ice Time\n(Each dot = one player-game)")
ax.set_xlim(0, 35)
ax.set_ylim(0, 20)
ax.legend()

fig.suptitle("Why Exposure Definition Matters", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "F7_exposure_comparison.png", bbox_inches="tight")
plt.close()
print("  saved")

# F8: TOI distribution — scorers get more ice time?
print("F8: TOI by scorer status...")

fig, ax = plt.subplots(figsize=(9, 5))

data_plot = []
for label, filt, color in [("Non-Scorers", panel["scored_first_goal"]==0, "#2563eb"),
                             ("Scorers", panel["scored_first_goal"]==1, "#dc2626")]:
    sub = panel[filt]
    toi_pre = sub["toi_5v5_pre"].mean() / 60
    toi_post = sub["toi_5v5_post"].mean() / 60
    data_plot.append((label, toi_pre, toi_post))

x = np.arange(2)
w = 0.3
pre_vals = [d[1] for d in data_plot]
post_vals = [d[2] for d in data_plot]

b1 = ax.bar(x - w/2, pre_vals, w, label="Pre-split TOI", color="#2563eb", alpha=0.8)
b2 = ax.bar(x + w/2, post_vals, w, label="Post-split TOI", color="#dc2626", alpha=0.8)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.1f}", ha="center", fontsize=11)

ax.set_ylabel("Mean 5v5 Ice Time (minutes)")
ax.set_title("Ice Time Before and After Split Point\nby Scorer Status")
ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in data_plot])
ax.legend()
ax.set_ylim(0, max(pre_vals + post_vals) * 1.15)

fig.tight_layout()
fig.savefig(FIGURES / "F8_toi_by_scorer.png", bbox_inches="tight")
plt.close()
print("  saved")

# F9: Side-by-side robustness comparison
print("F9: Side-by-side robustness (game-clock vs TOI)...")

checks = [
    ("Baseline + SS",    1.0061, 0.9444),
    ("Forwards",         1.0081, 0.9431),
    ("Defensemen",       0.9957, 0.9505),
    ("Early goals",      1.0101, 0.9528),
    ("Single-goal",      0.9454, 0.8894),
    ("Consecutive",      1.0068, 0.9469),
    ("Experienced",      1.0238, 0.9712),
    ("Recent (2018+)",   1.0117, 0.9524),
]

fig, ax = plt.subplots(figsize=(12, 6))
y = np.arange(len(checks))

labels_c = [c[0] for c in checks]
gc_vals = [c[1] for c in checks]
toi_vals = [c[2] for c in checks]

ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, zorder=0)

ax.scatter(gc_vals, y - 0.12, color="#2563eb", s=80, zorder=5, label="Game-clock exposure")
ax.scatter(toi_vals, y + 0.12, color="#dc2626", s=80, zorder=5, marker="D", label="TOI exposure")

# Connect pairs
for i in range(len(checks)):
    ax.plot([gc_vals[i], toi_vals[i]], [y[i] - 0.12, y[i] + 0.12],
            color="#94a3b8", linewidth=1, zorder=1)
    # Labels
    ax.text(gc_vals[i], y[i] - 0.3, f"{gc_vals[i]:.4f}", ha="center",
            fontsize=8, color="#2563eb")
    ax.text(toi_vals[i], y[i] + 0.3, f"{toi_vals[i]:.4f}", ha="center",
            fontsize=8, color="#dc2626")

ax.set_yticks(y)
ax.set_yticklabels(labels_c)
ax.set_xlabel("IRR (Incidence Rate Ratio)")
ax.set_title("Robustness Checks: Game-Clock vs TOI Exposure\n(All with score-state controls)")
ax.legend(loc="upper left")
ax.invert_yaxis()

fig.tight_layout()
fig.savefig(FIGURES / "F9_robustness_comparison.png", bbox_inches="tight")
plt.close()
print("  saved")

# Update D12: Summary stats with TOI
print("D12b: Updated summary statistics with TOI...")
TABLES = PROJECT_ROOT / "output" / "tables"

desc_cols = [
    "corsi_total", "corsi_pre_split", "corsi_post_split",
    "sog_total", "goals_5v5", "goals_in_game",
    "first_goal_time", "time_after_split", "time_before_split",
    "toi_5v5_total", "toi_5v5_pre", "toi_5v5_post",
    "xg_per_shot_total", "xg_total",
    "career_games_pre", "career_season_num",
    "scored_first_goal", "scored_prev_team_game",
    "games_missed",
]

panel_desc = panel[desc_cols].describe().T
panel_desc["non_null"] = panel[desc_cols].notna().sum()
panel_desc = panel_desc[["non_null", "mean", "std", "min", "25%", "50%", "75%", "max"]]
panel_desc.columns = ["N", "Mean", "Std", "Min", "P25", "Median", "P75", "Max"]
panel_desc.round(3).to_csv(TABLES / "D12b_summary_statistics_toi.csv")
print("  saved")

print(f"\nAll figures saved to {FIGURES}")
print(f"Total new/updated figures: F2b, F3b, F4b, F5b, F7, F8, F9")
print(f"Updated table: D12b")