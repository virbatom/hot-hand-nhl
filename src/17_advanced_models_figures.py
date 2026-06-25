# 17_advanced_models_figures

"""
17_advanced_models_figures.py

Figures and tables for the models added after the original 09/09b figures:
  F10  Zero-Truncated Poisson — four-model specification (intensive margin)
  F11  Hurdle decomposition (THE payoff: cold hand = extensive margin)
  F12  NB vs Poisson — distributional robustness (point estimates ~identical)
  F13  Score-state TIME shares by scorer status (team & on-ice)

  T1   Master specification table  (Poisson / NB / ZTP x exposure x score state)
  T2   Hurdle results table        (logit OR, ZTP IRR, net E[Y] ratio)
  T3   Score-state time-share table

Model numbers are taken from the finalised script outputs (07a/07b, 08*, 13, 16).
F13 and T3 are computed live from analysis_panel_ss.parquet.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES   = PROJECT_ROOT / "output" / "figures"
TABLES    = PROJECT_ROOT / "output" / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12, "figure.dpi": 150,
})

# F10: Zero-Truncated Poisson — four-model specification
print("F10: ZTP four-model forest...")
# (label, IRR, SE, colour)
ztp = [
    ("M1. Game-clock, no SS",  1.0283, 0.0047, "#94a3b8"),
    ("M2. TOI, no SS",         1.0066, 0.0047, "#60a5fa"),
    ("M3. TOI + SS",           1.0554, 0.0099, "#dc2626"),
    ("M4. Game-clock + SS",    1.1066, 0.0099, "#16a34a"),
]
fig, ax = plt.subplots(figsize=(10, 5))
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, zorder=0)
y = len(ztp) - 1
ticks, ticklabels = [], []
for label, irr, se, color in ztp:
    lo = np.exp(np.log(irr) - 1.96 * se)
    hi = np.exp(np.log(irr) + 1.96 * se)
    ax.errorbar(irr, y, xerr=[[irr - lo], [hi - irr]],
                fmt="o", color=color, markersize=9, capsize=4, linewidth=2)
    ax.text(hi + 0.002, y, f"{irr:.4f}", va="center", fontsize=10, color=color)
    ticks.append(y); ticklabels.append(label); y -= 1
ax.set_yticks(ticks); ax.set_yticklabels(ticklabels)
ax.set_xlabel("Incidence Rate Ratio (IRR)")
ax.set_title("Zero-Truncated Poisson: Effect of Scoring on Shot Count\n"
             "(intensive margin, given \u22651 post-split shot; n=407,027)")
fig.tight_layout()
fig.savefig(FIGURES / "F10_ztp_four_models.png", bbox_inches="tight")
plt.close()
print("  saved")

# F11: Hurdle decomposition  (THE payoff figure)
print("F11: Hurdle decomposition...")
# margins for the two clean score-state variants
groups = ["Extensive margin\n(P shoot again)\nLogit OR",
          "Intensive margin\n(shots | \u22651)\nZTP IRR",
          "Net effect\nE[shots] ratio"]
binary = [0.7012, 1.0554, 0.9407]
onice  = [0.7183, 1.0604, 0.9505]

x = np.arange(len(groups)); w = 0.36
fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, zorder=1,
           label="No effect (=1.0)")
b1 = ax.bar(x - w/2, binary, w, color="#2563eb", alpha=0.9,
            label="Binary score state (primary)", edgecolor="white")
b2 = ax.bar(x + w/2, onice, w, color="#f59e0b", alpha=0.9,
            label="On-ice time shares", edgecolor="white")
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.4f}",
                ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("Effect of scoring (OR or IRR)")
ax.set_ylim(0.6, 1.15)
ax.set_title("Hurdle Decomposition: the Cold Hand Lives in the Extensive Margin\n"
             "scoring \u2192 ~30% lower odds of shooting again; no fall in rate once shooting")
ax.legend(loc="lower right", fontsize=9)
# shaded annotation
ax.text(0.0, 0.64, "27% of at-risk\nplayer-games take\nZERO post-split shots",
        ha="center", fontsize=8.5, color="#374151",
        bbox=dict(boxstyle="round", fc="#f3f4f6", ec="#d1d5db"))
fig.tight_layout()
fig.savefig(FIGURES / "F11_hurdle_decomposition.png", bbox_inches="tight")
plt.close()
print("  saved")

# F12: NB vs Poisson — distributional robustness
print("F12: NB vs Poisson...")
# (label, NB IRR, Poisson IRR, is_placebo)  -- TOI robustness checks
nbp = [
    ("Baseline + SS", 0.9444, 0.9449, False),
    ("Forwards",      0.9431, 0.9434, False),
    ("Defensemen",    0.9505, 0.9521, False),
    ("Early goals",   0.9528, 0.9531, False),
    ("Fwd x Scored",  0.9392, 0.9399, False),
    ("Single-goal",   0.8894, 0.8897, False),
    ("Consecutive",   0.9469, 0.9474, False),
    ("Experienced",   0.9712, 0.9717, False),
    ("Recent (2018+)",0.9524, 0.9531, False),
    ("No score state",0.9050, 0.9054, False),
    ("Placebo s=296", 1.0048, 1.0048, True),
    ("Placebo s=185", 1.0007, 1.0006, True),
    ("Placebo s=142", 0.9964, 0.9964, True),
]
fig, ax = plt.subplots(figsize=(7.5, 7.5))
lo, hi = 0.88, 1.02
ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="45\u00b0 line (identical)")
for label, nb, po, plac in nbp:
    c = "#16a34a" if plac else "#dc2626"
    ax.scatter(nb, po, s=70, color=c, zorder=5, edgecolor="white")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Negative Binomial IRR")
ax.set_ylabel("Poisson IRR")
ax.set_title("Negative Binomial vs Poisson (TOI robustness checks)\n"
             "Point estimates are virtually identical")
leg = [Patch(facecolor="#dc2626", label="Cold-hand checks"),
       Patch(facecolor="#16a34a", label="Placebo checks")]
ax.legend(handles=leg + [plt.Line2D([0],[0], ls="--", color="k", alpha=0.4,
          label="45\u00b0 line")], loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "F12_nb_vs_poisson.png", bbox_inches="tight")
plt.close()
print("  saved")

# F13: Score-state TIME shares by scorer status (computed live)
print("F13: Score-state time shares...")
panel = pl.read_parquet(PROCESSED / "analysis_panel_ss.parquet").to_pandas()

def shares(df, prefix):
    return [df[f"{prefix}_leading"].mean(),
            df[f"{prefix}_tied"].mean(),
            df[f"{prefix}_trailing"].mean()]

sc  = panel[panel["scored_first_goal"] == 1]
nsc = panel[panel["scored_first_goal"] == 0]
team_sc,  team_nsc  = shares(sc, "pct_time"),  shares(nsc, "pct_time")
onice_sc, onice_nsc = shares(sc, "pct_onice"), shares(nsc, "pct_onice")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
states = ["Leading", "Tied", "Trailing"]
colors = ["#16a34a", "#9ca3af", "#dc2626"]
for ax, (title, sc_v, nsc_v) in zip(
        axes, [("Team time (game-clock)", team_sc, team_nsc),
               ("Player on-ice time", onice_sc, onice_nsc)]):
    x = np.arange(2); bottom = np.zeros(2)
    for i, st in enumerate(states):
        vals = [nsc_v[i], sc_v[i]]
        ax.bar(x, vals, 0.55, bottom=bottom, color=colors[i], label=st,
               edgecolor="white")
        for j in range(2):
            if vals[j] > 0.04:
                ax.text(x[j], bottom[j] + vals[j]/2, f"{vals[j]*100:.0f}%",
                        ha="center", va="center", fontsize=10, color="white",
                        fontweight="bold")
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(["Non-scorers", "Scorers"])
    ax.set_ylim(0, 1); ax.set_ylabel("Share of post-split time")
    ax.set_title(title)
    ax.grid(axis="x")
axes[1].legend(loc="upper right", fontsize=9, framealpha=0.9)
fig.suptitle("Post-Split Score-State Exposure: Scorers Spend More Time Leading",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "F13_time_share_scorestate.png", bbox_inches="tight")
plt.close()
print("  saved")

# T1: Master specification table
print("T1: Master specification table...")
master = pd.DataFrame([
    ("Game-clock", "Poisson", "No",  0.9379, "<.001", 561051),
    ("Game-clock", "Neg. Binomial", "No",  0.9370, "<.001", 561051),
    ("Game-clock", "Neg. Binomial", "Yes", 1.0061, ".463", 561051),
    ("Game-clock", "Poisson", "Yes", 1.0061, ".439", 561051),
    ("TOI",        "Poisson", "No",  0.9054, "<.001", 557355),
    ("TOI",        "Neg. Binomial", "No",  0.9050, "<.001", 557355),
    ("TOI",        "Neg. Binomial", "Yes", 0.9444, "<.001", 557355),
    ("TOI",        "Poisson", "Yes", 0.9449, "<.001", 557355),
    ("Game-clock", "Zero-Trunc. Poisson", "No",  1.0283, "<.001", 407027),
    ("Game-clock", "Zero-Trunc. Poisson", "Yes", 1.1066, "<.001", 407027),
    ("TOI",        "Zero-Trunc. Poisson", "No",  1.0066, ".163", 407027),
    ("TOI",        "Zero-Trunc. Poisson", "Yes", 1.0554, "<.001", 407027),
], columns=["Exposure", "Distribution", "Score state", "IRR_scored", "p_value", "N"])
master.to_csv(TABLES / "T1_master_specification.csv", index=False)
print("  saved")

# T2: Hurdle results table
print("T2: Hurdle results table...")
hurdle = pd.DataFrame([
    ("A) Binary score state (primary)", 0.7012, "<.001", 1.0554, "<.001", 0.9407),
    ("B) On-ice time shares",           0.7183, "<.001", 1.0604, "<.001", 0.9505),
], columns=["Variant", "Logit_OR_scored", "Logit_p",
            "ZTP_IRR_scored", "ZTP_p", "Net_EY_ratio"])
hurdle.to_csv(TABLES / "T2_hurdle_results.csv", index=False)
print("  saved")

# T3: Score-state time-share table (computed live)
print("T3: Score-state time-share table...")
t3 = pd.DataFrame({
    "Group":       ["Scorers", "Scorers", "Non-scorers", "Non-scorers"],
    "Time base":   ["Team (game-clock)", "On-ice", "Team (game-clock)", "On-ice"],
    "Leading":     [team_sc[0], onice_sc[0], team_nsc[0], onice_nsc[0]],
    "Tied":        [team_sc[1], onice_sc[1], team_nsc[1], onice_nsc[1]],
    "Trailing":    [team_sc[2], onice_sc[2], team_nsc[2], onice_nsc[2]],
}).round(3)
t3.to_csv(TABLES / "T3_time_share_scorestate.csv", index=False)
print("  saved")

print(f"\nFigures -> {FIGURES}")
print("  F10_ztp_four_models, F11_hurdle_decomposition, F12_nb_vs_poisson, F13_time_share_scorestate")
print(f"Tables  -> {TABLES}")
print("  T1_master_specification, T2_hurdle_results, T3_time_share_scorestate")