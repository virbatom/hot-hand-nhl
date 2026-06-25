# 08c_robustness_v2_robustness_toi_poisson.py

"""
Robustness checks with TOI-based exposure - POISSON ONLY.
All models: Poisson + Season FE + Score State, offset = log(toi_5v5_post).
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import time

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

print("Loading panel...")
import polars as pl
panel = pl.read_parquet(PROCESSED / "analysis_panel.parquet").to_pandas()

# --- Prep ---
panel = panel[panel["toi_5v5_post"] > 0].copy()
key_vars = ["corsi_post_split", "scored_first_goal", "toi_5v5_post",
            "career_games_pre", "career_season_num", "Position"]
panel = panel.dropna(subset=key_vars).copy()

panel["log_toi_post"] = np.log(panel["toi_5v5_post"])
for col, newcol in [("career_games_pre", "career_games_std"),
                     ("career_season_num", "career_season_std")]:
    mu, sd = panel[col].mean(), panel[col].std()
    panel[newcol] = (panel[col] - mu) / sd if sd > 0 else 0
panel["is_forward"] = (panel["Position"] == "F").astype(int)
panel["scored_prev"] = panel["scored_prev_team_game"].fillna(0).astype(int)
panel["games_missed_capped"] = panel["games_missed"].fillna(0).clip(upper=20).astype(int)
panel["corsi_rate_pre"] = np.where(
    panel["toi_5v5_pre"] > 0,
    panel["corsi_pre_split"] / panel["toi_5v5_pre"] * 3600, 0)
mu, sd = panel["corsi_rate_pre"].mean(), panel["corsi_rate_pre"].std()
panel["corsi_rate_pre_std"] = (panel["corsi_rate_pre"] - mu) / sd if sd > 0 else 0

# Score state
events = pl.read_parquet(PROJECT_ROOT / "data" / "interim" / "events.parquet")
goal_ss = (events
    .filter((pl.col("Goal") == 1) & (pl.col("Season") >= 20102011) &
            (pl.col("SeasonState") == "regular") &
            (pl.col("StrengthState").is_in(["5v5", "ENF"])))
    .select(pl.col("Player1_ID").alias("PlayerID"), "GameID", "GameTime",
            pl.col("ScoreState").alias("ss_raw"))
    .sort(["PlayerID", "GameID", "GameTime"])
    .group_by(["PlayerID", "GameID"]).first()
    .to_pandas())
goal_ss["score_state_at_goal"] = pd.to_numeric(goal_ss["ss_raw"], errors="coerce").fillna(0).astype(int)
panel = panel.merge(goal_ss[["PlayerID", "GameID", "score_state_at_goal"]],
                     on=["PlayerID", "GameID"], how="left")
panel["score_state_at_goal"] = panel["score_state_at_goal"].fillna(0).astype(int)
panel["is_leading"] = (panel["score_state_at_goal"] > 0).astype(int)
panel["is_trailing"] = (panel["score_state_at_goal"] < 0).astype(int)

panel = panel.reset_index(drop=True)
panel["Season_str"] = panel["Season"].astype(str)
season_dummies = pd.get_dummies(panel["Season_str"], prefix="s", drop_first=True, dtype=float)

# Restandardize
for col, newcol in [("career_games_pre", "career_games_std"),
                     ("career_season_num", "career_season_std")]:
    mu, sd = panel[col].mean(), panel[col].std()
    panel[newcol] = (panel[col] - mu) / sd if sd > 0 else 0
mu, sd = panel["corsi_rate_pre"].mean(), panel["corsi_rate_pre"].std()
panel["corsi_rate_pre_std"] = (panel["corsi_rate_pre"] - mu) / sd if sd > 0 else 0

print(f"Full sample: {len(panel):,}\n")

# Helper
BASE_COLS = ["scored_first_goal", "is_forward", "career_games_std",
             "career_season_std", "scored_prev", "games_missed_capped",
             "corsi_rate_pre_std", "is_leading", "is_trailing"]

def run_poisson(data, s_dum, cols=None, label="", drop_cols=None):
    t0 = time.time()
    use_cols = list(cols or BASE_COLS)
    if drop_cols:
        use_cols = [c for c in use_cols if c not in drop_cols]

    X = sm.add_constant(data[use_cols].copy())
    sd = s_dum.loc[data.index]
    X = pd.concat([X, sd], axis=1).fillna(0).replace([np.inf, -np.inf], 0)

    y = data["corsi_post_split"].values.astype(float)
    offset = data["log_toi_post"].values
    col_list = list(X.columns)

    try:
        model = sm.Poisson(y, X, offset=offset).fit(disp=0, maxiter=200)
        idx = col_list.index("scored_first_goal")
        irr = np.exp(model.params.iloc[idx])
        pval = model.pvalues.iloc[idx]
        sig = "***" if pval < 0.001 else "** " if pval < 0.01 else "* " if pval < 0.05 else "   "
        print(f"  {label}")
        print(f"    N={len(data):,}  IRR={irr:.4f}  p={pval:.4f} {sig}  ({time.time()-t0:.1f}s)")
        return irr, pval, len(data)
    except Exception as e:
        print(f"  {label}: FAILED ({e})")
        return None

# Run all checks
results = {}

print("=" * 72)
print("ALL MODELS: POISSON + SEASON FE + SCORE STATE, TOI EXPOSURE")
print("=" * 72)

# R0: Baseline
print("\nR0: BASELINE + SCORE STATE (TOI)")
r = run_poisson(panel, season_dummies, label="Full sample")
if r: results["R0. Baseline + SS (TOI)"] = r

# R2: Forwards only
print("\nR2: FORWARDS ONLY")
fwd = panel[panel["Position"] == "F"].copy()
r = run_poisson(fwd, season_dummies, label="Forwards only")
if r: results["R2. Forwards only"] = r

# R3: Defensemen only
print("\nR3: DEFENSEMEN ONLY")
dmen = panel[panel["Position"] == "D"].copy()
r = run_poisson(dmen, season_dummies, drop_cols=["is_forward"], label="Defensemen only")
if r: results["R3. Defensemen only"] = r

# R4: Placebo (random treatment)
print("\nR4: PLACEBO (random treatment)")
np.random.seed(296)
panel_pl = panel.copy()
panel_pl["scored_first_goal"] = (
    panel_pl.groupby("Season")["scored_first_goal"]
    .transform(lambda x: np.random.permutation(x.values)))
r = run_poisson(panel_pl, season_dummies, label="Random treatment (seed=296)")
if r: results["R4a. Placebo (s=296)"] = r

np.random.seed(185)
panel_pl["scored_first_goal"] = (
    panel.groupby("Season")["scored_first_goal"]
    .transform(lambda x: np.random.permutation(x.values)))
r = run_poisson(panel_pl, season_dummies, label="Random treatment (seed=185)")
if r: results["R4b. Placebo (s=185)"] = r

np.random.seed(142)
panel_pl["scored_first_goal"] = (
    panel.groupby("Season")["scored_first_goal"]
    .transform(lambda x: np.random.permutation(x.values)))
r = run_poisson(panel_pl, season_dummies, label="Random treatment (seed=142)")
if r: results["R4c. Placebo (s=142)"] = r

# R5: Early goals (<=40 min)
print("\nR5: EARLY GOALS ONLY (<=40 min)")
early = panel[
    (panel["scored_first_goal"] == 0) |
    ((panel["scored_first_goal"] == 1) & (panel["first_goal_time"] <= 2400))
].copy()
r = run_poisson(early, season_dummies, label="Goals within first 40 min")
if r: results["R5. Early goals (<=40m)"] = r

# R6: Position interaction
print("\nR6: POSITION INTERACTION")
panel["fwd_x_scored"] = panel["is_forward"] * panel["scored_first_goal"]
r = run_poisson(panel, season_dummies, cols=BASE_COLS + ["fwd_x_scored"],
           label="Forward x Scored interaction")
if r: results["R6. Fwd x Scored"] = r

# R7: Single-goal games
print("\nR7: SINGLE-GOAL GAMES ONLY")
single = panel[panel["goals_in_game"] <= 1].copy()
r = run_poisson(single, season_dummies, label="0 or 1 goal per game")
if r: results["R7. Single-goal only"] = r

# R8: Consecutive play
print("\nR8: CONSECUTIVE PLAY ONLY")
consec = panel[panel["games_missed_capped"] == 0].copy()
r = run_poisson(consec, season_dummies, drop_cols=["games_missed_capped"],
           label="No missed games")
if r: results["R8. Consecutive play"] = r

# R9: Experienced players (200+)
print("\nR9: EXPERIENCED PLAYERS (200+ career games)")
exp = panel[panel["career_games_pre"] >= 200].copy()
r = run_poisson(exp, season_dummies, label="career_games_pre >= 200")
if r: results["R9. Experienced (200+)"] = r

# R10: Recent seasons (2018+)
print("\nR10: RECENT SEASONS (2018-2025)")
recent = panel[panel["Season"] >= 20182019].copy()
s_dum_r = pd.get_dummies(recent["Season_str"], prefix="s", drop_first=True, dtype=float)
r = run_poisson(recent, s_dum_r, label="Seasons 2018-19 through 2024-25")
if r: results["R10. Recent (2018+)"] = r

# R11 NEW: Without score-state controls (to show the decomposition)
print("\nR11: WITHOUT SCORE STATE (TOI) — for comparison")
r = run_poisson(panel, season_dummies,
           cols=[c for c in BASE_COLS if c not in ["is_leading", "is_trailing"]],
           label="No score-state controls")
if r: results["R11. No score state (TOI)"] = r

# Summary
print("\n" + "=" * 72)
print("ROBUSTNESS SUMMARY (TOI exposure, all with score-state controls) [POISSON]")
print("=" * 72)
print(f"\n  {'Check':<35s} {'N':>10s} {'IRR':>8s} {'p-val':>10s} {'Sig':>5s}")
print(f"  {'-'*72}")

for label, (irr, pval, n) in results.items():
    sig = "***" if pval < 0.001 else "** " if pval < 0.01 else "* " if pval < 0.05 else "   "
    print(f"  {label:<35s} {n:>10,} {irr:>8.4f} {pval:>10.4f} {sig}")