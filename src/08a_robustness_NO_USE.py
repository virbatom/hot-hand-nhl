# 08a_robustness_NO_USE.py

"""
Robustness checks for the Hot Hand analysis.

Tests:
  R1. Score-state control (leading/trailing/tied after goal)
  R2. Forwards only
  R3. Defensemen only
  R4. Placebo test (first SOG that isn't a goal as treatment)
  R5. Early goals only (first 40 min, ensures >=20 min observation)
  R6. Position interaction (Forward × scored_first_goal)
  R7. Multi-goal games excluded (pure first-goal effect)
  R8. Consecutive play only (games_missed == 0)

All use NB2 + Season FE (Model 3 spec) as the benchmark.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import time

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT    = PROJECT_ROOT / "output" / "tables"
OUTPUT.mkdir(parents=True, exist_ok=True)

# 1. Load and prepare (same as 06_models.py)
print("Loading panel...")
import polars as pl
panel = pl.read_parquet(PROCESSED / "analysis_panel.parquet").to_pandas()

panel = panel[panel["time_after_split"] > 0].copy()
key_vars = ["corsi_post_split", "scored_first_goal", "time_after_split",
            "career_games_pre", "career_season_num", "Position"]
panel = panel.dropna(subset=key_vars).copy()

panel["log_exposure"] = np.log(panel["time_after_split"])

for col, newcol in [("career_games_pre", "career_games_std"),
                     ("career_season_num", "career_season_std")]:
    mu, sd = panel[col].mean(), panel[col].std()
    panel[newcol] = (panel[col] - mu) / sd if sd > 0 else 0

panel["is_forward"] = (panel["Position"] == "F").astype(int)
panel["scored_prev"] = panel["scored_prev_team_game"].fillna(0).astype(int)
panel["games_missed_capped"] = panel["games_missed"].fillna(0).clip(upper=20).astype(int)

panel["corsi_rate_pre"] = np.where(
    panel["time_before_split"] > 0,
    panel["corsi_pre_split"] / panel["time_before_split"] * 3600,
    0
)
mu, sd = panel["corsi_rate_pre"].mean(), panel["corsi_rate_pre"].std()
panel["corsi_rate_pre_std"] = (panel["corsi_rate_pre"] - mu) / sd if sd > 0 else 0

panel["Season_str"] = panel["Season"].astype(str)
season_dummies = pd.get_dummies(panel["Season_str"], prefix="s", drop_first=True, dtype=float)

print(f"Full sample: {len(panel):,}")

# 2. Helper: run NB2 + season FE on a given sample
def run_nb(data, season_dum, extra_cols=None, label=""):
    """Run NB2 + season FE. Returns (model, col_list) or None on failure."""
    t0 = time.time()

    base = ["scored_first_goal", "is_forward", "career_games_std",
            "career_season_std", "scored_prev", "games_missed_capped",
            "corsi_rate_pre_std"]

    if extra_cols:
        base = base + extra_cols

    X = sm.add_constant(data[base].copy())

    # Align season dummies to data index
    s_dum = season_dum.loc[data.index]
    X = pd.concat([X, s_dum], axis=1)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    y = data["corsi_post_split"].values.astype(float)
    offset = data["log_exposure"].values

    col_list = list(X.columns)

    try:
        model = sm.NegativeBinomial(y, X, offset=offset, loglike_method="nb2").fit(
            disp=0, maxiter=200)

        idx = col_list.index("scored_first_goal")
        irr = np.exp(model.params.iloc[idx])
        se = model.bse.iloc[idx]
        pval = model.pvalues.iloc[idx]
        sig = "***" if pval < 0.001 else "** " if pval < 0.01 else "*  " if pval < 0.05 else "   "

        print(f"  {label}")
        print(f"    N = {len(data):,}")
        print(f"    scored_first_goal:  coef={model.params.iloc[idx]:.4f}  "
              f"SE={se:.4f}  IRR={irr:.4f}  p={pval:.4f} {sig}")
        print(f"    LL={model.llf:,.1f}  AIC={model.aic:,.1f}  "
              f"alpha={model.params.iloc[-1]:.4f}  ({time.time()-t0:.1f}s)")

        # Print extra cols if any
        if extra_cols:
            for c in extra_cols:
                if c in col_list:
                    ci = col_list.index(c)
                    c_irr = np.exp(model.params.iloc[ci])
                    c_pv = model.pvalues.iloc[ci]
                    c_sig = "***" if c_pv < 0.001 else "** " if c_pv < 0.01 else "*  " if c_pv < 0.05 else "   "
                    print(f"    {c}:  coef={model.params.iloc[ci]:.4f}  "
                          f"IRR={c_irr:.4f}  p={c_pv:.4f} {c_sig}")

        return irr, pval, len(data), model
    except Exception as e:
        print(f"  {label}: FAILED ({e})")
        return None

# 3. Run robustness checks
results = {}

print("\n" + "=" * 72)
print("ROBUSTNESS CHECK R0: BASELINE (Model 3 replication)")
print("=" * 72)
r = run_nb(panel, season_dummies, label="Full sample, NB2 + Season FE")
if r: results["R0. Baseline"] = r[:3]

# R1: Score-state control
print("\n" + "=" * 72)
print("R1: SCORE-STATE CONTROL")
print("=" * 72)
print("  Adding score_state_at_split: leading (+), trailing (-), tied (0)\n")

# Use ScoreState from the panel — approximate with goals_in_game context
# ScoreState is at event level; we need it at split time.
# Reload events to get score state at first goal time
events = pl.read_parquet(PROJECT_ROOT / "data" / "interim" / "events.parquet")
schedule = pl.read_parquet(PROJECT_ROOT / "data" / "interim" / "schedule.parquet")

# For scorers: ScoreState at the goal event
# ScoreState format is typically like "1" (home leads by 1), "-2" (away leads by 2), "0" (tied)
# But we need the state AFTER the goal, so +1 from the perspective of the scoring team
goal_score_state = (events
    .filter(
        (pl.col("Goal") == 1) &
        (pl.col("Season") >= 20102011) &
        (pl.col("SeasonState") == "regular") &
        (pl.col("StrengthState").is_in(["5v5", "ENF"]))
    )
    .select(
        pl.col("Player1_ID").alias("PlayerID"),
        "GameID", "GameTime",
        pl.col("ScoreState").alias("score_state_raw"),
    )
    .sort(["PlayerID", "GameID", "GameTime"])
    .group_by(["PlayerID", "GameID"])
    .first()  # first goal per player-game
    .to_pandas()
)

# Parse score state: try to convert to int
goal_score_state["score_state_at_goal"] = pd.to_numeric(
    goal_score_state["score_state_raw"], errors="coerce"
).fillna(0).astype(int)

panel = panel.merge(
    goal_score_state[["PlayerID", "GameID", "score_state_at_goal"]],
    on=["PlayerID", "GameID"],
    how="left"
)
panel["score_state_at_goal"] = panel["score_state_at_goal"].fillna(0).astype(int)

# Create categories: leading (>0), trailing (<0), tied (0)
panel["is_leading"] = (panel["score_state_at_goal"] > 0).astype(int)
panel["is_trailing"] = (panel["score_state_at_goal"] < 0).astype(int)

# Rebuild season dummies on the (possibly reindexed) panel
panel = panel.reset_index(drop=True)
season_dummies = pd.get_dummies(panel["Season_str"], prefix="s", drop_first=True, dtype=float)

# Restandardize after merge
for col, newcol in [("career_games_pre", "career_games_std"),
                     ("career_season_num", "career_season_std")]:
    mu, sd = panel[col].mean(), panel[col].std()
    panel[newcol] = (panel[col] - mu) / sd if sd > 0 else 0
mu, sd = panel["corsi_rate_pre"].mean(), panel["corsi_rate_pre"].std()
panel["corsi_rate_pre_std"] = (panel["corsi_rate_pre"] - mu) / sd if sd > 0 else 0
panel["log_exposure"] = np.log(panel["time_after_split"])

r = run_nb(panel, season_dummies,
           extra_cols=["is_leading", "is_trailing"],
           label="With score-state controls (leading/trailing)")
if r: results["R1. + Score state"] = r[:3]

# R2: Forwards only
print("\n" + "=" * 72)
print("R2: FORWARDS ONLY")
print("=" * 72)
fwd = panel[panel["Position"] == "F"].copy()
s_dum_fwd = season_dummies.loc[fwd.index]
r = run_nb(fwd, s_dum_fwd, label="Forwards only")
if r: results["R2. Forwards only"] = r[:3]

# R3: Defensemen only
print("\n" + "=" * 72)
print("R3: DEFENSEMEN ONLY")
print("=" * 72)
dmen = panel[panel["Position"] == "D"].copy()
s_dum_d = season_dummies.loc[dmen.index]
r = run_nb(dmen, s_dum_d, label="Defensemen only")
if r: results["R3. Defensemen only"] = r[:3]

# R4: Placebo test — first SOG (non-goal) as treatment
print("\n" + "=" * 72)
print("R4: PLACEBO TEST (first SOG non-goal as 'treatment')")
print("=" * 72)
print("  If hot hand is psychological, non-goal SOG should show NO effect.\n")

# Reload events for placebo
ev_modern = (events
    .filter(
        (pl.col("Season") >= 20102011) &
        (pl.col("SeasonState") == "regular") &
        (pl.col("StrengthState") == "5v5") &
        (pl.col("Event") == "shot-on-goal") &
        (pl.col("Goal") == 0) &
        (pl.col("Player1_ID").is_not_null()) &
        (pl.col("Player1_ID") > 0)
    )
    .select(pl.col("Player1_ID").alias("PlayerID"), "GameID", "GameTime")
    .sort(["PlayerID", "GameID", "GameTime"])
    .group_by(["PlayerID", "GameID"])
    .first()
    .rename({"GameTime": "first_sog_time"})
    .to_pandas()
)

panel_placebo = panel.merge(ev_modern[["PlayerID", "GameID", "first_sog_time"]],
                             on=["PlayerID", "GameID"], how="inner")

# Create placebo treatment: did player have a SOG (non-goal) before the split?
panel_placebo["placebo_treatment"] = (
    panel_placebo["first_sog_time"] < panel_placebo["split_time"]
).astype(int)

# Swap treatment variable
panel_placebo["scored_first_goal_orig"] = panel_placebo["scored_first_goal"]
panel_placebo["scored_first_goal"] = panel_placebo["placebo_treatment"]

s_dum_pl = season_dummies.loc[panel_placebo.index]
r = run_nb(panel_placebo, s_dum_pl, label="Placebo: first non-goal SOG as treatment")
if r: results["R4. Placebo (SOG)"] = r[:3]

# Restore
panel_placebo["scored_first_goal"] = panel_placebo["scored_first_goal_orig"]

# R5: Early goals only (first 40 min)
print("\n" + "=" * 72)
print("R5: EARLY GOALS ONLY (scored within first 40 min)")
print("=" * 72)
print("  Ensures >=20 min of post-goal observation.\n")

# Keep all non-scorers + scorers who scored within 2400 sec
early = panel[
    (panel["scored_first_goal"] == 0) |
    ((panel["scored_first_goal"] == 1) & (panel["first_goal_time"] <= 2400))
].copy()
s_dum_e = season_dummies.loc[early.index]
r = run_nb(early, s_dum_e, label="Scorers: goal within first 40 min only")
if r: results["R5. Early goals (<=40m)"] = r[:3]

# R6: Position interaction
print("\n" + "=" * 72)
print("R6: POSITION INTERACTION (Forward × scored_first_goal)")
print("=" * 72)
panel["fwd_x_scored"] = panel["is_forward"] * panel["scored_first_goal"]
r = run_nb(panel, season_dummies,
           extra_cols=["fwd_x_scored"],
           label="With Forward × Scored interaction")
if r: results["R6. Fwd × Scored interaction"] = r[:3]

# R7: Exclude multi-goal games
print("\n" + "=" * 72)
print("R7: EXCLUDE MULTI-GOAL GAMES")
print("=" * 72)
print("  Keep only 0-goal and exactly 1-goal player-games.\n")
single = panel[panel["goals_in_game"] <= 1].copy()
s_dum_s = season_dummies.loc[single.index]
r = run_nb(single, s_dum_s, label="Single-goal games only (0 or 1 goal)")
if r: results["R7. Single-goal only"] = r[:3]

# R8: Consecutive play only
print("\n" + "=" * 72)
print("R8: CONSECUTIVE PLAY ONLY (games_missed == 0)")
print("=" * 72)
consec = panel[panel["games_missed_capped"] == 0].copy()
s_dum_c = season_dummies.loc[consec.index]
r = run_nb(consec, s_dum_c, label="Consecutive play only (no missed games)")
if r: results["R8. Consecutive play"] = r[:3]

# 4. Summary table
print("\n" + "=" * 72)
print("ROBUSTNESS SUMMARY")
print("=" * 72)
print(f"\n  {'Check':<35s} {'N':>10s} {'IRR':>8s} {'p-val':>10s} {'vs Base':>8s}")
print(f"  {'-'*75}")

base_irr = results.get("R0. Baseline", (None,))[0]

for label, (irr, pval, n, *_) in results.items():
    sig = "***" if pval < 0.001 else "** " if pval < 0.01 else "*  " if pval < 0.05 else "   "
    diff = f"{irr - base_irr:+.4f}" if base_irr and "Baseline" not in label else "  ---"
    print(f"  {label:<35s} {n:>10,} {irr:>8.4f}{sig} {pval:>10.4f} {diff:>8s}")

