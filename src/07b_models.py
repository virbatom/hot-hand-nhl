# 07b_models.py

"""
Rerun key models with shift-based TOI as exposure term.
Uses log(toi_5v5_post) instead of log(time_after_split).

Models:
  M1: Pooled Poisson (TOI exposure)
  M2: Pooled NB (TOI exposure)
  M3: NB + Season FE (TOI exposure)
  M3+SS: NB + Season FE + Score State (TOI exposure)  [KEY MODEL]
  M5: Linear FE with TOI-based rate as dep var
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

# 1. Load and prepare
print("Loading panel...")
import polars as pl
panel = pl.read_parquet(PROCESSED / "analysis_panel.parquet").to_pandas()
print(f"Panel shape: {panel.shape}")

# Filter: need toi_5v5_post > 0
panel = panel[panel["toi_5v5_post"] > 0].copy()
key_vars = ["corsi_post_split", "scored_first_goal", "toi_5v5_post",
            "career_games_pre", "career_season_num", "Position"]
panel = panel.dropna(subset=key_vars).copy()

# Exposure: log of actual ice time after split
panel["log_toi_post"] = np.log(panel["toi_5v5_post"])

# Also keep old exposure for comparison
panel["log_time_post"] = np.where(
    panel["time_after_split"] > 0,
    np.log(panel["time_after_split"]),
    np.nan
)

# Controls (same as before)
for col, newcol in [("career_games_pre", "career_games_std"),
                     ("career_season_num", "career_season_std")]:
    mu, sd = panel[col].mean(), panel[col].std()
    panel[newcol] = (panel[col] - mu) / sd if sd > 0 else 0

panel["is_forward"] = (panel["Position"] == "F").astype(int)
panel["scored_prev"] = panel["scored_prev_team_game"].fillna(0).astype(int)
panel["games_missed_capped"] = panel["games_missed"].fillna(0).clip(upper=20).astype(int)

panel["corsi_rate_pre"] = np.where(
    panel["toi_5v5_pre"] > 0,
    panel["corsi_pre_split"] / panel["toi_5v5_pre"] * 3600,
    0
)
mu, sd = panel["corsi_rate_pre"].mean(), panel["corsi_rate_pre"].std()
panel["corsi_rate_pre_std"] = (panel["corsi_rate_pre"] - mu) / sd if sd > 0 else 0

# Score state controls
events = pl.read_parquet(PROJECT_ROOT / "data" / "interim" / "events.parquet")
goal_ss = (events
    .filter(
        (pl.col("Goal") == 1) & (pl.col("Season") >= 20102011) &
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

# Restandardize after merge
for col, newcol in [("career_games_pre", "career_games_std"),
                     ("career_season_num", "career_season_std")]:
    mu, sd = panel[col].mean(), panel[col].std()
    panel[newcol] = (panel[col] - mu) / sd if sd > 0 else 0
mu, sd = panel["corsi_rate_pre"].mean(), panel["corsi_rate_pre"].std()
panel["corsi_rate_pre_std"] = (panel["corsi_rate_pre"] - mu) / sd if sd > 0 else 0

print(f"\nFinal sample:  {len(panel):,}")
print(f"Treatment:     {panel['scored_first_goal'].sum():,} ({panel['scored_first_goal'].mean()*100:.1f}%)")
print(f"Dep var mean:  {panel['corsi_post_split'].mean():.3f}")
print(f"Var/Mean:      {panel['corsi_post_split'].var()/panel['corsi_post_split'].mean():.3f}")
print(f"\nExposure comparison:")
print(f"  log(toi_5v5_post):    mean={panel['log_toi_post'].mean():.2f}  (= {np.exp(panel['log_toi_post'].mean())/60:.1f} min)")
print(f"  log(time_after_split): mean={panel['log_time_post'].dropna().mean():.2f}  (= {np.exp(panel['log_time_post'].dropna().mean())/60:.1f} min)")

# 2. Helper
def print_results(model, X_cols, show_cols=None):
    n_params = len(X_cols)
    if show_cols is None:
        show_cols = X_cols
    print(f"\n  {'Variable':<28s} {'Coef':>8s} {'SE':>8s} {'IRR':>8s} {'p-val':>8s}")
    print(f"  {'-'*62}")
    for col in show_cols:
        if col not in X_cols:
            continue
        i = X_cols.index(col)
        coef = model.params.iloc[i]
        se = model.bse.iloc[i]
        irr = np.exp(coef)
        pv = model.pvalues.iloc[i]
        sig = "***" if pv < 0.001 else "** " if pv < 0.01 else "*  " if pv < 0.05 else "   "
        print(f"  {col:<28s} {coef:>8.4f} {se:>8.4f} {irr:>8.4f} {pv:>8.4f} {sig}")
    if len(model.params) > n_params:
        print(f"\n  Alpha (dispersion): {model.params.iloc[-1]:.4f}")
    print(f"\n  LL={model.llf:,.1f}  AIC={model.aic:,.1f}  BIC={model.bic:,.1f}  N={int(model.nobs):,}")

# 3. X matrices
y = panel["corsi_post_split"].values.astype(float)
exposure_toi = panel["log_toi_post"].values

base_vars = ["scored_first_goal", "is_forward", "career_games_std",
             "career_season_std", "scored_prev", "games_missed_capped",
             "corsi_rate_pre_std"]
SHOW = ["const"] + base_vars

X_base = sm.add_constant(panel[base_vars].copy())
base_cols = list(X_base.columns)

X_season = pd.concat([X_base, season_dummies], axis=1)
season_cols = list(X_season.columns)

ss_vars = base_vars + ["is_leading", "is_trailing"]
X_ss = sm.add_constant(panel[ss_vars].copy())
X_ss = pd.concat([X_ss, season_dummies], axis=1)
ss_cols = list(X_ss.columns)

results = {}

# M1: Pooled Poisson (TOI exposure)
print("\n" + "=" * 72)
print("M1: POOLED POISSON (TOI exposure)")
print("=" * 72)
t0 = time.time()
m1 = sm.Poisson(y, X_base, offset=exposure_toi).fit(disp=0, maxiter=100)
print(f"  Converged in {time.time()-t0:.1f}s")
print_results(m1, base_cols, SHOW)
results["M1. Poisson (TOI)"] = (m1, base_cols)

# M2: Pooled NB (TOI exposure)
print("\n" + "=" * 72)
print("M2: POOLED NB (TOI exposure)")
print("=" * 72)
t0 = time.time()
m2 = sm.NegativeBinomial(y, X_base, offset=exposure_toi, loglike_method="nb2").fit(disp=0, maxiter=200)
print(f"  Converged in {time.time()-t0:.1f}s")
print_results(m2, base_cols, SHOW)
results["M2. NB (TOI)"] = (m2, base_cols)

# LR test
lr_stat = 2 * (m2.llf - m1.llf)
lr_pval = stats.chi2.sf(lr_stat, 1)
print(f"\n  LR test Poisson vs NB: chi2={lr_stat:,.1f}, p={lr_pval:.2e}")

# M3: NB + Season FE (TOI exposure)
print("\n" + "=" * 72)
print("M3: NB + SEASON FE (TOI exposure)")
print("=" * 72)
t0 = time.time()
m3 = sm.NegativeBinomial(y, X_season, offset=exposure_toi, loglike_method="nb2").fit(disp=0, maxiter=200)
print(f"  Converged in {time.time()-t0:.1f}s")
print_results(m3, season_cols, SHOW)
results["M3. NB + Season FE (TOI)"] = (m3, season_cols)

# M3+SS: NB + Season FE + Score State (TOI exposure) — KEY MODEL
print("\n" + "=" * 72)
print("M3+SS: NB + SEASON FE + SCORE STATE (TOI exposure) *** KEY MODEL ***")
print("=" * 72)
t0 = time.time()
m3ss = sm.NegativeBinomial(y, X_ss, offset=exposure_toi, loglike_method="nb2").fit(disp=0, maxiter=200)
print(f"  Converged in {time.time()-t0:.1f}s")
print_results(m3ss, ss_cols, SHOW + ["is_leading", "is_trailing"])
results["M3+SS. NB + SS (TOI)"] = (m3ss, ss_cols)

# M5: Linear FE with TOI-based rate
print("\n" + "=" * 72)
print("M5: LINEAR OLS + PLAYER FE (TOI-based rate, within-transform)")
print("=" * 72)
print("  Dep var: corsi_rate_post_toi60 (shots per 60 min actual ice time)")

panel_fe = panel.copy()
panel_fe["y_fe"] = panel_fe["corsi_rate_post_toi60"]
panel_fe = panel_fe.dropna(subset=["y_fe"])
# Remove infinite values
panel_fe = panel_fe[np.isfinite(panel_fe["y_fe"])]

fe_vars = ["y_fe", "scored_first_goal", "career_games_std", "career_season_std",
           "scored_prev", "games_missed_capped", "corsi_rate_pre_std",
           "is_leading", "is_trailing"]
for v in fe_vars:
    panel_fe[f"{v}_dm"] = panel_fe[v] - panel_fe.groupby("PlayerID")[v].transform("mean")

X_fe = panel_fe[[f"{v}_dm" for v in fe_vars[1:]]].copy()
s_dum_fe = pd.get_dummies(panel_fe["Season_str"], prefix="s", drop_first=True, dtype=float)
X_fe = pd.concat([X_fe.reset_index(drop=True), s_dum_fe.reset_index(drop=True)], axis=1)
y_fe = panel_fe["y_fe_dm"].values

t0 = time.time()
m5 = sm.OLS(y_fe, X_fe).fit(cov_type="cluster", cov_kwds={"groups": panel_fe["PlayerID"].values})
print(f"  Converged in {time.time()-t0:.1f}s")
print(f"  R-squared (within): {m5.rsquared:.4f}")
print(f"  N: {int(m5.nobs):,}, Clusters: {panel_fe['PlayerID'].nunique():,}")

fe_show = [f"{v}_dm" for v in fe_vars[1:]]
fe_cols = list(X_fe.columns)
print(f"\n  {'Variable':<32s} {'Coef':>8s} {'SE':>8s} {'p-val':>8s}")
print(f"  {'-'*56}")
for col in fe_show:
    i = fe_cols.index(col)
    coef = m5.params.iloc[i]
    se = m5.bse.iloc[i]
    pv = m5.pvalues.iloc[i]
    sig = "***" if pv < 0.001 else "** " if pv < 0.01 else "*  " if pv < 0.05 else "   "
    print(f"  {col:<32s} {coef:>8.4f} {se:>8.4f} {pv:>8.4f} {sig}")

results["M5. Linear FE (TOI rate)"] = (m5, fe_cols)

# SUMMARY
print("\n" + "=" * 72)
print("MODEL COMPARISON: GAME-CLOCK EXPOSURE vs TOI EXPOSURE")
print("=" * 72)

# Previous results (game-clock, from script 06/08) for comparison
print("\n  PREVIOUS (game-clock exposure):")
print(f"    M3 NB + Season FE:         IRR = 0.9370  p < 0.001  (apparent cold hand)")
print(f"    M3 + Score State:          IRR = 1.0061  p = 0.463  (null, no hot hand)")

print(f"\n  NEW (TOI exposure):")
print(f"  {'Model':<35s} {'IRR/Coef':>10s} {'p-val':>10s}")
print(f"  {'-'*60}")

for label, (model, cols) in results.items():
    if "Linear" in label:
        idx = cols.index("scored_first_goal_dm")
        coef = model.params.iloc[idx]
        pval = model.pvalues.iloc[idx]
        sig = "***" if pval < 0.001 else "** " if pval < 0.01 else "*  " if pval < 0.05 else "   "
        print(f"  {label:<35s} {coef:>9.4f}{sig} {pval:>10.4f}")
    else:
        idx = cols.index("scored_first_goal")
        irr = np.exp(model.params.iloc[idx])
        pval = model.pvalues.iloc[idx]
        sig = "***" if pval < 0.001 else "** " if pval < 0.01 else "*  " if pval < 0.05 else "   "
        print(f"  {label:<35s} {irr:>9.4f}{sig} {pval:>10.4f}")
