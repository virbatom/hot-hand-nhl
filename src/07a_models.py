# 07a_models.py

"""
Hot-Hand Regression Models (v2 — robust printing, fixed NaN handling)
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

# 1. Load and prepare data
print("Loading panel...")
import polars as pl
panel = pl.read_parquet(PROCESSED / "analysis_panel.parquet").to_pandas()
print(f"Panel shape: {panel.shape}")

# 2. Prepare regression variables
print("\nPreparing variables...")

panel = panel[panel["time_after_split"] > 0].copy()
key_vars = ["corsi_post_split", "scored_first_goal", "time_after_split",
            "career_games_pre", "career_season_num", "Position"]
panel = panel.dropna(subset=key_vars).copy()

# Exposure
panel["log_exposure"] = np.log(panel["time_after_split"])

# Standardize continuous controls
for col, newcol in [("career_games_pre", "career_games_std"),
                     ("career_season_num", "career_season_std")]:
    mu, sd = panel[col].mean(), panel[col].std()
    panel[newcol] = (panel[col] - mu) / sd if sd > 0 else 0

# Position dummy
panel["is_forward"] = (panel["Position"] == "F").astype(int)

# Previous game controls
panel["scored_prev"] = panel["scored_prev_team_game"].fillna(0).astype(int)
panel["games_missed_capped"] = panel["games_missed"].fillna(0).clip(upper=20).astype(int)

# Pre-split shooting rate
panel["corsi_rate_pre"] = np.where(
    panel["time_before_split"] > 0,
    panel["corsi_pre_split"] / panel["time_before_split"] * 3600,
    0
)
mu, sd = panel["corsi_rate_pre"].mean(), panel["corsi_rate_pre"].std()
panel["corsi_rate_pre_std"] = (panel["corsi_rate_pre"] - mu) / sd if sd > 0 else 0

# Score state controls (Merged from 07b_models)
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

# Season dummies
panel["Season_str"] = panel["Season"].astype(str)
season_dummies = pd.get_dummies(panel["Season_str"], prefix="s", drop_first=True, dtype=float)

print(f"  Final sample:  {len(panel):,}")
print(f"  Treatment:     {panel['scored_first_goal'].sum():,} ({panel['scored_first_goal'].mean()*100:.1f}%)")
vmr = panel["corsi_post_split"].var() / panel["corsi_post_split"].mean()
print(f"  Dep var mean:  {panel['corsi_post_split'].mean():.3f}")
print(f"  Var/Mean:      {vmr:.3f} ({'overdispersed' if vmr > 1 else 'underdispersed'})")

# 3. Helper: print model results
def print_results(model, X_cols, label, show_cols=None):
    """Print key coefficients from a fitted model."""
    n_params = len(X_cols)
    params = model.params[:n_params]
    bse = model.bse[:n_params]
    pvals = model.pvalues[:n_params]

    if show_cols is None:
        show_cols = X_cols

    print(f"\n  {'Variable':<28s} {'Coef':>8s} {'SE':>8s} {'IRR':>8s} {'p-val':>8s}")
    print(f"  {'-'*62}")
    for col in show_cols:
        if col not in X_cols:
            continue
        i = X_cols.index(col)
        coef = params.iloc[i]
        se = bse.iloc[i]
        irr = np.exp(coef)
        pv = pvals.iloc[i]
        sig = "***" if pv < 0.001 else "** " if pv < 0.01 else "* " if pv < 0.05 else "   "
        print(f"  {col:<28s} {coef:>8.4f} {se:>8.4f} {irr:>8.4f} {pv:>8.4f} {sig}")

    # NB alpha (dispersion) is the last parameter
    if hasattr(model, 'params') and len(model.params) > n_params:
        alpha = model.params.iloc[-1]
        print(f"\n  Alpha (dispersion): {alpha:.4f}")

    print(f"\n  Log-Likelihood: {model.llf:>12,.1f}")
    print(f"  AIC:            {model.aic:>12,.1f}")
    print(f"  BIC:            {model.bic:>12,.1f}")
    print(f"  N:              {int(model.nobs):>12,}")

# 4. Define X matrices
y = panel["corsi_post_split"].values.astype(float)
exposure = panel["log_exposure"].values

base_cols = ["const", "scored_first_goal", "is_forward", "career_games_std",
             "career_season_std", "scored_prev", "games_missed_capped",
             "corsi_rate_pre_std"]

X_base = panel[base_cols[1:]].copy()  # without const
X_base = sm.add_constant(X_base)
base_col_list = list(X_base.columns)

X_season = pd.concat([X_base, season_dummies], axis=1)
season_col_list = list(X_season.columns)

# Matrix for Model 4 (NB + Season FE + Score State)
ss_vars = base_cols[1:] + ["is_leading", "is_trailing"]
X_ss = sm.add_constant(panel[ss_vars].copy())
X_ss = pd.concat([X_ss, season_dummies], axis=1)
ss_col_list = list(X_ss.columns)

SHOW_COLS = base_cols  # only show these in output (skip season dummies)

results = {}

# MODEL 1: Pooled Poisson
print("\n" + "=" * 72)
print("MODEL 1: POOLED POISSON")
print("=" * 72)
t0 = time.time()
m1 = sm.Poisson(y, X_base, offset=exposure).fit(disp=0, maxiter=100)
print(f"  Converged in {time.time()-t0:.1f}s")
print_results(m1, base_col_list, "Poisson", SHOW_COLS)
results["1. Pooled Poisson"] = (m1, base_col_list)

# MODEL 2: Pooled Negative Binomial
print("\n" + "=" * 72)
print("MODEL 2: POOLED NEGATIVE BINOMIAL (NB2)")
print("=" * 72)
t0 = time.time()
m2 = sm.NegativeBinomial(y, X_base, offset=exposure, loglike_method="nb2").fit(disp=0, maxiter=200)
print(f"  Converged in {time.time()-t0:.1f}s")
print_results(m2, base_col_list, "NB2", SHOW_COLS)
results["2. Pooled NB"] = (m2, base_col_list)

# LR test: Poisson vs NB
lr_stat = 2 * (m2.llf - m1.llf)
lr_pval = stats.chi2.sf(lr_stat, 1)
print(f"\n  LR test (Poisson vs NB): chi2={lr_stat:,.1f}, p={lr_pval:.2e}")
print(f"  -> {'NB strongly preferred' if lr_pval < 0.001 else 'Poisson adequate'}")

# MODEL 3: NB + Season FE
print("\n" + "=" * 72)
print("MODEL 3: NEGATIVE BINOMIAL + SEASON FE")
print("=" * 72)
t0 = time.time()
m3 = sm.NegativeBinomial(y, X_season, offset=exposure, loglike_method="nb2").fit(disp=0, maxiter=200)
print(f"  Converged in {time.time()-t0:.1f}s")
print_results(m3, season_col_list, "NB + Season FE", SHOW_COLS)
results["3. NB + Season FE"] = (m3, season_col_list)

# MODEL 4: NB + Season FE + Score State (Replaced from 07b_models)
print("\n" + "=" * 72)
print("MODEL 4: NB + SEASON FE + SCORE STATE *** KEY MODEL ***")
print("=" * 72)
t0 = time.time()
m4 = sm.NegativeBinomial(y, X_ss, offset=exposure, loglike_method="nb2").fit(disp=0, maxiter=200)
print(f"  Converged in {time.time()-t0:.1f}s")
print_results(m4, ss_col_list, "NB + SS", SHOW_COLS + ["is_leading", "is_trailing"])
results["4. NB + SS"] = (m4, ss_col_list)

# MODEL 5: Linear FE (within-transformation) as robustness
print("\n" + "=" * 72)
print("MODEL 5: LINEAR OLS WITH PLAYER FIXED EFFECTS (within-transform)")
print("=" * 72)
print("  Dep var: corsi_rate_post_60 (shot rate per 60 min after split)")
print("  This avoids count-model complications; less efficient but")
print("  transparent and standard in applied micro.\n")

# Within-transformation: demean by player
panel_fe = panel.copy()
panel_fe["y_fe"] = panel_fe["corsi_rate_post_60"]
panel_fe = panel_fe.dropna(subset=["y_fe"])

# Demean all variables by player
fe_vars = ["y_fe", "scored_first_goal", "career_games_std", "career_season_std",
           "scored_prev", "games_missed_capped", "corsi_rate_pre_std"]
for v in fe_vars:
    panel_fe[f"{v}_dm"] = panel_fe[v] - panel_fe.groupby("PlayerID")[v].transform("mean")

X_fe = panel_fe[[f"{v}_dm" for v in fe_vars[1:]]].copy()
# Add season dummies (not demeaned — absorbed differently)
s_dum_fe = pd.get_dummies(panel_fe["Season_str"], prefix="s", drop_first=True, dtype=float)
X_fe = pd.concat([X_fe.reset_index(drop=True), s_dum_fe.reset_index(drop=True)], axis=1)
y_fe = panel_fe["y_fe_dm"].values

t0 = time.time()
m5 = sm.OLS(y_fe, X_fe).fit(cov_type="cluster", cov_kwds={"groups": panel_fe["PlayerID"].values})
print(f"  Converged in {time.time()-t0:.1f}s")
print(f"  R-squared (within): {m5.rsquared:.4f}")
print(f"  N: {int(m5.nobs):,}")
print(f"  Clusters (players): {panel_fe['PlayerID'].nunique():,}")

fe_show = [f"{v}_dm" for v in fe_vars[1:]]
fe_col_list = list(X_fe.columns)
print(f"\n  {'Variable':<30s} {'Coef':>8s} {'SE':>8s} {'p-val':>8s}")
print(f"  {'-'*60}")
for col in fe_show:
    i = fe_col_list.index(col)
    coef = m5.params.iloc[i]
    se = m5.bse.iloc[i]
    pv = m5.pvalues.iloc[i]
    sig = "***" if pv < 0.001 else "** " if pv < 0.01 else "* " if pv < 0.05 else "   "
    print(f"  {col:<30s} {coef:>8.4f} {se:>8.4f} {pv:>8.4f} {sig}")

results["5. Linear FE (within)"] = (m5, fe_col_list)

# SUMMARY TABLE
print("\n" + "=" * 72)
print("MODEL COMPARISON SUMMARY")
print("=" * 72)
print(f"\n  {'Model':<35s} {'LL':>12s} {'AIC':>12s} {'BIC':>12s} {'IRR/Coef':>10s} {'p-val':>10s}")
print(f"  {'-'*85}")

for label, (model, cols) in results.items():
    if "Linear" in label:
        # Linear model: report coefficient directly
        idx = cols.index("scored_first_goal_dm")
        coef = model.params.iloc[idx]
        pval = model.pvalues.iloc[idx]
        sig = "***" if pval < 0.001 else "** " if pval < 0.01 else "* " if pval < 0.05 else "   "
        print(f"  {label:<35s} {model.llf:>12,.1f} {model.aic:>12,.1f} {model.bic:>12,.1f} {coef:>9.4f}{sig} {pval:>10.4f}")
    else:
        idx = cols.index("scored_first_goal")
        irr = np.exp(model.params.iloc[idx])
        pval = model.pvalues.iloc[idx]
        sig = "***" if pval < 0.001 else "** " if pval < 0.01 else "* " if pval < 0.05 else "   "
        print(f"  {label:<35s} {model.llf:>12,.1f} {model.aic:>12,.1f} {model.bic:>12,.1f} {irr:>9.4f}{sig} {pval:>10.4f}")
