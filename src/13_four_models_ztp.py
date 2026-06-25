# 13_four_models_ztp.py

"""
13_four_models_ztp.py

Four-model specification using ZERO-TRUNCATED POISSON (ZTP) on corsi_post_split,
to demonstrate that BOTH time-on-ice exposure and score state matter before
building the final model.

Sample: corsi_post_split >= 1  (zero-truncated)
        AND toi_5v5_post > 0 AND time_after_split > 0
        -> all four models run on the IDENTICAL sample.

Baseline variables (same as 07b_models.py):
  scored_first_goal, is_forward, career_games_std, career_season_std,
  scored_prev, games_missed_capped, corsi_rate_pre_std   + Season FE

Models:
  M1: Baseline,  game-clock exposure,  NO score state
  M2: Baseline,  TOI exposure,         NO score state
  M3: Baseline,  TOI exposure,         + score state
  M4: Baseline,  game-clock exposure,  + score state

Score state = is_leading / is_trailing at the first-goal (split) moment.
"""
from pathlib import Path
import warnings, time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import polars as pl

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM   = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Zero-Truncated Poisson: built-in if available, else custom
from scipy.special import gammaln
from statsmodels.base.model import GenericLikelihoodModel

USE_BUILTIN = True
try:
    from statsmodels.discrete.truncated_model import TruncatedLFPoisson
except Exception:
    USE_BUILTIN = False

class ZeroTruncatedPoisson(GenericLikelihoodModel):
    """ZTP via explicit log-likelihood (numerically stabilised)."""
    def __init__(self, endog, exog, offset=None, **kwds):
        super().__init__(endog, exog, **kwds)
        self._offset = np.zeros(len(endog)) if offset is None else np.asarray(offset)
    def loglikeobs(self, params):
        xb = np.clip(np.asarray(self.exog) @ params + self._offset, -30, 30)
        lam = np.exp(xb)
        y = np.asarray(self.endog)
        # log P(Y=y | Y>0) = -lam + y*log(lam) - log(y!) - log(1 - e^-lam)
        return -lam + y*np.log(lam) - gammaln(y+1) - np.log(-np.expm1(-lam))

def fit_ztp(y, X, offset, label):
    t0 = time.time()
    Xv = np.asarray(X, dtype=float)
    cols = list(X.columns)
    try:
        sp = sm.Poisson(y, Xv, offset=offset).fit(disp=0, maxiter=100).params
    except Exception:
        sp = None
    if USE_BUILTIN:
        try:
            m = TruncatedLFPoisson(y, Xv, offset=offset, truncation=0).fit(
                start_params=sp, method="bfgs", maxiter=400, disp=0)
            method = "built-in ZTP"
        except Exception as e:
            print(f"    built-in failed ({e}); using custom")
            m = ZeroTruncatedPoisson(y, Xv, offset=offset).fit(
                start_params=sp, method="bfgs", maxiter=600, disp=0)
            method = "custom ZTP"
    else:
        m = ZeroTruncatedPoisson(y, Xv, offset=offset).fit(
            start_params=sp, method="bfgs", maxiter=600, disp=0)
        method = "custom ZTP"
    conv = getattr(m, "mle_retvals", {}).get("converged", "?")
    print(f"  [{label}] {method}, {time.time()-t0:.1f}s, converged={conv}")
    return m, cols

def show(m, cols, focus):
    p  = np.asarray(m.params); s = np.asarray(m.bse); pv = np.asarray(m.pvalues)
    print(f"  {'Variable':<26s}{'Coef':>9s}{'SE':>9s}{'IRR':>9s}{'p':>9s}")
    print(f"  {'-'*62}")
    for c in focus:
        if c not in cols: continue
        i = cols.index(c)
        sig = "***" if pv[i]<.001 else "** " if pv[i]<.01 else "*  " if pv[i]<.05 else "   "
        print(f"  {c:<26s}{p[i]:>9.4f}{s[i]:>9.4f}{np.exp(p[i]):>9.4f}{pv[i]:>9.4f} {sig}")
    print(f"  LL={m.llf:,.1f}  AIC={m.aic:,.1f}  N={int(m.nobs):,}")

# 1. Load & prepare (identical sample for all four models)
print("Loading panel...")
panel = pl.read_parquet(PROCESSED / "analysis_panel.parquet").to_pandas()
print(f"  Full panel: {len(panel):,}")

panel = panel[(panel["corsi_post_split"] >= 1) &
              (panel["toi_5v5_post"] > 0) &
              (panel["time_after_split"] > 0)].copy()
key = ["corsi_post_split","scored_first_goal","toi_5v5_post","time_after_split",
       "career_games_pre","career_season_num","Position"]
panel = panel.dropna(subset=key).reset_index(drop=True)
print(f"  Zero-truncated sample (corsi_post_split>=1): {len(panel):,}")

# offsets
panel["log_time_post"] = np.log(panel["time_after_split"])  # game-clock
panel["log_toi_post"]  = np.log(panel["toi_5v5_post"])      # TOI

# baseline vars
panel["is_forward"]          = (panel["Position"] == "F").astype(int)
panel["scored_prev"]         = panel["scored_prev_team_game"].fillna(0).astype(int)
panel["games_missed_capped"] = panel["games_missed"].fillna(0).clip(upper=20).astype(int)
panel["corsi_rate_pre"]      = np.where(panel["toi_5v5_pre"] > 0,
                                panel["corsi_pre_split"] / panel["toi_5v5_pre"] * 3600, 0)
for col, new in [("career_games_pre","career_games_std"),
                 ("career_season_num","career_season_std")]:
    mu, sd = panel[col].mean(), panel[col].std()
    panel[new] = (panel[col]-mu)/sd if sd > 0 else 0
mu, sd = panel["corsi_rate_pre"].mean(), panel["corsi_rate_pre"].std()
panel["corsi_rate_pre_std"] = (panel["corsi_rate_pre"]-mu)/sd if sd > 0 else 0

# score state at first goal (same recompute as 07b)
events = pl.read_parquet(INTERIM / "events.parquet")
goal_ss = (events
    .filter((pl.col("Goal")==1) & (pl.col("Season")>=20102011) &
            (pl.col("SeasonState")=="regular") &
            (pl.col("StrengthState").is_in(["5v5","ENF"])))
    .select(pl.col("Player1_ID").alias("PlayerID"),"GameID","GameTime",
            pl.col("ScoreState").alias("ss_raw"))
    .sort(["PlayerID","GameID","GameTime"]).group_by(["PlayerID","GameID"]).first()
    .to_pandas())
goal_ss["sss"] = pd.to_numeric(goal_ss["ss_raw"], errors="coerce").fillna(0).astype(int)
panel = panel.merge(goal_ss[["PlayerID","GameID","sss"]], on=["PlayerID","GameID"], how="left")
panel["sss"] = panel["sss"].fillna(0).astype(int)
panel["is_leading"]  = (panel["sss"] > 0).astype(int)
panel["is_trailing"] = (panel["sss"] < 0).astype(int)

panel["Season_str"] = panel["Season"].astype(str)
season_dummies = pd.get_dummies(panel["Season_str"], prefix="s", drop_first=True, dtype=float)

print(f"  Treatment: {panel['scored_first_goal'].sum():,} "
      f"({panel['scored_first_goal'].mean()*100:.1f}%)")
print(f"  Dep var mean (post-split shots, >=1): {panel['corsi_post_split'].mean():.3f}")

# 2. Build design matrices
y = panel["corsi_post_split"].values.astype(float)
base_vars = ["scored_first_goal","is_forward","career_games_std",
             "career_season_std","scored_prev","games_missed_capped","corsi_rate_pre_std"]
ss_vars   = base_vars + ["is_leading","is_trailing"]

X_base = pd.concat([sm.add_constant(panel[base_vars]), season_dummies], axis=1)
X_ss   = pd.concat([sm.add_constant(panel[ss_vars]),   season_dummies], axis=1)

off_gc  = panel["log_time_post"].values
off_toi = panel["log_toi_post"].values

FOCUS_BASE = ["const"] + base_vars
FOCUS_SS   = ["const"] + ss_vars

# 3. Fit four models
print("\n" + "="*64)
print("M1: BASELINE | GAME-CLOCK exposure | NO score state")
print("="*64)
m1, c1 = fit_ztp(y, X_base, off_gc, "M1")
show(m1, c1, FOCUS_BASE)

print("\n" + "="*64)
print("M2: BASELINE | TOI exposure | NO score state")
print("="*64)
m2, c2 = fit_ztp(y, X_base, off_toi, "M2")
show(m2, c2, FOCUS_BASE)

print("\n" + "="*64)
print("M3: BASELINE | TOI exposure | + SCORE STATE")
print("="*64)
m3, c3 = fit_ztp(y, X_ss, off_toi, "M3")
show(m3, c3, FOCUS_SS)

print("\n" + "="*64)
print("M4: BASELINE | GAME-CLOCK exposure | + SCORE STATE")
print("="*64)
m4, c4 = fit_ztp(y, X_ss, off_gc, "M4")
show(m4, c4, FOCUS_SS)

# 4. Specification comparison table (treatment effect)
def irr_p(m, cols):
    i = cols.index("scored_first_goal")
    return np.exp(np.asarray(m.params)[i]), np.asarray(m.pvalues)[i]

rows = [
    ("M1  Baseline, game-clock, no SS", *irr_p(m1, c1)),
    ("M2  Baseline, TOI,        no SS", *irr_p(m2, c2)),
    ("M3  Baseline, TOI,        + SS ", *irr_p(m3, c3)),
    ("M4  Baseline, game-clock, + SS ", *irr_p(m4, c4)),
]

print("\n" + "="*64)
print("SPECIFICATION TABLE — effect of scoring (scored_first_goal)")
print("Zero-Truncated Poisson, identical sample, IRR for shot count")
print("="*64)
print(f"\n  {'Model':<34s}{'IRR':>9s}{'p-value':>10s}{'Sig':>5s}")
print(f"  {'-'*58}")
for label, irr, p in rows:
    sig = "***" if p<.001 else "** " if p<.01 else "*  " if p<.05 else "   "
    print(f"  {label:<34s}{irr:>9.4f}{p:>10.4f}{sig:>5s}")
