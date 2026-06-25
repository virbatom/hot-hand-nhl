# 16_hurdle_model.py

"""
16_hurdle_model.py  --  FINAL MODEL

Hurdle model for corsi_post_split, separating two margins:
  PART 1 (logit):  P(takes >=1 post-split shot)   -- EXTENSIVE margin (the cold hand)
  PART 2 (ZTP):    shots | shots>=1               -- INTENSIVE margin (supervisor's ask)

The hurdle log-likelihood factorises, so the two parts are estimated
separately -- this is exact. PART 2 is exactly the zero-truncated Poisson
the supervisor requested; PART 1 is the logit that explains the cold hand.

At-risk sample : player-games with toi_5v5_post > 0 (a 5v5 shot was possible)
Exposure       : TOI -- covariate in the logit, offset in the ZTP
Baseline       : same as 07b / 13 / 15
Score state    : TWO clean variants (never mixed, to avoid the collinearity
                 seen in script 15):
                   (A) binary at split     [PRIMARY]
                   (B) on-ice time shares
A combined prediction reconciles the hurdle with the full-sample NB.
"""
from pathlib import Path
import warnings, time
import numpy as np, pandas as pd, polars as pl
import statsmodels.api as sm
from scipy.special import gammaln
from statsmodels.base.model import GenericLikelihoodModel
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM, PROCESSED = PROJECT_ROOT/"data"/"interim", PROJECT_ROOT/"data"/"processed"

# ---- ZTP (built-in or custom), same as script 15 ----
USE_BUILTIN = True
try:
    from statsmodels.discrete.truncated_model import TruncatedLFPoisson
except Exception:
    USE_BUILTIN = False

class ZTP(GenericLikelihoodModel):
    def __init__(self, endog, exog, offset=None, **k):
        super().__init__(endog, exog, **k)
        self._o = np.zeros(len(endog)) if offset is None else np.asarray(offset)
    def loglikeobs(self, p):
        xb = np.clip(np.asarray(self.exog) @ p + self._o, -30, 30)
        lam = np.exp(xb); y = np.asarray(self.endog)
        return -lam + y*np.log(lam) - gammaln(y+1) - np.log(-np.expm1(-lam))

def fit_ztp(y, X, off):
    Xv = np.asarray(X, float)
    try: sp = sm.Poisson(y, Xv, offset=off).fit(disp=0, maxiter=100).params
    except Exception: sp = None
    try:
        m = (TruncatedLFPoisson(y, Xv, offset=off, truncation=0) if USE_BUILTIN
             else ZTP(y, Xv, offset=off)).fit(start_params=sp, method="bfgs", maxiter=500, disp=0)
    except Exception:
        m = ZTP(y, Xv, offset=off).fit(start_params=sp, method="bfgs", maxiter=700, disp=0)
    return m

def line(m, cols, name, kind):
    i = cols.index(name); p = np.asarray(m.params); s = np.asarray(m.bse); pv = np.asarray(m.pvalues)
    sig = "***" if pv[i]<.001 else "** " if pv[i]<.01 else "*  " if pv[i]<.05 else "   "
    tag = "OR " if kind == "logit" else "IRR"
    print(f"    {name:<22s} coef={p[i]:+.4f}  {tag}={np.exp(p[i]):.4f}  p={pv[i]:.4f} {sig}")

# ---- load & prep ----
print("Loading panel (_ss)...")
panel = pl.read_parquet(PROCESSED/"analysis_panel_ss.parquet").to_pandas()
panel = panel[panel["toi_5v5_post"] > 0].copy()
panel = panel.dropna(subset=["corsi_post_split","scored_first_goal","toi_5v5_post",
                             "career_games_pre","career_season_num","Position"]).reset_index(drop=True)

panel["shot_again"]   = (panel["corsi_post_split"] >= 1).astype(int)
panel["log_toi_post"] = np.log(panel["toi_5v5_post"])
panel["is_forward"]   = (panel["Position"] == "F").astype(int)
panel["scored_prev"]  = panel["scored_prev_team_game"].fillna(0).astype(int)
panel["games_missed_capped"] = panel["games_missed"].fillna(0).clip(upper=20).astype(int)
panel["corsi_rate_pre"] = np.where(panel["toi_5v5_pre"]>0,
                            panel["corsi_pre_split"]/panel["toi_5v5_pre"]*3600, 0)
for c, n in [("career_games_pre","career_games_std"),("career_season_num","career_season_std")]:
    mu, sd = panel[c].mean(), panel[c].std(); panel[n] = (panel[c]-mu)/sd if sd>0 else 0
mu, sd = panel["corsi_rate_pre"].mean(), panel["corsi_rate_pre"].std()
panel["corsi_rate_pre_std"] = (panel["corsi_rate_pre"]-mu)/sd

ev = pl.read_parquet(INTERIM/"events.parquet")
gss = (ev.filter((pl.col("Goal")==1)&(pl.col("Season")>=20102011)&(pl.col("SeasonState")=="regular")&
        (pl.col("StrengthState").is_in(["5v5","ENF"])))
       .select(pl.col("Player1_ID").alias("PlayerID"),"GameID","GameTime",pl.col("ScoreState").alias("r"))
       .sort(["PlayerID","GameID","GameTime"]).group_by(["PlayerID","GameID"]).first().to_pandas())
gss["s"] = pd.to_numeric(gss["r"], errors="coerce").fillna(0).astype(int)
panel = panel.merge(gss[["PlayerID","GameID","s"]], on=["PlayerID","GameID"], how="left")
panel["s"] = panel["s"].fillna(0).astype(int)
panel["is_leading"]  = (panel["s"] > 0).astype(int)
panel["is_trailing"] = (panel["s"] < 0).astype(int)

panel["Season_str"] = panel["Season"].astype(str)
SD = pd.get_dummies(panel["Season_str"], prefix="s", drop_first=True, dtype=float)

print(f"  At-risk sample (toi_5v5_post>0): {len(panel):,}")
print(f"  Shoot again (Y>=1): {panel['shot_again'].sum():,} ({panel['shot_again'].mean()*100:.1f}%)")
print(f"  Zero post-split shots: {(panel['shot_again']==0).sum():,} "
      f"({(panel['shot_again']==0).mean()*100:.1f}%)  <- the extensive margin")

base = ["scored_first_goal","is_forward","career_games_std","career_season_std",
        "scored_prev","games_missed_capped","corsi_rate_pre_std"]
variants = {
    "A) binary score-state (PRIMARY)": ["is_leading", "is_trailing"],
    "B) on-ice time shares":           ["pct_onice_leading", "pct_onice_trailing"],
}

results = {}
for vname, ss in variants.items():
    print("\n" + "="*64)
    print(f"HURDLE MODEL  --  score state = {vname}")
    print("="*64)

    # PART 1: LOGIT (extensive margin), exposure as COVARIATE, full at-risk sample
    Xl = pd.concat([sm.add_constant(panel[base+ss].assign(log_toi_post=panel["log_toi_post"])), SD], axis=1)
    cl = list(Xl.columns)
    t0 = time.time()
    logit = sm.Logit(panel["shot_again"].values, Xl).fit(disp=0, maxiter=200)
    print(f"\n  PART 1 -- LOGIT  P(shoot again)   [n={len(panel):,}, {time.time()-t0:.1f}s, pseudoR2={logit.prsquared:.4f}]")
    line(logit, cl, "scored_first_goal", "logit")
    for v in ss: line(logit, cl, v, "logit")
    line(logit, cl, "log_toi_post", "logit")

    # PART 2: ZTP (intensive margin), exposure as OFFSET, Y>0 subsample
    sub  = panel[panel["shot_again"] == 1]
    Xz   = pd.concat([sm.add_constant(sub[base+ss]), SD.loc[sub.index]], axis=1)
    cz   = list(Xz.columns)
    t0 = time.time()
    ztp  = fit_ztp(sub["corsi_post_split"].values.astype(float), Xz, sub["log_toi_post"].values)
    print(f"\n  PART 2 -- ZTP  shots | shots>=1   [n={len(sub):,}, {time.time()-t0:.1f}s]")
    line(ztp, cz, "scored_first_goal", "ztp")
    for v in ss: line(ztp, cz, v, "ztp")

    # COMBINED: predicted E[shots] over the at-risk sample, scored=0 vs 1
    Xz_full = pd.concat([sm.add_constant(panel[base+ss]), SD], axis=1)[cz]
    off_full = panel["log_toi_post"].values
    def predict_EY(val):
        a = Xl.copy(); a["scored_first_goal"] = val
        p_pos = logit.predict(a[cl]).values
        b = Xz_full.copy(); b["scored_first_goal"] = val
        lam = np.exp(np.asarray(b[cz], float) @ np.asarray(ztp.params) + off_full)
        return float(np.mean(p_pos * (lam / (1 - np.exp(-lam)))))
    EY0, EY1 = predict_EY(0), predict_EY(1)
    print(f"\n  COMBINED predicted E[shots] (avg over at-risk sample):")
    print(f"    not scored = {EY0:.4f}   scored = {EY1:.4f}   ratio = {EY1/EY0:.4f}")
    results[vname] = (logit, cl, ztp, cz, EY0, EY1)

# ---- summary ----
print("\n" + "="*64)
print("HURDLE SUMMARY -- effect of scoring on each margin")
print("="*64)
print(f"\n  {'Variant':<34s}{'Logit OR':>10s}{'ZTP IRR':>10s}{'E[Y] ratio':>12s}")
print("  " + "-"*64)
for vname, (logit, cl, ztp, cz, EY0, EY1) in results.items():
    orv  = np.exp(np.asarray(logit.params)[cl.index("scored_first_goal")])
    irrv = np.exp(np.asarray(ztp.params)[cz.index("scored_first_goal")])
    print(f"  {vname:<34s}{orv:>10.4f}{irrv:>10.4f}{EY1/EY0:>12.4f}")