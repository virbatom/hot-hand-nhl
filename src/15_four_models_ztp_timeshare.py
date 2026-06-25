# 15_four_models_ztp_timeshare.py

"""
15_four_models_ztp_timeshare.py
Four-model ZTP comparison, score-state controls = binary (at split)
+ team time-shares + player on-ice time-shares, all ALONGSIDE.
tied is the reference category for both share sets.
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

def fit_ztp(y, X, off, lab):
    t0=time.time(); Xv=np.asarray(X,float); cols=list(X.columns)
    try: sp=sm.Poisson(y,Xv,offset=off).fit(disp=0,maxiter=100).params
    except Exception: sp=None
    try:
        m=(TruncatedLFPoisson(y,Xv,offset=off,truncation=0) if USE_BUILTIN else ZTP(y,Xv,offset=off))\
            .fit(start_params=sp, method="bfgs", maxiter=500, disp=0)
        meth="built-in" if USE_BUILTIN else "custom"
    except Exception as e:
        print(f"   {e}; custom"); m=ZTP(y,Xv,offset=off).fit(start_params=sp,method="bfgs",maxiter=700,disp=0); meth="custom"
    print(f"  [{lab}] {meth} ZTP, {time.time()-t0:.1f}s, conv={getattr(m,'mle_retvals',{}).get('converged','?')}")
    return m, cols

def show(m, cols, focus):
    p,s,pv = np.asarray(m.params),np.asarray(m.bse),np.asarray(m.pvalues)
    print(f"  {'Variable':<22s}{'Coef':>9s}{'SE':>9s}{'IRR':>9s}{'p':>9s}")
    for c in focus:
        if c not in cols: continue
        i=cols.index(c); sig="***" if pv[i]<.001 else "** " if pv[i]<.01 else "*  " if pv[i]<.05 else "   "
        print(f"  {c:<22s}{p[i]:>9.4f}{s[i]:>9.4f}{np.exp(p[i]):>9.4f}{pv[i]:>9.4f} {sig}")
    print(f"  LL={m.llf:,.1f}  AIC={m.aic:,.1f}  N={int(m.nobs):,}")

print("Loading panel (_ss)...")
panel = pl.read_parquet(PROCESSED/"analysis_panel_ss.parquet").to_pandas()
panel = panel[(panel["corsi_post_split"]>=1)&(panel["toi_5v5_post"]>0)&(panel["time_after_split"]>0)].copy()
panel = panel.dropna(subset=["corsi_post_split","scored_first_goal","toi_5v5_post",
                             "time_after_split","career_games_pre","career_season_num","Position"]).reset_index(drop=True)
print(f"  ZTP sample: {len(panel):,}")

panel["log_time_post"]=np.log(panel["time_after_split"]); panel["log_toi_post"]=np.log(panel["toi_5v5_post"])
panel["is_forward"]=(panel["Position"]=="F").astype(int)
panel["scored_prev"]=panel["scored_prev_team_game"].fillna(0).astype(int)
panel["games_missed_capped"]=panel["games_missed"].fillna(0).clip(upper=20).astype(int)
panel["corsi_rate_pre"]=np.where(panel["toi_5v5_pre"]>0,panel["corsi_pre_split"]/panel["toi_5v5_pre"]*3600,0)
for c,n in [("career_games_pre","career_games_std"),("career_season_num","career_season_std")]:
    mu,sd=panel[c].mean(),panel[c].std(); panel[n]=(panel[c]-mu)/sd if sd>0 else 0
mu,sd=panel["corsi_rate_pre"].mean(),panel["corsi_rate_pre"].std(); panel["corsi_rate_pre_std"]=(panel["corsi_rate_pre"]-mu)/sd

# is_leading/is_trailing at split (from events, same as 13)
ev=pl.read_parquet(INTERIM/"events.parquet")
gss=(ev.filter((pl.col("Goal")==1)&(pl.col("Season")>=20102011)&(pl.col("SeasonState")=="regular")&
        (pl.col("StrengthState").is_in(["5v5","ENF"])))
      .select(pl.col("Player1_ID").alias("PlayerID"),"GameID","GameTime",pl.col("ScoreState").alias("r"))
      .sort(["PlayerID","GameID","GameTime"]).group_by(["PlayerID","GameID"]).first().to_pandas())
gss["s"]=pd.to_numeric(gss["r"],errors="coerce").fillna(0).astype(int)
panel=panel.merge(gss[["PlayerID","GameID","s"]],on=["PlayerID","GameID"],how="left")
panel["s"]=panel["s"].fillna(0).astype(int)
panel["is_leading"]=(panel["s"]>0).astype(int); panel["is_trailing"]=(panel["s"]<0).astype(int)

panel["Season_str"]=panel["Season"].astype(str)
sd_=pd.get_dummies(panel["Season_str"],prefix="s",drop_first=True,dtype=float)

y=panel["corsi_post_split"].values.astype(float)
base=["scored_first_goal","is_forward","career_games_std","career_season_std",
      "scored_prev","games_missed_capped","corsi_rate_pre_std"]
ss=base+["is_leading","is_trailing",
         "pct_time_leading","pct_time_trailing",      # tied = reference
         "pct_onice_leading","pct_onice_trailing"]    # tied = reference
Xb=pd.concat([sm.add_constant(panel[base]),sd_],axis=1)
Xs=pd.concat([sm.add_constant(panel[ss]),sd_],axis=1)
ogc,otoi=panel["log_time_post"].values,panel["log_toi_post"].values
FB=["const"]+base; FS=["const"]+ss

print("\n--- M1: baseline, game-clock, no SS ---"); m1,c1=fit_ztp(y,Xb,ogc,"M1"); show(m1,c1,FB)
print("\n--- M2: baseline, TOI, no SS ---");        m2,c2=fit_ztp(y,Xb,otoi,"M2"); show(m2,c2,FB)
print("\n--- M3: baseline, TOI, + SS(all) ---");    m3,c3=fit_ztp(y,Xs,otoi,"M3"); show(m3,c3,FS)
print("\n--- M4: baseline, game-clock, + SS(all) ---"); m4,c4=fit_ztp(y,Xs,ogc,"M4"); show(m4,c4,FS)

def ip(m,c): i=c.index("scored_first_goal"); return np.exp(np.asarray(m.params)[i]),np.asarray(m.pvalues)[i]
print("\n"+"="*60+"\nSPEC TABLE (with binary + team + on-ice score-state)\n"+"="*60)
print(f"  {'Model':<32s}{'IRR':>9s}{'p':>10s}")
for lab,(irr,p) in [("M1 base, game-clock, no SS",ip(m1,c1)),("M2 base, TOI, no SS",ip(m2,c2)),
                    ("M3 base, TOI, +SS",ip(m3,c3)),("M4 base, game-clock, +SS",ip(m4,c4))]:
    sig="***" if p<.001 else "** " if p<.01 else "*  " if p<.05 else "   "
    print(f"  {lab:<32s}{irr:>9.4f}{p:>10.4f} {sig}")
print(f"\n  N = {int(m1.nobs):,}")