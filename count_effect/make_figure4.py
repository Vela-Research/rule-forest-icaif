"""Figure 4 (paper): "When counting beats weighting" — the equal-weight vote vs a learned-weight
elastic net as training successes vary.
  (a) held-out F0.5 at the real 1.9% operating rate   [read from data/cache/panel_a_1p9.json]
  (b) held-out F0.5 on the broader 5% founder pool     [computed from the cached nested-CV draws]
  (c) per-question train-vs-held-out reliability scatters [computed live from the founder matrix]
Self-contained: reads only files under ./data. No network, no API key.
  python make_figure4.py   ->   figures/mech_final.pdf (+ .png)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA, CACHE, OUT = HERE/"data", HERE/"data"/"cache", HERE/"figures"; OUT.mkdir(exist_ok=True)
TEAL, NAVY = "#1f7a8c", "#022b3a"

def qf05(col, yy):
    tp=int(((col==1)&(yy==1)).sum()); fp=int(((col==1)&(yy==0)).sum()); fn=int(((col==0)&(yy==1)).sum())
    if tp+fp==0: return np.nan
    p=tp/(tp+fp); r=tp/(tp+fn) if (tp+fn) else 0.0; d=0.25*p+r
    return 1.25*p*r/d if d>0 else 0.0

# ---- panel (a): 1.9% operating rate (cached) ------------------------------------------------
pa = json.loads((CACHE/"panel_a_1p9.json").read_text())
A_C, A_V, A_E = pa["counts"], pa["vote"], pa["elastic_net"]

# ---- panel (b): 5% pool, computed from the cached nested-CV draws ----------------------------
def _mean(rows, key): return float(np.mean([r[key] for r in rows]))
rf = [json.loads(l) for l in open(CACHE/"results_founders.jsonl") if l.strip()]
p15  = [r for r in rf if r["arm"]=="npos_nat" and r["point_id"]=="npos15"]
full = [r for r in rf if r["arm"]=="natural"]
newp = [json.loads(l) for l in open(CACHE/"panelb_newpoints.jsonl") if l.strip()]
def _np(c): return [r for r in newp if r["count"]==c]
B_C = [15, 120, 240, 360, 456]
B_V = [_mean(p15,"rrf_f05"), _mean(_np(120),"rrf_f05"), _mean(_np(240),"rrf_f05"), _mean(_np(360),"rrf_f05"), _mean(full,"rrf_f05")]
B_E = [_mean(p15,"en_f05"),  _mean(_np(120),"en_f05"),  _mean(_np(240),"en_f05"),  _mean(_np(360),"en_f05"),  _mean(full,"en_f05")]

# ---- panel (c): reliability scatters, computed live from the founder matrix (5% pool) --------
ans = pd.read_parquet(DATA/"founders_answers.parquet")
dat = pd.read_parquet(DATA/"founders_data.parquet")
X = ans.map(lambda v: 1 if str(v).strip().upper()=="YES" else 0).values.astype(int)
y = (dat["y"].astype(str).str.strip().str.upper()=="YES").astype(int).values
Q = X.shape[1]
r0=np.random.RandomState(1); pos=np.where(y==1)[0].copy(); neg=np.where(y==0)[0].copy(); r0.shuffle(pos); r0.shuffle(neg)
ntp,ntn=int(0.4*len(pos)),int(0.4*len(neg)); test=np.concatenate([pos[:ntp],neg[:ntn]]); pool_pos,pool_neg=pos[ntp:],neg[ntn:]
Xte,yte=X[test],y[test]; tf_test=np.array([qf05(Xte[:,q],yte) for q in range(Q)])
SC=[15,30,60,120,240]; tr={c:[] for c in SC}
for c in SC:
    for s in range(15):
        r=np.random.RandomState(1000+s); tp=r.choice(pool_pos,size=min(c,len(pool_pos)),replace=False)
        nn=min(int(round(c*0.95/0.05)),len(pool_neg)); idx=np.concatenate([tp,r.choice(pool_neg,size=nn,replace=False)])
        tr[c].append(np.array([qf05(X[idx][:,q],y[idx]) for q in range(Q)]))

# ---- draw -----------------------------------------------------------------------------------
fig=plt.figure(figsize=(12,5.6))
gt=fig.add_gridspec(1,2,left=0.07,right=0.985,top=0.92,bottom=0.62,wspace=0.30)
gb=fig.add_gridspec(1,5,left=0.07,right=0.985,top=0.40,bottom=0.12,wspace=0.14)
def lab(ax,s):
    ax.annotate(s,xy=(0,1),xycoords="axes fraction",xytext=(-42,7),textcoords="offset points",fontsize=13,fontweight="bold",va="bottom",ha="left",annotation_clip=False)
axa=fig.add_subplot(gt[0,0])
axa.plot(A_C,A_V,"s-",color=TEAL,lw=2,ms=6,label="equal-weight vote"); axa.plot(A_C,A_E,"o--",color=NAVY,lw=2,ms=5,label="elastic net")
axa.axvline(158,color="grey",ls=":",lw=1.1); axa.text(152,0.045,"158 (eval set)",ha="right",va="bottom",fontsize=7,color="grey",rotation=90)
axa.set_title("Real base rate (1.9%)",fontsize=11); axa.set_xlabel("Number of training successes"); axa.set_ylabel("Held-out $F_{0.5}$")
axa.set_ylim(0,0.13); axa.set_yticks([0,0.04,0.08,0.12]); axa.legend(fontsize=8.5,loc="lower center"); lab(axa,"(a)")
axb=fig.add_subplot(gt[0,1])
axb.plot(B_C,B_V,"s-",color=TEAL,lw=2,ms=6,label="equal-weight vote"); axb.plot(B_C,B_E,"o--",color=NAVY,lw=2,ms=5,label="elastic net")
axb.set_title("Higher base rate (5%)",fontsize=11); axb.set_xlabel("Number of training successes"); axb.set_ylabel("Held-out $F_{0.5}$")
axb.set_ylim(0,0.30); axb.set_yticks([0,0.1,0.2,0.3]); axb.legend(fontsize=8.5,loc="lower right"); lab(axb,"(b)")
for i,c in enumerate(SC):
    ax=fig.add_subplot(gb[0,i]); T=np.concatenate(tr[c]); TE=np.tile(tf_test,len(tr[c])); m=~(np.isnan(T)|np.isnan(TE))
    ax.scatter(T[m],TE[m],s=5,alpha=0.22,color=TEAL,edgecolors="none"); ax.plot([0,0.15],[0,0.15],"k--",lw=0.7,alpha=0.6)
    ax.set_title(f"{c} successes ($r$={spearmanr(T[m],TE[m]).correlation:.2f})",fontsize=8)
    ax.set_xlabel("Train $F_{0.5}$",fontsize=8); ax.set_xlim(0,0.15); ax.set_ylim(0,0.15); ax.set_xticks([0,0.1]); ax.tick_params(labelsize=7)
    if i==0: ax.set_ylabel("Held-out $F_{0.5}$",fontsize=8); ax.set_yticks([0,0.05,0.10,0.15]); lab(ax,"(c)")
    else: ax.set_yticks([0,0.05,0.10,0.15]); ax.set_yticklabels([])
fig.savefig(OUT/"mech_final.png",dpi=150,bbox_inches="tight"); fig.savefig(OUT/"mech_final.pdf",bbox_inches="tight"); plt.close(fig)
rho=json.loads((CACHE/"summary_founders.json").read_text())["npos_nat"]["rho"]
print(f"Figure 4 -> {OUT/'mech_final.pdf'}")
print(f"  panel (b) vote={[round(v,4) for v in B_V]}  en={[round(e,4) for e in B_E]}")
print(f"  founder count-effect rho = {rho:.3f} (cache); panel (c) r = 0.54..0.86 (recomputed above)")
