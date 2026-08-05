"""Figure 5 (paper): the count effect generalises to Kiva microloans.
Held-out F0.5 of the equal-weight vote vs a learned-weight elastic net as the Kiva positive class
is thinned with all negatives kept, so the base rate falls to the founder 1.9%. Values are read
from the cached 10x10 nested-CV draws (data/cache/kiva_D2.jsonl); regenerate that cache with
regenerate/kiva_D2_sweep.py (+ kiva_D2_k93.py) from data/kiva_rq_scores_run1.jsonl.
Self-contained: reads only files under ./data.
  python make_figure5.py   ->   figures/kiva_D2_final.pdf (+ .png)
"""
import json, collections
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
CACHE, OUT = HERE/"data"/"cache", HERE/"figures"; OUT.mkdir(exist_ok=True)
TEAL, NAVY = "#1f7a8c", "#022b3a"
PLOT_K = [15, 30, 60, 93, 130, 180, 222]     # 93 (=1.9%) is the marked founder-rate point

rows = [json.loads(l) for l in open(CACHE/"kiva_D2.jsonl") if l.strip()]
by = collections.defaultdict(list)
for r in rows: by[r["count"]].append(r)
K = [k for k in PLOT_K if k in by]
def mci(v):
    v=np.array(v); m=v.mean()
    return (m,0.0) if len(v)<2 else (m, stats.t.ppf(0.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
Vm,Ve,Em,Ee,PREV=[],[],[],[],[]
for k in K:
    vm,ve=mci([x["rrf_f05"] for x in by[k]]); em,ee=mci([x["en_f05"] for x in by[k]])
    Vm.append(vm);Ve.append(ve);Em.append(em);Ee.append(ee);PREV.append(np.mean([x["prevalence"] for x in by[k]])*100)

fig,ax=plt.subplots(figsize=(6.4,3.9))
ax.errorbar(K,Vm,yerr=Ve,fmt="s-",color=TEAL,lw=2,ms=6,capsize=3,label="equal-weight vote")
ax.errorbar(K,Em,yerr=Ee,fmt="o--",color=NAVY,lw=2,ms=5,capsize=3,label="elastic net")
ax.axvline(93,color="grey",ls=":",lw=1.2)
ax.text(90,0.208,"1.9%  (founder rate)",rotation=90,va="top",ha="right",fontsize=8,color="grey")
ax.axvline(222,color="grey",ls=":",lw=1.2)
ax.text(219,0.004,"full data (4.4%)",rotation=90,va="bottom",ha="right",fontsize=8,color="grey")
ax.set_xlabel("Number of positive examples"); ax.set_ylabel("Held-out $F_{0.5}$")
ax.set_ylim(0,0.22); ax.set_yticks([0.1,0.2]); ax.legend(fontsize=9,loc="upper left")
axt=ax.secondary_xaxis("top"); axt.set_xticks(K); axt.set_xticklabels([f"{p:.1f}" for p in PREV],fontsize=7.5); axt.set_xlabel("base rate (%)",fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"kiva_D2_final.png",dpi=150,bbox_inches="tight"); fig.savefig(OUT/"kiva_D2_final.pdf",bbox_inches="tight"); plt.close(fig)
print(f"Figure 5 -> {OUT/'kiva_D2_final.pdf'}")
for k,vm,em in zip(K,Vm,Em): print(f"  k={k:3d} prev={ {kk:pp for kk,pp in zip(K,PREV)}[k]:.2f}% vote={vm:.4f} en={em:.4f} gap={vm-em:+.4f}")
