import pandas as pd, numpy as np, glob, os
from sksurv.metrics import concordance_index_censored

base='/nvme/jin/HECKTOR26/task2_3_prediction/results/predictions'
lab=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','Relapse','RFS']]
rsf_df=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','rf_rfs_risk']].rename(columns={'rf_rfs_risk':'RSF'})

# ---- Collect OOF risk for all DL configurations ----
pool={}
for f in sorted(glob.glob(f'{base}/*/*/oof_predictions.csv')):
    d=pd.read_csv(f)
    if 'risk_score' not in d.columns: continue
    name=os.path.relpath(f,base).replace('/oof_predictions.csv','')
    pool[name]=d[['PatientID','risk_score']]

# Align all sources to the same batch of patients
m=lab.copy()
for name,d in pool.items():
    m=m.merge(d.rename(columns={'risk_score':name}),on='PatientID',how='inner')
m=m.merge(rsf_df,on='PatientID',how='inner')
e=m['Relapse'].astype(bool); t=m['RFS']
def cidx(col): return concordance_index_censored(e,t,m[col])[0]

# ---- Mechanical Rules ----
THR_STRENGTH=0.65   # Pre-determined: Individual member OOF must be >0.65
THR_CORR=0.50       # Pre-defined: The correlation with selected members must be <0.5
print("="*60)
print(f"Mechanical Rules: Strength>{THR_STRENGTH}, corelated to the selected member in Spearman <{THR_CORR}")
print("="*60)

# 1. All DL configurations are sorted by intensity.
dl_names=list(pool.keys())
strengths={n:cidx(n) for n in dl_names}
ranked=sorted(dl_names,key=lambda n:strengths[n],reverse=True)

# 2. Greedy Choice
selected=[]
for n in ranked:
    if strengths[n]<=THR_STRENGTH:
        continue
    # Maximum correlation with selected members
    if selected:
        corrs=[m[[n,s]].corr(method='spearman').iloc[0,1] for s in selected]
        maxc=max(corrs)
    else:
        maxc=0.0
    if maxc<THR_CORR:
        selected.append(n)
        print(f"  selected: {n:55s} OOF={strengths[n]:.4f} maxcorr={maxc:.3f}")
    else:
        print(f"  skipped: {n:55s} OOF={strengths[n]:.4f} maxcorr={maxc:.3f} (correlation too high)")

print(f"\nMechanical rules selected {len(selected)} members:")
for s in selected: print(f"    {s}")

# Check if RSF is selected
rsf_str=cidx('RSF')
print(f"\nRSF Strength={rsf_str:.4f} -> {'Selected' if rsf_str>THR_STRENGTH else 'Excluded (weaker than the threshold)'}")

# ---- Ensemble Resluts ----
for c in selected+['RSF']: m[c+'_r']=(m[c].rank()-1)/(len(m)-1)
def fuse(cols,weights=None):
    if weights is None: weights=[1/len(cols)]*len(cols)
    r=sum(w*m[c+'_r'] for c,w in zip(cols,weights))
    return concordance_index_censored(e,t,r)[0]

print("\n"+"="*60)
print("Fusion results (equal-weighted rank average)")
print("="*60)
print(f"  Mechanical rules members equal rights:           {fuse(selected):.4f}")

# Short name mapping (assuming A/B/C are selected)
print("\n  Comparison of each combination (using short names):")
A='multitask/clinical+CT+PET+hard_T+hard_N'
B='task3/clinical+CT+PET'
C='task3/CT+PET+prob_T+prob_N+hard_T+hard_N'
combos={
 'A':[A],'B':[B],'C':[C],'RSF':['RSF'],
 'A+B':[A,B],'A+C':[A,C],'B+C':[B,C],'A+B+C':[A,B,C],
 'RSF+A':['RSF',A],'RSF+B':['RSF',B],'RSF+C':['RSF',C],
 'RSF+A+B':['RSF',A,B],'RSF+A+C':['RSF',A,C],'RSF+B+C':['RSF',B,C],
 'RSF+A+B+C':['RSF',A,B,C],
}
for name,cols in combos.items():
    try: print(f"    {name:12s}: {fuse(cols):.4f}")
    except Exception as ex: print(f"    {name:12s}: ERR")
