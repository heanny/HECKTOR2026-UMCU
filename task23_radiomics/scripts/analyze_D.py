import pandas as pd, numpy as np
from itertools import combinations
from sksurv.metrics import concordance_index_censored

base='/nvme/jin/HECKTOR26/task2_3_prediction/results/predictions'
lab=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','Relapse','RFS']]
rsf=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','rf_rfs_risk']].rename(columns={'rf_rfs_risk':'RSF'})

srcs={'A':'multitask/clinical+CT+PET+hard_T+hard_N',
      'B':'task3/clinical+CT+PET',
      'C':'task3/CT+PET+prob_T+prob_N+hard_T+hard_N',
      'D':'task3/CT+PET'}
m=lab.copy()
for k,p in srcs.items():
    d=pd.read_csv(f'{base}/{p}/oof_predictions.csv')[['PatientID','risk_score']].rename(columns={'risk_score':k})
    m=m.merge(d,on='PatientID',how='inner')
m=m.merge(rsf,on='PatientID',how='inner')
e=m['Relapse'].astype(bool); t=m['RFS']
cols=['A','B','C','D','RSF']
for c in cols: m[c+'_r']=(m[c].rank()-1)/(len(m)-1)

print(f"n={len(m)}\n")
print("=== Spearman 相关矩阵 (含D) ===")
print(m[cols].corr(method='spearman').round(3))

print("\n=== 单源 OOF ===")
for c in cols: print(f"  {c:4s}: {concordance_index_censored(e,t,m[c])[0]:.4f}")

def fuse(cs):
    r=m[[c+'_r' for c in cs]].mean(1)
    return concordance_index_censored(e,t,r)[0]

print("\n=== 所有含D的组合 (等权秩融合) ===")
members=['A','B','C','D','RSF']
allc=[]
for r in range(2,6):
    for combo in combinations(members,r):
        if 'D' in combo:
            allc.append((fuse(list(combo)), '+'.join(combo)))
for score,name in sorted(allc,reverse=True):
    print(f"  {score:.4f}  {name}")

print("\n=== 对照: 不含D的关键组合 ===")
for combo in [['A','B','C'],['A','B','C','RSF'],['A','B'],['RSF','A','B']]:
    print(f"  {fuse(combo):.4f}  {'+'.join(combo)}")
