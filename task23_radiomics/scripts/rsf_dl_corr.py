import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored

base='/nvme/jin/HECKTOR26/task2_3_prediction/results/predictions'
lab=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','Relapse','RFS']]
rsf=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','rf_rfs_risk']].rename(columns={'rf_rfs_risk':'RSF'})

srcs={'A':'multitask/clinical+CT+PET+hard_T+hard_N',
      'B':'task3/clinical+CT+PET',
      'C':'task3/CT+PET+prob_T+prob_N+hard_T+hard_N'}
m=lab.copy()
for k,p in srcs.items():
    d=pd.read_csv(f'{base}/{p}/oof_predictions.csv')[['PatientID','risk_score']].rename(columns={'risk_score':k})
    m=m.merge(d,on='PatientID',how='inner')
m=m.merge(rsf,on='PatientID',how='inner')
print(f"aligned n={len(m)}")

cols=['A','B','C','RSF']
print("\nSpearman correlation matrix (final members):")
print(m[cols].corr(method='spearman').round(3))

e=m['Relapse'].astype(bool); t=m['RFS']
print("\nEach member's individual OOF C-index:")
for c in cols:
    print(f"  {c:4s}: {concordance_index_censored(e,t,m[c])[0]:.4f}")
