import pandas as pd
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored

base='/nvme/jin/HECKTOR26/task2_3_prediction/results/predictions'
lab=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','Relapse','RFS']]
rsf=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','rf_rfs_risk']].rename(columns={'rf_rfs_risk':'RSF'})

srcs={
 'DL_best':'multitask/clinical+CT+PET+hard_T+hard_N',
 'DL_t3cpp':'task3/clinical+prob_T+prob_N',
 'DL_ctpet_all':'task3/CT+PET+prob_T+prob_N+hard_T+hard_N',
 'DL_t3ccp':'task3/clinical+CT+PET',
}
m=lab.copy()
for k,p in srcs.items():
    d=pd.read_csv(f'{base}/{p}/oof_predictions.csv')[['PatientID','risk_score']].rename(columns={'risk_score':k})
    m=m.merge(d,on='PatientID',how='inner')
m=m.merge(rsf,on='PatientID',how='inner')
print(f"aligned n={len(m)}")

cols=list(srcs)+['RSF']
print("\nSpearman corr:")
print(m[cols].corr(method='spearman').round(3))

e=m['Relapse'].astype(bool); t=m['RFS']
print("\nsingle OOF on aligned set:")
for c in cols:
    print(f"  {c:14s} {concordance_index_censored(e,t,m[c])[0]:.4f}")
