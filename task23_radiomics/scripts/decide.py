import pandas as pd, numpy as np
from sksurv.metrics import concordance_index_censored

base='/nvme/jin/HECKTOR26/task2_3_prediction/results/predictions'
lab=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','Relapse','RFS']]
rsf=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','rf_rfs_risk']].rename(columns={'rf_rfs_risk':'RSF'})
srcs={'DL_best':'multitask/clinical+CT+PET+hard_T+hard_N',
      'DL_t3ccp':'task3/clinical+CT+PET',
      'DL_ctpet_all':'task3/CT+PET+prob_T+prob_N+hard_T+hard_N'}
m=lab.copy()
for k,p in srcs.items():
    d=pd.read_csv(f'{base}/{p}/oof_predictions.csv')[['PatientID','risk_score']].rename(columns={'risk_score':k})
    m=m.merge(d,on='PatientID',how='inner')
m=m.merge(rsf,on='PatientID',how='inner')
m['center']=m['PatientID'].str.split('-').str[0]

# 秩归一化在每个留出fold外单独做, 避免全局秩泄漏
def eval_scheme(cols):
    """逐中心: 秩在'其余中心'上fit再map到留出中心, 收集全局oof risk"""
    oof=np.full(len(m),np.nan)
    for held in m['center'].unique():
        tr=m['center']!=held; te=m['center']==held
        risk=np.zeros(te.sum())
        for c in cols:
            # 用训练中心的分布把留出中心的值转成分位
            ref=m.loc[tr,c].values
            val=m.loc[te,c].values
            q=np.searchsorted(np.sort(ref),val)/len(ref)
            risk+=q
        oof[te.values]=risk/len(cols)
    e=m['Relapse'].astype(bool); t=m['RFS']
    glob=concordance_index_censored(e,t,oof)[0]
    # 逐中心
    per={}
    mm=m.copy(); mm['r']=oof
    for ct,g in mm.groupby('center'):
        if g['Relapse'].sum()>=5:
            per[ct]=concordance_index_censored(g['Relapse'].astype(bool),g['RFS'],g['r'])[0]
    return glob,per

schemes={
 'current_best (ref 0.6784)':None,
 '2src A+B':['DL_best','DL_t3ccp'],
 '3src A+B+C':['DL_best','DL_t3ccp','DL_ctpet_all'],
 '3src+RSF':['DL_best','DL_t3ccp','DL_ctpet_all','RSF'],
 'DL_best alone':['DL_best'],
}
for name,cols in schemes.items():
    if cols is None: 
        print(f"{name}"); continue
    g,per=eval_scheme(cols)
    big=[f"{k}:{v:.3f}" for k,v in per.items() if k in('MDA','CHUS','HGJ','CHUP')]
    print(f"{g:.4f}  {name:30s} [MDA/CHUS/HGJ/CHUP: {' '.join(big)}]")
