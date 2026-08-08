import pandas as pd, numpy as np
base='/nvme/jin/HECKTOR26/task2_3_prediction/results/predictions'
out='/nvme/jin/HECKTOR26/task2_3_prediction/hecktor_docker/model/radiomics'

# OOF risk of the training set (non-CHUV) as a reference for the quantile table
srcs={'A':'multitask/clinical+CT+PET+hard_T+hard_N',
      'B':'task3/clinical+CT+PET',
      'C':'task3/CT+PET+prob_T+prob_N+hard_T+hard_N'}
for k,p in srcs.items():
    d=pd.read_csv(f'{base}/{p}/oof_predictions.csv')
    arr=np.sort(d['risk_score'].values.astype(np.float64))
    np.save(f'{out}/quantile_{k}.npy', arr)
    print(f"{k}: n={len(arr)}, range=[{arr[0]:.3f}, {arr[-1]:.3f}]")

# RSF quantile table
rsf=pd.read_csv('features/oof_clinical_rfs_v2.csv')
arr=np.sort(rsf['rf_rfs_risk'].values.astype(np.float64))
np.save(f'{out}/quantile_RSF.npy', arr)
print(f"RSF: n={len(arr)}, range=[{arr[0]:.3f}, {arr[-1]:.3f}]")
print("\n分位表已存入 model/radiomics/")
