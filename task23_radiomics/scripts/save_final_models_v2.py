import pandas as pd
import numpy as np
import warnings, joblib, json, os
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

FEAT_DIR = '/nvme/jin/HECKTOR26/codes/task23_radiomics/features/'
PRED_DIR = '/nvme/jin/HECKTOR26/task2_3_prediction/results/predictions/'
RES_DIR  = '/nvme/jin/HECKTOR26/codes/task23_radiomics/resources/'
SEEDS    = [12345, 0, 1, 2, 3]

df_train = pd.read_csv(f'{FEAT_DIR}train_predicted_mask_features_v2.csv')
df_labels = pd.read_csv(f'{FEAT_DIR}train_dataset.csv')[
    ['PatientID', 'Relapse', 'RFS', 'Age', 'Gender_enc', 'HPV_enc']
]
df_train = df_train.merge(df_labels, on='PatientID', how='left')
df = df_train.dropna(subset=['Relapse', 'RFS']).copy()
print(f'训练population: {len(df)}')

def make_y(d):
    return np.array([(bool(r), float(t)) for r, t in zip(d['Relapse'], d['RFS'])],
                     dtype=[('event', bool), ('time', float)])

y_train = make_y(df)

# 1. Clinical RSF (CT+PET + Age/Gender/HPV) 
print('\n[1/4] training Clinical RSF (CT+PET+Clinical, 433 features)...')
radio_cols = sorted([c for c in df.columns if c.startswith('CT_') or c.startswith('PET_')])
clinical_cols = ['Age', 'Gender_enc', 'HPV_enc']
clinical_rfs_feats = radio_cols + clinical_cols

X_train = df[clinical_rfs_feats].fillna(0).values
scaler_rfs = StandardScaler().fit(X_train)
X_train_s  = scaler_rfs.transform(X_train)

models_rfs = []
for seed in SEEDS:
    rsf = RandomSurvivalForest(
        n_estimators=300, min_samples_leaf=10,
        max_features='sqrt', random_state=seed, n_jobs=-1
    )
    rsf.fit(X_train_s, y_train)
    models_rfs.append(rsf)
    print(f'  seed={seed} done')

joblib.dump(models_rfs,  f'{RES_DIR}clinical_rfs_model_v2_ensemble.joblib')
joblib.dump(scaler_rfs,  f'{RES_DIR}clinical_rfs_scaler_v2.joblib')
with open(f'{RES_DIR}clinical_rfs_features_v2.json', 'w') as f:
    json.dump(clinical_rfs_feats, f)
print(f'[OK] Clinical RSF is saved ({len(clinical_rfs_feats)} features)')

# 2. OOF risk scores (LOCO CV) for Cox ensemble training 
print('\n[2/4] Calculate Clinical RSF OOF risk scores (LOCO CV)...')
from sklearn.preprocessing import StandardScaler as SS
df_labels_full = pd.read_csv(f'{FEAT_DIR}train_dataset.csv')[['PatientID', 'CenterID']]
df = df.merge(df_labels_full, on='PatientID', how='left')
centers = df['CenterID'].values
unique_centers = sorted(pd.unique(centers))

loco_risk = np.zeros(len(df))
for c in unique_centers:
    tr_idx = centers != c
    te_idx = centers == c
    if te_idx.sum() < 2:
        continue
    X_tr = df.loc[tr_idx, clinical_rfs_feats].fillna(0).values
    X_te = df.loc[te_idx, clinical_rfs_feats].fillna(0).values
    sc   = SS().fit(X_tr)
    fold_risks = []
    for seed in SEEDS:
        rsf = RandomSurvivalForest(
            n_estimators=300, min_samples_leaf=10,
            max_features='sqrt', random_state=seed, n_jobs=-1
        )
        rsf.fit(sc.transform(X_tr), y_train[tr_idx])
        fold_risks.append(rsf.predict(sc.transform(X_te)))
    loco_risk[te_idx] = np.mean(fold_risks, axis=0)
    print(f'  center {c} done')

oof_clinical_rfs = pd.DataFrame({
    'PatientID':  df['PatientID'].values,
    'rf_rfs_risk': loco_risk,
    'Relapse':    df['Relapse'].values,
    'RFS':        df['RFS'].values,
})
oof_clinical_rfs.to_csv(f'{FEAT_DIR}oof_clinical_rfs_v2.csv', index=False)
print(f'[OK] OOF risk scores is saved: oof_clinical_rfs_v2.csv')

# 3. Docker 1: Cox ensemble (multitask/CT+PET+hTN, alpha=0.001) 
print('\n[3/4] Train: Docker 1 Cox ensemble (multitask/CT+PET+hTN, alpha=0.1)...')
dl_cfg1  = 'multitask/CT+PET+prob_T+prob_N+hard_T+hard_N'
dl_oof1  = pd.read_csv(f'{PRED_DIR}{dl_cfg1}/oof_predictions.csv').rename(
             columns={'risk_score': 'dl_risk'})

mr1 = oof_clinical_rfs.merge(dl_oof1[['PatientID', 'dl_risk']], on='PatientID', how='inner')
mr1 = mr1.dropna(subset=['Relapse', 'RFS', 'dl_risk']).copy()
yr1 = make_y(mr1)
print(f'  Training set (intersection) has: {len(mr1)} cases')

mms1 = MinMaxScaler()
Xr1  = mms1.fit_transform(mr1[['rf_rfs_risk', 'dl_risk']].values)
cox1 = CoxPHSurvivalAnalysis(alpha=0.1)
cox1.fit(Xr1, yr1)
print(f'  coefficients: RF={cox1.coef_[0]:.4f}, DL={cox1.coef_[1]:.4f}')

joblib.dump(cox1, f'{RES_DIR}cox_docker1.joblib')
joblib.dump(mms1, f'{RES_DIR}cox_minmax_docker1.joblib')
with open(f'{RES_DIR}cox_docker1_info.json', 'w') as f:
    json.dump({'dl_config': dl_cfg1, 'alpha': 0.1,
               'coef_rf': cox1.coef_[0], 'coef_dl': cox1.coef_[1]}, f)
print('[OK] Docker 1 Cox ensemble is saved')

# 4. Docker 2: Cox ensemble (multitask/Clin+CT+PET+hTN, alpha=1.0) 
print('\n[4/4] Train Docker 2 Cox ensemble (multitask/Clin+CT+PET+hTN, alpha=1.0)...')
dl_cfg2  = 'multitask/clinical+CT+PET+prob_T+prob_N+hard_T+hard_N'
dl_oof2  = pd.read_csv(f'{PRED_DIR}{dl_cfg2}/oof_predictions.csv').rename(
             columns={'risk_score': 'dl_risk'})

mr2 = oof_clinical_rfs.merge(dl_oof2[['PatientID', 'dl_risk']], on='PatientID', how='inner')
mr2 = mr2.dropna(subset=['Relapse', 'RFS', 'dl_risk']).copy()
yr2 = make_y(mr2)
print(f'  Training set (intersection) has: {len(mr2)} cases')

mms2 = MinMaxScaler()
Xr2  = mms2.fit_transform(mr2[['rf_rfs_risk', 'dl_risk']].values)
cox2 = CoxPHSurvivalAnalysis(alpha=1.0)
cox2.fit(Xr2, yr2)
print(f'  coefficients: RF={cox2.coef_[0]:.4f}, DL={cox2.coef_[1]:.4f}')

joblib.dump(cox2, f'{RES_DIR}cox_docker2.joblib')
joblib.dump(mms2, f'{RES_DIR}cox_minmax_docker2.joblib')
with open(f'{RES_DIR}cox_docker2_info.json', 'w') as f:
    json.dump({'dl_config': dl_cfg2, 'alpha': 1.0,
               'coef_rf': cox2.coef_[0], 'coef_dl': cox2.coef_[1]}, f)
print('[OK] Docker 2 Cox ensemble is saved')

print('\n=== 全部完成，保存的文件列表 ===')
files = [
    'clinical_rfs_model_v2_ensemble.joblib',
    'clinical_rfs_scaler_v2.joblib',
    'clinical_rfs_features_v2.json',
    'cox_docker1.joblib',
    'cox_minmax_docker1.joblib',
    'cox_docker1_info.json',
    'cox_docker2.joblib',
    'cox_minmax_docker2.joblib',
    'cox_docker2_info.json',
]
for fname in files:
    path = f'{RES_DIR}{fname}'
    size = os.path.getsize(path) / 1024 / 1024
    print(f'  {fname}: {size:.1f} MB')
