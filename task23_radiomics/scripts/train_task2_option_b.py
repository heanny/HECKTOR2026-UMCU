import pandas as pd
import numpy as np
import warnings, joblib, json, os
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression

FEAT_DIR = '/nvme/jin/HECKTOR26/codes/task23_radiomics/features/'
PRED_DIR = '/nvme/jin/HECKTOR26/task2_3_prediction/results/predictions/task2/'
RES_DIR  = '/nvme/jin/HECKTOR26/codes/task23_radiomics/resources/'

DL_T_CLASSES = [1, 2, 3, 4]
DL_N_CLASSES = [0, 1, 2, 3]
DL_CONFIG = 'clinical+CT+PET+hard_T+hard_N'
C_VALUE = 1.0
SEED = 12345

# RF OOF (the LOCO OOF we generated previously)
oof_t = pd.read_csv(f'{FEAT_DIR}oof_rf_t_stage_v2.csv')
oof_n = pd.read_csv(f'{FEAT_DIR}oof_rf_n_stage_v2.csv')

df_labels = pd.read_csv(f'{FEAT_DIR}train_dataset.csv')[['PatientID','T_label','N_label']]

t_rf_prob_cols = [f'rf_t_stage_prob_{c}' for c in [0,1,2,3,4]]
n_rf_prob_cols = [f'rf_n_stage_prob_{c}' for c in [0,1,2,3]]

# DL OOF (fixed, clinical+CT+PET+hTN)
dl_oof = pd.read_csv(f'{PRED_DIR}{DL_CONFIG}/oof_predictions.csv')
t_dl_cols = [f'T_prob_T{c}' for c in DL_T_CLASSES]
n_dl_cols = [f'N_prob_N{c}' for c in DL_N_CLASSES]

# T-stacking training (T0 excluded, DL has no T0 category)
mt = oof_t.merge(df_labels[['PatientID','T_label']], on='PatientID', how='inner')
mt = mt.merge(dl_oof[['PatientID']+t_dl_cols], on='PatientID', how='inner')
mt = mt.dropna(subset=['T_label']+t_dl_cols)
mt = mt[mt['T_label'].isin(DL_T_CLASSES)].copy()
Xt = mt[t_rf_prob_cols + t_dl_cols].values
yt = mt['T_label'].values.astype(int)
print(f'T staging stacking training sample number: {len(mt)}')

lr_t = LogisticRegression(C=C_VALUE, max_iter=1000, random_state=SEED, class_weight='balanced')
lr_t.fit(Xt, yt)

# N stacking训练
mn = oof_n.merge(df_labels[['PatientID','N_label']], on='PatientID', how='inner')
mn = mn.merge(dl_oof[['PatientID']+n_dl_cols], on='PatientID', how='inner')
mn = mn.dropna(subset=['N_label']+n_dl_cols).copy()
Xn = mn[n_rf_prob_cols + n_dl_cols].values
yn = mn['N_label'].values.astype(int)
print(f'N staging stacking training sample number: {len(mn)}')

lr_n = LogisticRegression(C=C_VALUE, max_iter=1000, random_state=SEED, class_weight='balanced')
lr_n.fit(Xn, yn)

# save
joblib.dump(lr_t, f'{RES_DIR}task2_optionB_stack_t_model.joblib')
joblib.dump(lr_n, f'{RES_DIR}task2_optionB_stack_n_model.joblib')
with open(f'{RES_DIR}task2_optionB_info.json', 'w') as f:
    json.dump({
        'dl_config': DL_CONFIG, 'C': C_VALUE,
        't_rf_prob_cols': t_rf_prob_cols, 't_dl_cols': t_dl_cols,
        'n_rf_prob_cols': n_rf_prob_cols, 'n_dl_cols': n_dl_cols,
    }, f, indent=2)
print('已保存: task2_optionB_stack_t_model.joblib / stack_n_model.joblib')

# Quickly verify the numbers on CHUV to see if they are correct.
df_test_gt = pd.read_csv(f'{FEAT_DIR}test_dataset.csv')
df_chuv_feat = pd.read_csv(f'{FEAT_DIR}chuv_predicted_mask_features_v2.csv')
_clin = df_test_gt[['PatientID','Age','Gender_enc','HPV_enc']]
df_chuv_feat = df_chuv_feat.merge(_clin, on='PatientID', how='left')
df_chuv_feat['GTVp_TLG_pynorm'] = (df_chuv_feat['PET_GTVp_shape_VoxelVolume']*df_chuv_feat['PET_GTVp_firstorder_Mean']).fillna(0)
df_chuv_feat['GTVn_TLG_pynorm'] = (df_chuv_feat['PET_GTVn_shape_VoxelVolume']*df_chuv_feat['PET_GTVn_firstorder_Mean']).fillna(0)

models_t = joblib.load(f'{RES_DIR}t_stage_model_v2_ensemble.joblib')
scaler_t = joblib.load(f'{RES_DIR}t_stage_scaler_v2.joblib')
with open(f'{RES_DIR}t_stage_features_v2.json') as f: t_feats = json.load(f)
t_classes_rf = list(models_t[0].classes_)
X_t = df_chuv_feat[t_feats].fillna(0).values
chuv_rf_t_proba = np.mean([m.predict_proba(scaler_t.transform(X_t)) for m in models_t], axis=0)

models_n = joblib.load(f'{RES_DIR}n_stage_model_v2_ensemble.joblib')
scaler_n = joblib.load(f'{RES_DIR}n_stage_scaler_v2.joblib')
with open(f'{RES_DIR}n_stage_features_v2.json') as f: n_feats = json.load(f)
n_classes_rf = list(models_n[0].classes_)
X_n = df_chuv_feat[n_feats].fillna(0).values
chuv_rf_n_proba = np.mean([m.predict_proba(scaler_n.transform(X_n)) for m in models_n], axis=0)

dl_test = pd.read_csv(f'{PRED_DIR}{DL_CONFIG}/test_predictions.csv')
df_chuv_lbl = df_test_gt[['PatientID','T_label','N_label']].dropna()

t_dl_test = dl_test.merge(df_chuv_lbl[['PatientID']], on='PatientID')[t_dl_cols].values
n_dl_test = dl_test.merge(df_chuv_lbl[['PatientID']], on='PatientID')[n_dl_cols].values

Xt_test = np.hstack([chuv_rf_t_proba, t_dl_test])
Xn_test = np.hstack([chuv_rf_n_proba, n_dl_test])

from sklearn.metrics import balanced_accuracy_score
t_pred = lr_t.predict(Xt_test)
n_pred = lr_n.predict(Xn_test)
t_ba = balanced_accuracy_score(df_chuv_lbl['T_label'].astype(int), t_pred)
n_ba = balanced_accuracy_score(df_chuv_lbl['N_label'].astype(int), n_pred)
print(f'\n Verify on CHUV: T BA={t_ba:.4f}, N BA={n_ba:.4f}, Mean={((t_ba+n_ba)/2):.4f}')
print(f'Expected (numbers in the table): T=0.514, N=0.726, Mean=0.620')
