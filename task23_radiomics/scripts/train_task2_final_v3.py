import pandas as pd
import numpy as np
import json, joblib, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

FEAT_DIR = '/nvme/jin/HECKTOR26/codes/task23_radiomics/features/'
RES_DIR  = '/nvme/jin/HECKTOR26/codes/task23_radiomics/resources/'
SEEDS = [12345, 0, 1, 2, 3]

df_train_full = pd.read_csv(f'{FEAT_DIR}train_predicted_mask_features_v2.csv')
df_labels = pd.read_csv(f'{FEAT_DIR}train_dataset.csv')[
    ['PatientID', 'T_label', 'N_label', 'Age', 'Gender_enc', 'HPV_enc']
]
df_train_full = df_train_full.merge(df_labels, on='PatientID', how='left')

df_test_gt = pd.read_csv(f'{FEAT_DIR}test_dataset.csv')
df_chuv = pd.read_csv(f'{FEAT_DIR}chuv_predicted_mask_features_v2.csv')
clin_cols = ['PatientID', 'T_label', 'N_label', 'Age', 'Gender_enc', 'HPV_enc']
df_chuv = df_chuv.merge(df_test_gt[clin_cols], on='PatientID', how='inner')

for df in [df_train_full, df_chuv]:
    df['GTVp_TLG_pynorm'] = (df['PET_GTVp_shape_VoxelVolume'] * df['PET_GTVp_firstorder_Mean']).fillna(0)
    df['GTVn_TLG_pynorm'] = (df['PET_GTVn_shape_VoxelVolume'] * df['PET_GTVn_firstorder_Mean']).fillna(0)

# Fixed column order: sorted, maintaining the same processing method as Task3.
radio_cols = sorted([c for c in df_train_full.columns if c.startswith('CT_') or c.startswith('PET_')])
shape_cols = sorted([c for c in radio_cols if 'shape' in c])
clinical_cols = ['Age', 'Gender_enc', 'HPV_enc']
tlg_cols = ['GTVp_TLG_pynorm', 'GTVn_TLG_pynorm']
empty_cols = sorted([c for c in ['CT_GTVp_empty', 'CT_GTVn_empty'] if c in df_train_full.columns])

FINAL_FEATURES = {
    't_stage': shape_cols,
    'n_stage': shape_cols + clinical_cols + tlg_cols + empty_cols,
}

for model_name, target_col in [('t_stage', 'T_label'), ('n_stage', 'N_label')]:
    feats = FINAL_FEATURES[model_name]
    df = df_train_full.dropna(subset=[target_col]).copy()
    y_train = df[target_col].values.astype(int)
    y_test  = df_chuv[target_col].values.astype(int)
    classes = np.array(sorted(np.unique(y_train)))

    X_train = df[feats].fillna(0).values
    X_test  = df_chuv[feats].fillna(0).values
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    probas = []
    ba_per_seed = []
    models = []
    for seed in SEEDS:
        model = RandomForestClassifier(
            n_estimators=500, max_depth=6, class_weight='balanced',
            random_state=seed, n_jobs=-1
        )
        model.fit(X_train_s, y_train)
        p = np.zeros((X_test_s.shape[0], len(classes)))
        for i, c in enumerate(model.classes_):
            p[:, list(classes).index(c)] = model.predict_proba(X_test_s)[:, i]
        probas.append(p)
        pred_seed = classes[np.argmax(p, axis=1)]
        ba_seed = balanced_accuracy_score(y_test, pred_seed)
        ba_per_seed.append(ba_seed)
        models.append(model)
        print(f'  {model_name} seed={seed}: Test BA = {ba_seed:.4f}')

    avg_proba = np.mean(probas, axis=0)
    pred_avg = classes[np.argmax(avg_proba, axis=1)]
    ba_avg = balanced_accuracy_score(y_test, pred_avg)
    cm = confusion_matrix(y_test, pred_avg, labels=classes)

    print(f'{model_name}: training population={len(df)}, feature number ={len(feats)}')
    print(f'  Single seed range: {min(ba_per_seed):.4f} ~ {max(ba_per_seed):.4f}')
    print(f'  After averaging the probabilities of multiple seeds Test BA = {ba_avg:.4f}  (A more robust final number or not?)')
    print(f'  Confusion matrix (rows = true, columns = predicted), classes={list(classes)}:')
    print(cm)
    print()

    joblib.dump(models, f'{RES_DIR}{model_name}_model_v2_ensemble.joblib')
    joblib.dump(scaler, f'{RES_DIR}{model_name}_scaler_v2.joblib')
    with open(f'{RES_DIR}{model_name}_features_v2.json', 'w') as f:
        json.dump(feats, f)
    print(f'  [OK] saved A list of models with {len(SEEDS)}seeds; during inference, the argmax is calculated by averaging the probabilities.')
    print()
