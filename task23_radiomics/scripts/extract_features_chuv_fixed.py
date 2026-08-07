import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor
import logging
import warnings
import os
warnings.filterwarnings('ignore')
logging.getLogger('radiomics').setLevel(logging.ERROR)

DATA_DIR = '/nvme/jin/HECKTOR26/data/organized/'
PRED_DIR = '/nvme/jin/HECKTOR26/predictions/ensemble_5fold/'
SUV_CSV  = '/nvme/jin/HECKTOR26/data/suv_conversion_tags.csv'
OUTPUT   = '/nvme/jin/HECKTOR26/codes/task23_radiomics/features/chuv_predicted_mask_features_v2.csv'

suv_df = pd.read_csv(SUV_CSV)
SUV_FACTORS = dict(zip(suv_df['pid'], suv_df['bqml_to_suvbw_factor']))

chuv_ids = sorted(set([f.split('.')[0] for f in os.listdir(PRED_DIR) if f.endswith('.nii.gz')]))
print(f'CHUV患者数: {len(chuv_ids)}')

settings = {
    'binWidth': 25,
    'resampledPixelSpacing': [1, 1, 1],
    'normalize': True,
    'normalizeScale': 100,
}
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableFeatureClassByName('shape')
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')

def extract_one(pid):
    try:
        ct_path = f'{DATA_DIR}{pid}/{pid}__CT.nii.gz'
        pt_path = f'{DATA_DIR}{pid}/{pid}__PT.nii.gz'
        pred_mask_path = f'{PRED_DIR}{pid}.nii.gz'

        ct = sitk.ReadImage(ct_path)
        pt = sitk.ReadImage(pt_path)
        pred_mask = sitk.ReadImage(pred_mask_path)

        if pid in SUV_FACTORS:
            pt_arr = sitk.GetArrayFromImage(pt) * SUV_FACTORS[pid]
            pt_corrected = sitk.GetImageFromArray(pt_arr)
            pt_corrected.CopyInformation(pt)
            pt = pt_corrected

        resampler_mask = sitk.ResampleImageFilter()
        resampler_mask.SetReferenceImage(ct)
        resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler_mask.SetDefaultPixelValue(0)
        pred_mask = resampler_mask.Execute(pred_mask)

        resampler_pt = sitk.ResampleImageFilter()
        resampler_pt.SetReferenceImage(ct)
        resampler_pt.SetInterpolator(sitk.sitkLinear)
        resampler_pt.SetDefaultPixelValue(0.0)
        pt = resampler_pt.Execute(pt)

        result = {'PatientID': pid}

        for label, region in [(1, 'GTVp'), (2, 'GTVn')]:
            mask_arr = sitk.GetArrayFromImage(pred_mask)
            n_voxels = (mask_arr == label).sum()

            if n_voxels == 0:
                result[f'CT_{region}_empty'] = 1
                continue
            result[f'CT_{region}_empty'] = 0

            mask_label = sitk.BinaryThreshold(pred_mask, label, label, 1, 0)

            try:
                ct_features = extractor.execute(ct, mask_label, label=1)
                for k, v in ct_features.items():
                    if 'diagnostics' not in k:
                        result[f'CT_{region}_{k.split("_",1)[1] if "_" in k else k}'] = float(v)
            except Exception:
                pass

            try:
                pt_features = extractor.execute(pt, mask_label, label=1)
                for k, v in pt_features.items():
                    if 'diagnostics' not in k:
                        result[f'PET_{region}_{k.split("_",1)[1] if "_" in k else k}'] = float(v)
            except Exception:
                pass

            pt_arr_full = sitk.GetArrayFromImage(pt)
            mask_arr_full = sitk.GetArrayFromImage(mask_label)
            spacing = pt.GetSpacing()
            voxel_vol = spacing[0]*spacing[1]*spacing[2]
            suv_vals = pt_arr_full[mask_arr_full == 1]
            if len(suv_vals) > 0:
                result[f'{region}_SUVmean'] = float(suv_vals.mean())
                result[f'{region}_SUVmax']  = float(suv_vals.max())
                result[f'{region}_TLG']     = float(suv_vals.mean() * len(suv_vals) * voxel_vol)
            else:
                result[f'{region}_SUVmean'] = 0.0
                result[f'{region}_SUVmax']  = 0.0
                result[f'{region}_TLG']     = 0.0

        return result
    except Exception as e:
        print(f'[ERROR] {pid}: {e}')
        return {'PatientID': pid, 'error': str(e)}

from multiprocessing import Pool
print('开始提取(预测mask, CHUV, 修复版)...')
with Pool(processes=10) as pool:
    results = pool.map(extract_one, chuv_ids)

df_pred = pd.DataFrame(results)

for region in ['GTVp', 'GTVn']:
    empty_col = f'CT_{region}_empty'
    if empty_col not in df_pred.columns:
        continue
    empty_rows = df_pred[empty_col] == 1
    cols_to_zero = [c for c in df_pred.columns
                    if c.startswith(f'CT_{region}_') or c.startswith(f'PET_{region}_')]
    cols_to_zero += [f'{region}_SUVmean', f'{region}_SUVmax', f'{region}_TLG']
    cols_to_zero = [c for c in cols_to_zero if c in df_pred.columns and c != empty_col]
    df_pred.loc[empty_rows, cols_to_zero] = 0.0
    print(f'{region}: {int(empty_rows.sum())} 个空mask患者, 已对 {len(cols_to_zero)} 个特征列填0')

remaining_nan = df_pred.drop(columns=['error'], errors='ignore').isna().sum().sum()
print(f'[CHECK] 修复后剩余NaN总数: {remaining_nan}')

print(f'完成: {df_pred.shape}')
df_pred.to_csv(OUTPUT, index=False)
print('[OK] 已保存到', OUTPUT)
