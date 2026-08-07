# Build v2 (resubmit) - Task2=Option B stacking(RF+DL clin_hTN), Task3=Docker1 original(alpha=0.1)
# Docker 2: multitask/Clin+CT+PET+hTN Cox ensemble, alpha=1.0
"""
HECKTOR 2026 inference pipeline (Task 1 + 2 + 3).

Pipeline:
  CT + PET + ehr.json
    -> Stage A: resample to 1mm + PET-centred 200x200x310 crop
    -> Stage 1: STU-Net 5-fold ensemble -> segmentation -> output.mha
    -> Stage 2: DenseNet multitask 5-fold -> DL risk score (Task3 only)
    -> Stage 3: PyRadiomics feature extraction on original CT grid
    -> Stage 4: RF ensemble -> T/N stage (Task2)
    -> Stage 5: Clinical RSF + Cox ensemble -> RFS (Task3)
"""

import json
import os
import re
import tempfile
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

warnings.filterwarnings('ignore')

os.environ.setdefault("nnUNet_raw",          "/tmp/nnunet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet_preprocessed")
os.environ.setdefault("nnUNet_results",      "/tmp/nnunet_results")
for _p in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    Path(os.environ[_p]).mkdir(parents=True, exist_ok=True)

from constants       import IN_CHANNELS, N_STAGING_CLINICAL, N_RFS_CLINICAL, TEST_CENTER
from clinical_utils  import prepare_staging_clinical, prepare_rfs_clinical, ehr_to_dataframe
from image_preprocess import resample_and_crop, build_densenet_volume, sitk_to_xyz
from models          import DenseNet121MultiTask, DenseNet121Task3

INPUT_PATH  = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH  = Path("/opt/ml/model")

SEG_MODEL_DIR = MODEL_PATH / "seg" / "STUNetTrainer_small__nnUNetPlans__3d_fullres"
MT_CKPT_DIR   = MODEL_PATH / "multitask"
B_CKPT_DIR    = MODEL_PATH / "task3_B"
C_CKPT_DIR    = MODEL_PATH / "task3_C"
TRAIN_CSV     = MODEL_PATH / "clinical" / "HECKTOR_2026_training_data.csv"
RADIO_DIR     = MODEL_PATH / "radiomics"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# T/N stage label maps (matching training labels)
T_MAP = {0: "T1", 1: "T2", 2: "T3", 3: "T4", 4: "T4"}  # T0->T4 fallback
N_MAP = {0: "N0", 1: "N1", 2: "N2", 3: "N3"}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run():
    ct_path  = get_image_file(INPUT_PATH / "images/ct")
    pet_path = get_image_file(INPUT_PATH / "images/pet")
    ehr      = load_json(INPUT_PATH / "ehr.json")
    print(f"[io] CT={ct_path}  PET={pet_path}", flush=True)

    # Stage A: resample + crop
    ct_crop, pet_crop = resample_and_crop(ct_path, pet_path)

    # Stage 1: segmentation
    seg_crop, prob_t_crop, prob_n_crop = run_segmentation(ct_crop, pet_crop)
    write_segmentation(
        OUTPUT_PATH / "images/head-neck-tumor-segmentation",
        seg_crop, ct_path,
    )

    # Stage 2: DL risk score (needed for Task 3 Cox ensemble)
    dl_risk, dl_t_proba, dl_n_proba = run_dl_risk(ct_crop, pet_crop, seg_crop, ehr)
    print(f"[stage2] DL risk = {dl_risk:.4f}", flush=True)
    print(f"[stage2] DL T proba = {dl_t_proba}", flush=True)
    print(f"[stage2] DL N proba = {dl_n_proba}", flush=True)

    # Stage 2b: B, C risk (for Task3 four-source fusion)
    risk_B, risk_C = run_dl_risk_BC(ct_crop, pet_crop, seg_crop, prob_t_crop, prob_n_crop, ehr)

    # Stage 3: PyRadiomics feature extraction (shared by Stage 4 and 5)
    feat_dict = extract_radiomics(ct_path, pet_path, seg_crop)

    # Stage 4: RF+DL stacking staging (Task 2, Option B)
    t_stage, n_stage = run_rf_staging(feat_dict, ehr, dl_t_proba, dl_n_proba)
    print(f"[stage4] T={t_stage}  N={n_stage}", flush=True)

    # Stage 5: Clinical RSF + Cox ensemble (Task 3)
    rfs_value = run_rfs(feat_dict, ehr, dl_risk, risk_B, risk_C)
    print(f"[stage5] RFS = {rfs_value:.4f}", flush=True)

    write_json(OUTPUT_PATH / "t-stage.json", t_stage)
    write_json(OUTPUT_PATH / "n-stage.json", n_stage)
    write_json(OUTPUT_PATH / "rfs.json",     float(rfs_value))
    print(f"[done] T={t_stage}  N={n_stage}  RFS={rfs_value:.4f}", flush=True)
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: STU-Net segmentation
# ═══════════════════════════════════════════════════════════════════════════

_SEG_PREDICTOR = None


def _build_seg_predictor():
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from batchgenerators.utilities.file_and_folder_operations import (
        load_json as nn_load_json,
    )
    from STUNetTrainer import STUNetTrainer_small

    plans        = nn_load_json(str(SEG_MODEL_DIR / "plans.json"))
    dataset_json = nn_load_json(str(SEG_MODEL_DIR / "dataset.json"))
    plans_manager  = PlansManager(plans)
    config_manager = plans_manager.get_configuration("3d_fullres")
    n_in = len(dataset_json["channel_names"])

    network = STUNetTrainer_small.build_network_architecture(
        plans_manager, dataset_json, config_manager,
        n_in, enable_deep_supervision=False,
    )

    parameters, mirror_axes = [], None
    for f in range(5):
        ckpt   = torch.load(
            SEG_MODEL_DIR / f"fold_{f}" / "checkpoint_final.pth",
            map_location="cpu", weights_only=False,
        )
        weights = ckpt.get("network_weights", ckpt.get("state_dict"))
        parameters.append(weights)
        if mirror_axes is None:
            mirror_axes = ckpt.get("inference_allowed_mirroring_axes", (0, 1, 2))

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=(DEVICE.type == "cuda"),
        device=DEVICE, verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.manual_initialization(
        network, plans_manager, config_manager, parameters,
        dataset_json, "STUNetTrainer_small", mirror_axes,
    )
    return predictor


def run_segmentation(ct_crop, pet_crop):
    global _SEG_PREDICTOR
    if _SEG_PREDICTOR is None:
        _SEG_PREDICTOR = _build_seg_predictor()

    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    with tempfile.TemporaryDirectory() as tmp:
        tmp   = Path(tmp)
        ct_f  = str(tmp / "case_0000.nii.gz")
        pet_f = str(tmp / "case_0001.nii.gz")
        sitk.WriteImage(ct_crop,  ct_f)
        sitk.WriteImage(pet_crop, pet_f)

        io = SimpleITKIO()
        img, props = io.read_images([ct_f, pet_f])
        seg_arr, prob = _SEG_PREDICTOR.predict_single_npy_array(
            img, props, None, None, True,
        )
        out = str(tmp / "seg.nii.gz")
        io.write_seg(seg_arr, out, props)
        seg = sitk.ReadImage(out)

    seg.CopyInformation(ct_crop)
    # prob[1] = GTVp (prob_T), prob[2] = GTVn (prob_N), same grid as seg
    prob_t_img = sitk.GetImageFromArray(prob[1].astype(np.float32))
    prob_n_img = sitk.GetImageFromArray(prob[2].astype(np.float32))
    prob_t_img.CopyInformation(seg)
    prob_n_img.CopyInformation(seg)
    return seg, prob_t_img, prob_n_img


def write_segmentation(location, seg_crop, reference_path):
    location.mkdir(parents=True, exist_ok=True)
    ref = sitk.ReadImage(str(reference_path))
    seg_orig = sitk.Resample(
        seg_crop, ref, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
    )
    seg_orig.CopyInformation(ref)
    sitk.WriteImage(seg_orig, str(location / "output.mha"), useCompression=True)
    print(f"[seg] wrote output.mha  size={seg_orig.GetSize()}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: DenseNet — DL risk score only (for Task 3)
# ═══════════════════════════════════════════════════════════════════════════

_CLIN_CACHE = None


def _clinical_stats():
    global _CLIN_CACHE
    if _CLIN_CACHE is None:
        df_all   = pd.read_csv(TRAIN_CSV).set_index("PatientID")
        df_train = df_all.loc[~df_all.index.str.startswith(TEST_CENTER)]
        _, stg_stats = prepare_staging_clinical(df_train)
        _, rfs_stats = prepare_rfs_clinical(df_train)
        _CLIN_CACHE  = (stg_stats, rfs_stats)
    return _CLIN_CACHE


def _load_multitask(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = DenseNet121MultiTask(
        in_channels  = ckpt.get("in_channels",         IN_CHANNELS),
        n_stg_clc    = ckpt.get("n_staging_clinical",  N_STAGING_CLINICAL),
        n_rfs_clc    = ckpt.get("n_rfs_clinical",       N_RFS_CLINICAL),
        use_image    = ckpt.get("use_image",            True),
        use_clinical = ckpt.get("use_clinical",         True),
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def run_dl_risk(ct_crop, pet_crop, seg_crop, ehr):
    """Run DenseNet 5-fold ensemble. Returns (dl_risk, dl_t_proba, dl_n_proba).
    dl_t_proba: np.array of 4 values [T1,T2,T3,T4] softmax, averaged over folds.
    dl_n_proba: np.array of 4 values [N0,N1,N2,N3] softmax, averaged over folds.
    """
    ct_xyz  = sitk_to_xyz(ct_crop)
    pet_xyz = sitk_to_xyz(pet_crop)
    seg_xyz = sitk_to_xyz(seg_crop).astype(np.uint8)
    vol     = build_densenet_volume(ct_xyz, pet_xyz, seg_xyz)
    image   = torch.from_numpy(vol).unsqueeze(0).to(DEVICE)

    stg_stats, rfs_stats = _clinical_stats()
    df_one  = ehr_to_dataframe(ehr)
    clc_stg = torch.from_numpy(
        prepare_staging_clinical(df_one, stg_stats)[0]
    ).to(DEVICE)
    clc_rfs = torch.from_numpy(
        prepare_rfs_clinical(df_one, rfs_stats)[0]
    ).to(DEVICE)

    risks, t_probas, n_probas = [], [], []
    for fold in range(5):
        model = _load_multitask(MT_CKPT_DIR / f"fold{fold}.pt")
        with torch.no_grad():
            t_logits, n_logits, risk = model(image, clc_stg, clc_rfs)
            t_prob = torch.softmax(t_logits, dim=1).cpu().numpy()[0]  # [T1,T2,T3,T4]
            n_prob = torch.softmax(n_logits, dim=1).cpu().numpy()[0]  # [N0,N1,N2,N3]
            t_probas.append(t_prob)
            n_probas.append(n_prob)
            risks.append(float(risk.cpu().numpy().reshape(-1)[0]))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    dl_risk    = float(np.mean(risks))
    dl_t_proba = np.mean(t_probas, axis=0)
    dl_n_proba = np.mean(n_probas, axis=0)
    return dl_risk, dl_t_proba, dl_n_proba


def _load_task3(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = DenseNet121Task3(
        in_channels  = ckpt.get("in_channels",   2),
        n_clinical   = ckpt.get("n_rfs_clinical", N_RFS_CLINICAL),
        use_image    = ckpt.get("use_image",      True),
        use_clinical = ckpt.get("use_clinical",   True),
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def run_dl_risk_BC(ct_crop, pet_crop, seg_crop, prob_t_crop, prob_n_crop, ehr):
    """risk_B (task3 clinical+CT+PET, 2ch) and risk_C (task3 6ch), 5-fold avg."""
    from image_preprocess import build_densenet_volume_6ch

    ct_xyz    = sitk_to_xyz(ct_crop)
    pet_xyz   = sitk_to_xyz(pet_crop)
    seg_xyz   = sitk_to_xyz(seg_crop).astype(np.uint8)
    probt_xyz = sitk_to_xyz(prob_t_crop)
    probn_xyz = sitk_to_xyz(prob_n_crop)

    # B: 2ch [CT, PET] = first 2 channels of the 4ch volume
    vol4  = build_densenet_volume(ct_xyz, pet_xyz, seg_xyz)
    vol_B = vol4[:2]
    img_B = torch.from_numpy(vol_B).unsqueeze(0).to(DEVICE)

    # C: 6ch [CT, PET, prob_T, prob_N, hard_T, hard_N]
    vol_C = build_densenet_volume_6ch(ct_xyz, pet_xyz, seg_xyz, probt_xyz, probn_xyz)
    img_C = torch.from_numpy(vol_C).unsqueeze(0).to(DEVICE)

    # clinical (B uses rfs clinical; C ignores it but forward needs the arg)
    _, rfs_stats = _clinical_stats()
    df_one  = ehr_to_dataframe(ehr)
    clc_rfs = torch.from_numpy(prepare_rfs_clinical(df_one, rfs_stats)[0]).to(DEVICE)

    risks_B, risks_C = [], []
    for fold in range(5):
        mB = _load_task3(B_CKPT_DIR / f"fold{fold}.pt")
        with torch.no_grad():
            risks_B.append(float(mB(img_B, clc_rfs).cpu().numpy().reshape(-1)[0]))
        del mB
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        mC = _load_task3(C_CKPT_DIR / f"fold{fold}.pt")
        with torch.no_grad():
            risks_C.append(float(mC(img_C, clc_rfs).cpu().numpy().reshape(-1)[0]))
        del mC
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    risk_B = float(np.mean(risks_B))
    risk_C = float(np.mean(risks_C))
    print(f"[stage2b] risk_B={risk_B:.4f} risk_C={risk_C:.4f}", flush=True)
    return risk_B, risk_C


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: PyRadiomics feature extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_radiomics(ct_path, pet_path, seg_crop):
    """Extract PyRadiomics features. Returns feature dict.
    Empty mask regions are filled with 0.0 (matches training convention).
    """
    import logging
    from radiomics import featureextractor
    logging.getLogger('radiomics').setLevel(logging.ERROR)

    ct  = sitk.ReadImage(str(ct_path))
    pet = sitk.ReadImage(str(pet_path))

    # SUV conversion
    suv_csv = RADIO_DIR / "suv_conversion_tags.csv"
    if suv_csv.exists():
        suv_df  = pd.read_csv(str(suv_csv))
        suv_map = dict(zip(suv_df['pid'], suv_df['bqml_to_suvbw_factor']))
        pid     = Path(ct_path).stem.split('__')[0].replace('.nii', '')
        if pid in suv_map:
            arr = sitk.GetArrayFromImage(pet) * suv_map[pid]
            pet2 = sitk.GetImageFromArray(arr)
            pet2.CopyInformation(pet)
            pet  = pet2

    # Resample mask and PT to CT grid (unconditional)
    def resample_to_ct(img, interp):
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(ct)
        r.SetInterpolator(interp)
        r.SetDefaultPixelValue(0)
        return r.Execute(img)

    pred_mask = resample_to_ct(seg_crop,  sitk.sitkNearestNeighbor)
    pet       = resample_to_ct(pet,       sitk.sitkLinear)

    settings  = {
        'binWidth': 25,
        'resampledPixelSpacing': [1, 1, 1],
        'normalize': True,
        'normalizeScale': 100,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    for fc in ['shape', 'firstorder', 'glcm', 'glrlm', 'glszm']:
        extractor.enableFeatureClassByName(fc)

    mask_arr = sitk.GetArrayFromImage(pred_mask)
    result   = {}

    for label, region in [(1, 'GTVp'), (2, 'GTVn')]:
        empty = int((mask_arr == label).sum() == 0)
        result[f'CT_{region}_empty'] = empty

        if empty:
            # All features for this region are 0.0
            # (will be applied after we know the full feature list)
            result[f'{region}_SUVmean']    = 0.0
            result[f'{region}_SUVmax']     = 0.0
            result[f'{region}_TLG']        = 0.0
            result[f'{region}_TLG_pynorm'] = 0.0
            result[f'{region}_count']      = 0
            continue

        mask_label = sitk.BinaryThreshold(pred_mask, label, label, 1, 0)

        try:
            for k, v in extractor.execute(ct, mask_label, label=1).items():
                if 'diagnostics' not in k:
                    ck = k.split('_', 1)[1] if '_' in k else k
                    result[f'CT_{region}_{ck}'] = float(v)
        except Exception:
            pass

        try:
            for k, v in extractor.execute(pet, mask_label, label=1).items():
                if 'diagnostics' not in k:
                    ck = k.split('_', 1)[1] if '_' in k else k
                    result[f'PET_{region}_{ck}'] = float(v)
        except Exception:
            pass

        pt_arr    = sitk.GetArrayFromImage(pet)
        mask_reg  = sitk.GetArrayFromImage(mask_label)
        spacing   = pet.GetSpacing()
        vox_vol   = spacing[0] * spacing[1] * spacing[2]
        suv_vals  = pt_arr[mask_reg == 1]

        if len(suv_vals) > 0:
            result[f'{region}_SUVmean'] = float(suv_vals.mean())
            result[f'{region}_SUVmax']  = float(suv_vals.max())
            result[f'{region}_TLG']     = float(suv_vals.mean() * len(suv_vals) * vox_vol)
        else:
            result[f'{region}_SUVmean'] = 0.0
            result[f'{region}_SUVmax']  = 0.0
            result[f'{region}_TLG']     = 0.0

        cc = sitk.ConnectedComponentImageFilter()
        cc.Execute(mask_label)
        result[f'{region}_count'] = cc.GetObjectCount()

    # TLG pynorm
    result['GTVp_TLG_pynorm'] = (
        result.get('PET_GTVp_shape_VoxelVolume', 0.0) *
        result.get('PET_GTVp_firstorder_Mean',   0.0)
    )
    result['GTVn_TLG_pynorm'] = (
        result.get('PET_GTVn_shape_VoxelVolume', 0.0) *
        result.get('PET_GTVn_firstorder_Mean',   0.0)
    )
    return result


def _fill_empty_region(feat_dict, region, feat_list):
    """Fill all features for an empty-mask region with 0.0."""
    if feat_dict.get(f'CT_{region}_empty', 0) == 1:
        for k in feat_list:
            if f'CT_{region}_' in k or f'PET_{region}_' in k:
                feat_dict.setdefault(k, 0.0)


def _ehr_clinical(ehr):
    """Extract Age, Gender_enc, HPV_enc from ehr dict."""
    df  = ehr_to_dataframe(ehr)
    age = df['Age'].iloc[0]
    gen = df.get('Gender', pd.Series([np.nan])).iloc[0]
    hpv = df.get('HPV Status', pd.Series([np.nan])).iloc[0]
    return (
        0.0 if pd.isna(age) else float(age),
        0.0 if pd.isna(gen) else float(gen),
        0.0 if pd.isna(hpv) else float(hpv),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: RF ensemble -> T/N staging (Task 2)
# ═══════════════════════════════════════════════════════════════════════════

def run_rf_staging(feat_dict, ehr, dl_t_proba, dl_n_proba):
    """RF+DL stacking (Option B): Logistic Regression on
    [RF_prob(5 or 4 classes), DL_prob(4 classes)] -> T/N stage.
    dl_t_proba: array of 4 values [T1,T2,T3,T4] probabilities from DL
    dl_n_proba: array of 4 values [N0,N1,N2,N3] probabilities from DL
    """
    import joblib

    t_models = joblib.load(str(RADIO_DIR / "t_stage_model_v2_ensemble.joblib"))
    t_scaler = joblib.load(str(RADIO_DIR / "t_stage_scaler_v2.joblib"))
    n_models = joblib.load(str(RADIO_DIR / "n_stage_model_v2_ensemble.joblib"))
    n_scaler = joblib.load(str(RADIO_DIR / "n_stage_scaler_v2.joblib"))
    stack_t  = joblib.load(str(RADIO_DIR / "task2_optionB_stack_t_model.joblib"))
    stack_n  = joblib.load(str(RADIO_DIR / "task2_optionB_stack_n_model.joblib"))

    with open(str(RADIO_DIR / "t_stage_features_v2.json")) as f:
        t_feats = json.load(f)
    with open(str(RADIO_DIR / "n_stage_features_v2.json")) as f:
        n_feats = json.load(f)

    _fill_empty_region(feat_dict, 'GTVp', t_feats + n_feats)
    _fill_empty_region(feat_dict, 'GTVn', t_feats + n_feats)

    age, gen, hpv = _ehr_clinical(ehr)
    feat_dict['Age']        = age
    feat_dict['Gender_enc'] = gen
    feat_dict['HPV_enc']    = hpv

    # RF T proba: 5 classes [T0,T1,T2,T3,T4], DL T proba: 4 classes [T1,T2,T3,T4]
    X_t = np.array([feat_dict.get(f, 0.0) for f in t_feats], dtype=np.float32)
    X_t_sc = t_scaler.transform(X_t.reshape(1, -1))
    t_classes = list(t_models[0].classes_)
    rf_t_proba = np.mean([m.predict_proba(X_t_sc)[0] for m in t_models], axis=0)
    # verify column order matches classes_ each time (defensive check)
    assert t_classes == sorted(t_classes), f"T classes not sorted: {t_classes}"

    stack_input_t = np.hstack([rf_t_proba, dl_t_proba]).reshape(1, -1)
    t_pred = int(stack_t.predict(stack_input_t)[0])
    T_LABEL_MAP = {1: "T1", 2: "T2", 3: "T3", 4: "T4"}
    t_stage = T_LABEL_MAP.get(t_pred, "T2")

    # RF N proba: 4 classes [N0,N1,N2,N3], DL N proba: 4 classes [N0,N1,N2,N3]
    X_n = np.array([feat_dict.get(f, 0.0) for f in n_feats], dtype=np.float32)
    X_n_sc = n_scaler.transform(X_n.reshape(1, -1))
    n_classes = list(n_models[0].classes_)
    rf_n_proba = np.mean([m.predict_proba(X_n_sc)[0] for m in n_models], axis=0)
    assert n_classes == sorted(n_classes), f"N classes not sorted: {n_classes}"

    stack_input_n = np.hstack([rf_n_proba, dl_n_proba]).reshape(1, -1)
    n_pred = int(stack_n.predict(stack_input_n)[0])
    n_stage = N_MAP.get(n_pred, "N2")

    return t_stage, n_stage


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5: Clinical RSF + Cox ensemble -> RFS (Task 3)
# ═══════════════════════════════════════════════════════════════════════════

def run_rfs(feat_dict, ehr, dl_risk_A, risk_B, risk_C):
    """Four-source rank-average fusion: A, B, C (DL) + RSF (radiomics).
    Each risk -> [0,1] via training quantile table, then equal-weight average.
    """
    import joblib

    rfs_models = joblib.load(str(RADIO_DIR / "clinical_rfs_model_v2_ensemble.joblib"))
    rfs_scaler = joblib.load(str(RADIO_DIR / "clinical_rfs_scaler_v2.joblib"))

    with open(str(RADIO_DIR / "clinical_rfs_features_v2.json")) as f:
        rfs_feats = json.load(f)

    _fill_empty_region(feat_dict, 'GTVp', rfs_feats)
    _fill_empty_region(feat_dict, 'GTVn', rfs_feats)

    age, gen, hpv = _ehr_clinical(ehr)
    feat_dict['Age']        = age
    feat_dict['Gender_enc'] = gen
    feat_dict['HPV_enc']    = hpv

    X    = np.array([feat_dict.get(f, 0.0) for f in rfs_feats], dtype=np.float32)
    X_sc = rfs_scaler.transform(X.reshape(1, -1))
    risk_RSF = float(np.mean([m.predict(X_sc)[0] for m in rfs_models]))

    def to_quantile(risk, name):
        tbl = np.load(str(RADIO_DIR / f"quantile_{name}.npy"))
        return float(np.searchsorted(tbl, risk) / len(tbl))

    q_A   = to_quantile(dl_risk_A, 'A')
    q_B   = to_quantile(risk_B,    'B')
    q_C   = to_quantile(risk_C,    'C')
    q_RSF = to_quantile(risk_RSF,  'RSF')

    fused = (q_A + q_B + q_C + q_RSF) / 4.0
    print(f"[stage5] qA={q_A:.3f} qB={q_B:.3f} qC={q_C:.3f} qRSF={q_RSF:.3f} fused={fused:.3f}", flush=True)

    # Grand Challenge: higher score = better prognosis (flip)
    return -1.0 * fused


# ═══════════════════════════════════════════════════════════════════════════
# I/O utilities
# ═══════════════════════════════════════════════════════════════════════════

def get_image_file(location):
    files = (
        glob(str(location / "*.mha")) + glob(str(location / "*.mhd"))
        + glob(str(location / "*.nii.gz")) + glob(str(location / "*.nii"))
        + glob(str(location / "*.tif"))    + glob(str(location / "*.tiff"))
    )
    if not files:
        raise FileNotFoundError(f"No image file in {location}")
    return sorted(files)[0]


def load_json(location):
    text = Path(location).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(cleaned)


def write_json(location, data):
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as e:
        import traceback
        print(f"[FATAL ERROR] {e}", flush=True)
        traceback.print_exc()
        # Fallback outputs to avoid missing submission penalty
        import json
        from pathlib import Path
        out = Path("/output")
        out.mkdir(parents=True, exist_ok=True)
        # T/N staging: fallback to most common class
        if not (out / "t-stage.json").exists():
            (out / "t-stage.json").write_text(json.dumps("T2"))
            print("[fallback] wrote t-stage.json = T2", flush=True)
        if not (out / "n-stage.json").exists():
            (out / "n-stage.json").write_text(json.dumps("N2"))
            print("[fallback] wrote n-stage.json = N2", flush=True)
        # RFS: fallback to median RFS (1354 days from training data)
        if not (out / "rfs.json").exists():
            (out / "rfs.json").write_text(json.dumps(1354.0))
            print("[fallback] wrote rfs.json = 1354.0", flush=True)
        # Segmentation: fallback to empty mask
        seg_dir = out / "images/head-neck-tumor-segmentation"
        seg_dir.mkdir(parents=True, exist_ok=True)
        if not (seg_dir / "output.mha").exists():
            import SimpleITK as sitk, numpy as np
            empty = sitk.GetImageFromArray(np.zeros((1,1,1), dtype=np.uint8))
            sitk.WriteImage(empty, str(seg_dir / "output.mha"))
            print("[fallback] wrote empty output.mha", flush=True)
        raise SystemExit(1)
