# HECKTOR 2026 - Team UMCU

Multimodal PET/CT pipeline for the HECKTOR 2026 challenge: tumor segmentation
(Task 1), TN staging (Task 2), and recurrence-free survival (Task 3).

## Pipeline overview

- Task 1 - Segmentation: STU-Net-Small (5-fold ensemble, nnU-Net v2), adapted
  from the HECKTOR 2025 Team MEDAI Top-1 solution.
- Task 2 - TN staging: "Option B" stacking ensemble combining a radiomics
  Random Forest with a multitask DenseNet, via a logistic-regression meta-learner.
- Task 3 - RFS: rank-average fusion of deep-learning and radiomics survival predictors.

## Submissions (testing phase)

| | Algorithm1 (main) | Algorithm2 (backup) |
|---|---|---|
| Task 1 | STU-Net 5-fold | STU-Net 5-fold |
| Task 2 | Option B stacking | Option B stacking |
| Task 3 | Four-source rank fusion (A+B+C+RSF) | RSF + DL CoxPH (alpha=0.1) |
| Validation Weighted Score | 0.7205 | 0.6720 |
| Validation RFS C-index | 0.8179* | 0.6964 |

*Small-sample (n~50) estimate; the honest level is the 676-patient OOF C-index (0.7155).

## Repository structure

- task1_segmentation/ : STU-Net config (plans.json, dataset.json) + SEGMENTATION_MODEL.md
- task23_radiomics/scripts/ : feature extraction, RF/RSF training, fusion scripts
- task23_radiomics/resources/ : trained radiomics models (RF, stacking, Cox) + quantile tables
- task23_DL/ : deep-learning branch training code
- Algorithm1/ : MAIN submission container code (four-source fusion)
- Algorithm2/ : BACKUP submission container code (RSF+DL CoxPH)

## Model weights (not included - reproducible from code)

Large model files are NOT committed due to size, but can be reproduced:

- Segmentation weights (STU-Net-Small, 5-fold, ~116 MB each): reproduce using
  STU-Net (see MEDAI HECKTOR 2025 repo) with our config in
  task1_segmentation/plans.json and dataset.json.
- Deep-learning weights: reproduce with the scripts in task23_DL/.
- RSF ensemble (clinical_rfs_model_v2_ensemble.joblib, ~776 MB): a
  RandomSurvivalForest stores the full tree structure, making the serialized
  model very large. It is NOT uploaded; retrain from scratch using
  task23_radiomics/scripts/save_final_models_v2.py (5-seed, 433 features).

Smaller radiomics models (RF staging, stacking, Cox, scalers, quantile tables)
ARE included in task23_radiomics/resources/.

## Reproduction (high-level)

1. Segmentation: train STU-Net-Small (5-fold) with our config, produce
   out-of-fold predicted masks.
2. Feature extraction: extract_features_oof_mask_fixed.py (PyRadiomics from
   predicted masks, CT + PET).
3. Task 2: train_task2_final_v3.py (RF) + train_task2_option_b.py (stacking).
4. Task 3: save_final_models_v2.py (RSF, Cox) + gen_quantile_tables.py;
   four-source fusion applied in Algorithm1/inference.py.
5. Inference: build from Algorithm1/ (main) or Algorithm2/ (backup).

## Data

Uses the official HECKTOR 2026 dataset, NOT redistributed here.
Access: https://hecktor26.grand-challenge.org/

## License

Released under CC BY-NC 4.0 (Attribution-NonCommercial), consistent with the
MEDAI segmentation component. See LICENSE.

## Acknowledgements & Citations

Segmentation adapts the HECKTOR 2025 Team MEDAI solution (we use 5-fold; their
solution used 10-fold):

    @inproceedings{cailess,
      title={Less is More: Efficient PET/CT Segmentation and Multimodal Prediction of Recurrence-Free Survival and HPV Status in Head and Neck Cancer},
      author={Cai, Lishan and Liang, Xinglong and Zhang, Tianyu and Huang, Jiaju and Tan, Tao and Yin, Yunchao},
      booktitle={Fourth Head and Neck Cancer Tumor Lesion Segmentation, Diagnosis and Prognosis}
    }

Please also cite:
- STU-Net: Huang et al., 2023.
- nnU-Net: Isensee et al., Nature Methods, 2021.
- PyRadiomics: van Griethuysen et al., Cancer Research, 2017.
