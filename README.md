![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)
![nnU-Net](https://img.shields.io/badge/nnU--Net-2.1-orange)

# HECKTOR 2026 - Team UMCU

<!---
Top-2 Solution (validation leaderboard) for Multitask Head and Neck Cancer Tumor Analysis
---->

This repository contains our solutions for the three tasks of the HECKTOR 2026 Challenge.

Multimodal PET/CT pipeline for the [HECKTOR 2026 challenge](https://hecktor26.grand-challenge.org/): tumor segmentation
(Task 1), TN staging (Task 2), and prognosis (Task 3).

## 🔍 Pipeline overview

- Task 1 - Segmentation: STU-Net-Small (5-fold ensemble, nnU-Net v2), adapted from the HECKTOR 2025 Team MEDAI Top-1 solution by Cai et al., 2025.
- Task 2 - TN staging: Stacking ensemble combining a radiomics Random Forest with a multitask DenseNet, via a logistic-regression meta-learner.
- Task 3 - Prognosis: rank-average fusion of deep-learning and radiomics survival predictors.

## 📊 Submissions and results (testing phase results will be released soon)

| | Algorithm1 (main) | Algorithm2 (backup) |
|---|---|---|
| Task 1 | STU-Net 5-fold | STU-Net 5-fold |
| Task 2 | Stacking ensemble | Option B stacking |
| Task 3 | Four-source rank ensemble (A+B+C+RSF) | RSF + DL CoxPH (alpha=0.1) |
| Validation Leaderboard Weighted Score* | 0.7205 | 0.6720 |
| Validation Leaderboard RFS C-index* | 0.8179 | 0.6964 |

*Small-sample (n~50) estimate; the honest level is the 676-patient out-of-fold C-index (0.7155).

## 📋 Repository structure

- task1_segmentation/ : STU-Net config (plans.json, dataset.json) + SEGMENTATION_MODEL.md
- task23_radiomics/scripts/ : feature extraction, RF/RSF training, fusion scripts
- task23_radiomics/resources/ : trained radiomics models (RF, stacking, Cox) + quantile tables
- task23_DL/ : deep-learning branch training code
- Algorithm1/ : MAIN submission container code (four-source fusion) with requirement.txt
- Algorithm2/ : BACKUP submission container code (RSF+DL CoxPH) with requirement.txt

### Setup

Requirements:
- Python 3.10+
- Segmentation: STU-Net (nnU-Net v2.1) for training; nnunetv2==2.5.1 for inference
- Radiomics: PyRadiomics, scikit-learn, scikit-survival
- Deep learning: PyTorch, MONAI

Install radiomics/DL dependencies (from Algorithm1/ and/or Algorithm2/ directory):
```
pip install -r requirements.txt
```
For segmentation, install STU-Net from https://github.com/uni-medical/STU-Net or the [MEDAI HECKTOR 2025 repository](https://github.com/Liiiii2101/HECKTOR2025-MEDAI) by Cai et al., 2025.

### Model weights (not included - reproducible from code)

Large model files are NOT committed due to size, but can be reproduced:

- Segmentation weights (STU-Net-Small, 5-fold, ~116 MB each): reproduce using STU-Net (see MEDAI HECKTOR 2025 repo) with our config in task1_segmentation/plans.json and dataset.json.
- Deep-learning weights: reproduce with the scripts in task23_DL/.
- RSF ensemble (clinical_rfs_model_v2_ensemble.joblib, ~776 MB): a RandomSurvivalForest stores the full tree structure, making the serialized model very large. It is NOT uploaded; retrain from scratch using task23_radiomics/scripts/save_final_models_v2.py (5-seed, 433 features).

Smaller radiomics models (RF staging, stacking, Cox, scalers, quantile tables) ARE included in task23_radiomics/resources/.

### Reproduction (high-level)

1. Segmentation: train STU-Net-Small (5-fold) with our config, produce out-of-fold predicted masks.
2. Feature extraction: extract_features_oof_mask_fixed.py (PyRadiomics from predicted masks, CT + PET).
3. Task 2: train_task2_final_v3.py (RF) + train_task2_option_b.py (stacking).
4. Task 3: save_final_models_v2.py (RSF, Cox) + gen_quantile_tables.py; four-source fusion applied in Algorithm1/inference.py.
5. Inference: build from Algorithm1/ (main) or Algorithm2/ (backup).

## 🏥 Data

Uses the official HECKTOR 2026 dataset, NOT redistributed here. Please check access on [HECKTOR 2026 dataset website](https://hecktor26.grand-challenge.org/dataset/).

## 🖇️ Citation
Thank you for finding our work helpful, please cite our work by: xxxx(to be filled). Please note that commercial use is prohibited (CC BY-NC 4.0).

## 🕊️ Acknowledgements & Citations

- Segmentation adapts the [HECKTOR 2025 Team MEDAI](https://github.com/Liiiii2101/HECKTOR2025-MEDAI) solution by Cai et al., 2025. We use 5-fold ensemble, and experimented other methods on top of it to improve the segmentation; their solution used 10-fold emsemble in HECKTOR 2025 challenge.
- STU-Net: Huang et al., 2023.
- nnU-Net: Isensee et al., Nature Methods, 2021.
- PyRadiomics: van Griethuysen et al., Cancer Research, 2017.
  
@inproceedings{cailess,
title={Less is More: Efficient PET/CT Segmentation and Multimodal Prediction of Recurrence-Free Survival and HPV Status in Head and Neck Cancer},
author={Cai, Lishan and Liang, Xinglong and Zhang, Tianyu and Huang, Jiaju and Tan, Tao and Yin, Yunchao},
booktitle={Fourth Head and Neck Cancer Tumor Lesion Segmentation, Diagnosis and Prognosis}}

@article{huang2023stunet,
  title={STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training},
  author={Huang, Ziyan and Wang, Haoyu and Deng, Zhongying and Ye, Jin and Su, Yanzhou and Sun, Hui and He, Junjun and Gu, Yun and Gu, Lixu and Zhang, Shaoting and Qiao, Yu},
  journal={arXiv preprint arXiv:2304.06716},
  year={2023}
}

@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}

@article{vanGriethuysen2017,
  title={Computational Radiomics System to Decode the Radiographic Phenotype},
  author={van Griethuysen, Joost J. M. and Fedorov, Andriy and Parmar, Chintan and Hosny, Ahmed and Aucoin, Nicole and Narayan, Vivek and Beets-Tan, Regina G. H. and Fillion-Robin, Jean-Christophe and Pieper, Steve and Aerts, Hugo J. W. L.},
  journal={Cancer Research},
  volume={77},
  number={21},
  pages={e104--e107},
  year={2017},
  publisher={AACR},
  doi={10.1158/0008-5472.CAN-17-0339},
  URL={https://aacrjournals.org}
}

## ✳️ License

Released under CC BY-NC 4.0 (Attribution-NonCommercial), consistent with the
MEDAI segmentation component. See LICENSE.
