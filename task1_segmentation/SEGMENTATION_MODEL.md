# Task 1 Segmentation Model

Our segmentation uses the STU-Net-Small model adapted from the HECKTOR 2025
Team MEDAI solution (Top-1 solution for segmentation), trained on the
HECKTOR 2026 training data (Dataset001).

## License and Attribution
The original MEDAI segmentation pipeline is licensed under
**CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International)**,
Copyright (c) 2025 Team MEDAI. We use it for non-commercial academic
purposes (the HECKTOR 2026 challenge) in accordance with that license,
with attribution.

## Changes made relative to the original MEDAI solution
- We use a **5-fold ensemble** (the original MEDAI solution used a 10-fold ensemble).
- We conducted additional experiments to further improve segmentation.

## Citation
Please cite the MEDAI HECKTOR 2025 work when using or building on this
segmentation model:

```bibtex
@inproceedings{cailess,
  title={Less is More: Efficient PET/CT Segmentation and Multimodal Prediction of Recurrence-Free Survival and HPV Status in Head and Neck Cancer},
  author={Cai, Lishan and Liang, Xinglong and Zhang, Tianyu and Huang, Jiaju and Tan, Tao and Yin, Yunchao},
  booktitle={Fourth Head and Neck Cancer Tumor Lesion Segmentation, Diagnosis and Prognosis}
}
```

## Additional references
- STU-Net architecture: Huang et al., "STU-Net: Scalable and Transferable
  Medical Image Segmentation Models Empowered by Large-Scale Supervised
  Pre-training", 2023. https://github.com/uni-medical/STU-Net
- Framework: nnU-Net v2 (Isensee et al., "nnU-Net: a self-configuring method
  for deep learning-based biomedical image segmentation", Nature Methods, 2021).

## Our contribution
We trained STU-Net-Small (5-fold ensemble) on HECKTOR 2026 data. The trained
weights are provided in `weights/` (or via an external link). The training
configuration is in `plans.json` and `dataset.json`.

## To reproduce
1. Install STU-Net from the original repository (link above).
2. Use our dataset configuration and training plans, or use our provided
   weights directly for inference.
