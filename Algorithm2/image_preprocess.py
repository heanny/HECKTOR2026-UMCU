"""
image_preprocess.py — Reproduce the exact image pipeline used in training.

Stage A (preprocess_jin.py):
    raw CT.mha + PET.mha  ->  resample to 1 mm (CT/PET intersection, identity
    direction)  ->  crop a fixed 200x200x310 (X,Y,Z) box centred on the PET
    hot-spot.  Output: cropped CT/PET SimpleITK images at 1 mm.
    These feed BOTH the STU-Net segmentation and the DenseNet.

Stage B (dataset.py preprocess_patient):
    cropped CT/PET (1 mm) + hard segmentation (labels 0/1/2)  ->  stack
    [CT, PET, hard_T, hard_N]  ->  pad to 200x200x310  ->  resample 1 mm -> 2 mm
    (scipy zoom, order=1)  ->  [4, 100, 100, 155] float32 for the DenseNet.

All array conventions match training: SimpleITK arrays are [z,y,x]; we
transpose to [x,y,z] to match the nibabel-based training code.
"""

import warnings

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from skimage.measure import label as sk_label

from constants import (
    NATIVE_SPACING, TARGET_SPACING, PAD_TO_SHAPE, CROP_BOX_SIZE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Stage A — resample + PET-centred crop  (from preprocess_jin.py, verbatim)
# ═══════════════════════════════════════════════════════════════════════════

def _get_bounding_boxes(ct, pet):
    ct_origin  = np.array(ct.GetOrigin())
    pet_origin = np.array(pet.GetOrigin())
    ct_max     = ct_origin  + np.array(ct.GetSize())  * np.array(ct.GetSpacing())
    pet_max    = pet_origin + np.array(pet.GetSize()) * np.array(pet.GetSpacing())
    return np.concatenate([np.maximum(ct_origin, pet_origin),
                           np.minimum(ct_max, pet_max)])


def _resample_images(ct, pet):
    resampling = [1, 1, 1]
    resampler  = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)
    bb   = _get_bounding_boxes(ct, pet)
    size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
    resampler.SetOutputOrigin(bb[:3].tolist())
    resampler.SetSize([int(k) for k in size])
    resampler.SetInterpolator(sitk.sitkBSpline)
    ct  = resampler.Execute(ct)
    pet = resampler.Execute(pet)
    return ct, pet, bb


def _register_pet_to_ct(ct, pet):
    ct  = sitk.Cast(ct,  sitk.sitkFloat32)
    pet = sitk.Cast(pet, sitk.sitkFloat32)
    transform = sitk.CenteredTransformInitializer(
        ct, pet, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(transform)
    final_transform = reg.Execute(ct, pet)
    return sitk.Resample(pet, ct, final_transform,
                         sitk.sitkLinear, 0.0, pet.GetPixelID())


def _get_roi_center(pet_tensor, z_top_fraction=0.75, z_score_threshold=1.0):
    shape  = np.array(pet_tensor.shape)
    crop_z = int(z_top_fraction * shape[2])
    top    = pet_tensor[..., crop_z:]
    mask   = ((top - top.mean()) / (top.std() + 1e-8)) > z_score_threshold
    if not mask.any():
        warnings.warn("No high-intensity region found - using geometric centre.")
        center_top = (np.array(top.shape) / 2).astype(int)
    else:
        labeled, n = sk_label(mask, return_num=True, connectivity=3)
        if n > 0:
            sizes   = np.bincount(labeled.ravel())[1:]
            largest = np.argmax(sizes) + 1
            idx     = np.argwhere(labeled == largest)
        else:
            idx = np.argwhere(mask)
        center_top = np.mean(idx, axis=0)
    return (center_top + np.array([0, 0, crop_z])).astype(int)


def _crop_neck_region(ct, pet, crop_box_size=CROP_BOX_SIZE,
                      z_top_fraction=0.75, z_score_threshold=1.0):
    pet_np_zyx = sitk.GetArrayFromImage(pet).astype(np.float32)
    pet_tensor = np.transpose(pet_np_zyx, (2, 1, 0))          # -> [x, y, z]
    crop_box_size = np.asarray(crop_box_size, dtype=int)
    center    = _get_roi_center(pet_tensor, z_top_fraction, z_score_threshold)
    img_shape = np.asarray(pet_tensor.shape)
    box_start = np.clip(center - crop_box_size // 2, 0, img_shape)
    box_end   = np.clip(box_start + crop_box_size,   0, img_shape)
    box_start = np.maximum(box_end - crop_box_size,  0)
    index = [int(i) for i in box_start]
    size  = [int(e - s) for s, e in zip(box_start, box_end)]
    ct_crop  = sitk.RegionOfInterest(ct,  size=size, index=index)
    pet_crop = sitk.RegionOfInterest(pet, size=size, index=index)
    return ct_crop, pet_crop


def resample_and_crop(ct_path, pet_path):
    """raw CT/PET .mha -> cropped CT/PET SimpleITK images at 1 mm.

    Mirrors preprocess_jin.process_case, including the PET->CT registration
    fallback when the intersection resample fails.
    """
    ct  = sitk.ReadImage(str(ct_path))
    pet = sitk.ReadImage(str(pet_path))
    try:
        ct_r, pet_r, _ = _resample_images(ct, pet)
    except Exception:
        pet = _register_pet_to_ct(ct, pet)
        ct_r, pet_r, _ = _resample_images(ct, pet)
    ct_crop, pet_crop = _crop_neck_region(ct_r, pet_r)
    return ct_crop, pet_crop


# ═══════════════════════════════════════════════════════════════════════════
# Stage B — DenseNet 4-channel preprocessing  (from dataset.py, verbatim)
# ═══════════════════════════════════════════════════════════════════════════

def _pad_to(volume, target_shape):
    spatial = volume.shape[-3:]
    pad_width = []
    for cur, tgt in zip(spatial, target_shape):
        diff   = max(0, tgt - cur)
        before = diff // 2
        after  = diff - before
        pad_width.append((before, after))
    full_pad = [(0, 0)] + pad_width if volume.ndim == 4 else pad_width
    return np.pad(volume, full_pad, mode="constant", constant_values=0)


def _resample(volume, native_spacing, target_spacing, order=1):
    zf = tuple(n / t for n, t in zip(native_spacing, target_spacing))
    zf = (1.0,) + zf if volume.ndim == 4 else zf
    return zoom(volume, zf, order=order).astype(np.float32)


def _normalize_ct(ct):
    ct = np.clip(ct, -200.0, 300.0)
    return (ct + 200.0) / 500.0


def _normalize_pet(pet):
    p99 = np.percentile(pet, 99)
    if p99 > 1e-6:
        pet = pet / p99
    return np.clip(pet, 0.0, 1.0)


def build_densenet_volume(ct_xyz, pet_xyz, hard_seg_xyz):
    """[x,y,z] raw CT/PET + integer seg (0/1/2) -> [4,100,100,155] float32.

    Channels in canonical order: CT, PET, hard_T, hard_N.
    """
    ct_norm = _normalize_ct(ct_xyz)
    pet_norm = _normalize_pet(pet_xyz)
    hard_t = (hard_seg_xyz == 1).astype(np.float32)
    hard_n = (hard_seg_xyz == 2).astype(np.float32)

    img = np.stack([ct_norm, pet_norm, hard_t, hard_n], axis=0)   # [4,x,y,z]

    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    img = _pad_to(img, PAD_TO_SHAPE)
    img = _resample(img, NATIVE_SPACING, TARGET_SPACING, order=1)

    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    return img.astype(np.float32)


def sitk_to_xyz(image):
    """SimpleITK image -> numpy array in [x, y, z] order (matches nibabel)."""
    return np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
