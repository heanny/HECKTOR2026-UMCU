import numpy as np, nibabel as nib

seg_path = r'Z:\HOMES\HECKTOR_2026\task1_segmentation\stu-net_2025\predictions\oof_probmaps\CHUM-001.nii.gz'
npz_path = r'Z:\HOMES\HECKTOR_2026\task1_segmentation\stu-net_2025\predictions\oof_probmaps\CHUM-001.npz'
ct_path  = r'Z:\HOMES\HECKTOR_2026\task1_segmentation\stu-net_2025\nnUNet_raw\Dataset001_HECKTOR2026\imagesTr\CHUM-001_0000.nii.gz'
pet_path = r'Z:\HOMES\HECKTOR_2026\task1_segmentation\stu-net_2025\nnUNet_raw\Dataset001_HECKTOR2026\imagesTr\CHUM-001_0001.nii.gz'

print('=== NPZ ===')
npz = np.load(npz_path)
print('keys:', list(npz.keys()))
for k in npz.keys():
    a = npz[k]
    print(f'  [{k}]  shape={a.shape}  dtype={a.dtype}  min={float(a.min()):.4f}  max={float(a.max()):.4f}')
    print(f'         per-channel sum: {[float(a[c].sum()) for c in range(a.shape[0])]}')

print()
for label, path in [('Hard seg', seg_path), ('CT', ct_path), ('PET', pet_path)]:
    img = nib.load(path)
    d   = np.asarray(img.dataobj)
    print(f'=== {label} ===')
    print(f'  shape   : {img.shape}')
    print(f'  zooms   : {tuple(round(float(z),4) for z in img.header.get_zooms())}')
    print(f'  affine diag: {tuple(round(float(v),4) for v in np.diag(img.affine)[:3])}')
    print(f'  origin  : {tuple(round(float(v),2) for v in img.affine[:3,3])}')
    if label == 'Hard seg':
        print(f'  unique labels: {np.unique(d)}')
    print()
