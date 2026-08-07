import monai, torch
print("MONAI:", monai.__version__)
print("torch:", torch.__version__)

# Quick smoke-test of all transforms we plan to use
import numpy as np
from monai.transforms import (
    RandFlip, RandAffine,
    RandGaussianNoise, RandScaleIntensity, RandShiftIntensity, RandGaussianSmooth,
)

x = np.random.rand(4, 100, 100, 155).astype(np.float32)
xt = torch.from_numpy(x)

# spatial
for T in [
    RandFlip(spatial_axis=0, prob=1.0),
    RandFlip(spatial_axis=1, prob=1.0),
    RandFlip(spatial_axis=2, prob=1.0),
    RandAffine(
        prob=1.0,
        rotate_range=[(-0.26, 0.26)]*3,
        scale_range=[(-0.1, 0.1)]*3,
        translate_range=[(-10, 10)]*3,
        spatial_size=(100, 100, 155),
        mode="bilinear",
        padding_mode="zeros",
    ),
]:
    out = T(xt)
    print(f"  {T.__class__.__name__}: in={xt.shape} out={out.shape} type={type(out).__name__}")
    xt = torch.as_tensor(np.array(out), dtype=torch.float32)  # normalise back

# intensity (1-channel)
x1 = xt[0:1]
for T in [
    RandGaussianNoise(prob=1.0, mean=0.0, std=0.02),
    RandScaleIntensity(factors=0.1, prob=1.0),
    RandShiftIntensity(offsets=0.05, prob=1.0),
    RandGaussianSmooth(sigma_x=(0.5,1.5), sigma_y=(0.5,1.5), sigma_z=(0.5,1.5), prob=1.0),
]:
    out = T(x1)
    print(f"  {T.__class__.__name__}: in={x1.shape} out={out.shape} type={type(out).__name__}")
    x1 = torch.as_tensor(np.array(out), dtype=torch.float32)

print("All transforms OK")
