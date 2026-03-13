"""
Richardson-Lucy deconvolution using skimage.restoration.richardson_lucy
"""
import numpy as np
import nrrd
from skimage.restoration import richardson_lucy

# ── parameters ────────────────────────────────────────────────────────────────
IMAGE_PATH  = "/nfs/data28/chuyu/data/20260109_095628/reference_stack_epi.nrrd"
PSF_PATH    = "/nfs/data26/chuyu/data/20250505_181137/20250505_181137_avg_bead_volume.nrrd"
OUTPUT_PREFIX = "/nfs/data28/chuyu/data/20260109_095628/reference_stack_epi_RL_deconv_iter"
CHECKPOINTS   = [5, 10, 20, 30, 50]   # save at these iteration counts
# ──────────────────────────────────────────────────────────────────────────────

# load image and PSF
print("Loading image and PSF...")
image, _ = nrrd.read(IMAGE_PATH)
image = np.transpose(image, (2, 1, 0))  # exchange x and z axis

psf, _ = nrrd.read(PSF_PATH)
psf = psf[::2, :, :]  # subsample z
print(f"  image shape: {image.shape}, PSF shape: {psf.shape}")

# normalize
image = image.astype(np.float32)
psf   = psf.astype(np.float32)

psf_sum = np.sum(psf)
if psf_sum <= 0:
    raise ValueError("PSF sum must be positive")
psf = psf / psf_sum

image = np.clip(image, 0, None)
orig_max = image.max()
if orig_max > 0:
    image = image / orig_max

# run RL from scratch for each checkpoint
checkpoints = sorted(CHECKPOINTS)
print(f"Running skimage RL, checkpoints: {checkpoints}")

for ckpt in checkpoints:
    print(f"  Running {ckpt} iterations...", flush=True)
    result = richardson_lucy(image, psf, num_iter=ckpt, clip=False)
    result = result * orig_max if orig_max > 0 else result
    out_path = f"{OUTPUT_PREFIX}{ckpt}.nrrd"
    nrrd.write(out_path, np.transpose(result, (2, 1, 0)).astype(np.float32))
    print(f"  --> Saved checkpoint: {out_path}", flush=True)

print("Done.")
