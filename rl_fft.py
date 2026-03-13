"""
Richardson-Lucy deconvolution — custom FFT-based implementation (no padding)
"""
import numpy as np
import nrrd

# ── parameters ────────────────────────────────────────────────────────────────
IMAGE_PATH  = "/nfs/data28/chuyu/data/20260109_095628/reference_stack_epi.nrrd"
PSF_PATH    = "/nfs/data26/chuyu/data/20250505_181137/20250505_181137_avg_bead_volume.nrrd"
OUTPUT_PREFIX = "/nfs/data28/chuyu/data/20260109_095628/reference_stack_epi_RL_deconv_scratch_iter"
CHECKPOINTS   = [5, 10, 20, 30, 50]   # save at these iteration counts
EPS = 1e-7
# ──────────────────────────────────────────────────────────────────────────────

def _pad_psf_to_image(psf, image_shape):
    psf_padded = np.zeros(image_shape, dtype=np.float32)
    psf_shape = np.array(psf.shape)
    img_shape = np.array(image_shape)
    start = (img_shape - psf_shape) // 2
    end   = start + psf_shape
    slices = tuple(slice(s, e) for s, e in zip(start, end))
    psf_padded[slices] = psf
    psf_padded = np.fft.ifftshift(psf_padded)
    return psf_padded

def _fft_convolve(image, psf_fft):
    return np.fft.ifftn(np.fft.fftn(image) * psf_fft).real

def _pad_psf_to_image(psf, image_shape):
    psf_padded = np.zeros(image_shape, dtype=np.float32)
    psf_shape = np.array(psf.shape)
    img_shape = np.array(image_shape)
    start = (img_shape - psf_shape) // 2
    end   = start + psf_shape
    slices = tuple(slice(s, e) for s, e in zip(start, end))
    psf_padded[slices] = psf
    psf_padded = np.fft.ifftshift(psf_padded)
    return psf_padded

# load image and PSF
print("Loading image and PSF...")
image, _ = nrrd.read(IMAGE_PATH)
image = np.transpose(image, (2, 1, 0))

psf, _ = nrrd.read(PSF_PATH)
psf = psf[::2, :, :]
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

# precompute FFTs once
psf_fft        = np.fft.fftn(_pad_psf_to_image(psf, image.shape))
psf_mirror_fft = np.fft.fftn(_pad_psf_to_image(psf[::-1, ::-1, ::-1], image.shape))

estimate = np.full_like(image, np.maximum(image.mean(), EPS))

# run RL, saving at each checkpoint
checkpoints = sorted(CHECKPOINTS)
max_iter    = checkpoints[-1]
next_ckpt   = iter(checkpoints)
save_at     = next(next_ckpt)

print(f"Running FFT RL up to {max_iter} iterations, checkpoints: {checkpoints}")
for i in range(1, max_iter + 1):
    conv          = _fft_convolve(estimate, psf_fft)
    relative_blur = image / (conv + EPS)
    estimate     *= _fft_convolve(relative_blur, psf_mirror_fft)
    print(f"  iter {i}/{max_iter}", flush=True)

    if i == save_at:
        result = estimate * orig_max if orig_max > 0 else estimate
        out_path = f"{OUTPUT_PREFIX}{i}.nrrd"
        nrrd.write(out_path, np.transpose(result, (2, 1, 0)).astype(np.float32))
        print(f"  --> Saved checkpoint: {out_path}", flush=True)
        try:
            save_at = next(next_ckpt)
        except StopIteration:
            break

print("Done.")
