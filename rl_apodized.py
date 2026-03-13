"""
Richardson-Lucy deconvolution — FFT-based with apodization to suppress boundary artefacts.

Instead of reflect-padding (which imposes mirror symmetry the data don't satisfy),
we smoothly taper the image to its background level near the edges before deconvolution.
This makes the padded region consistent with zero/reflect padding and pushes any
residual artefacts into the taper margin, which is then cropped out of the output.

Steps
-----
1. Estimate background from a low percentile of the image.
2. Build a 3D Tukey window (flat centre, cosine roll-off over `margin` voxels).
3. Blend: image_apo = background + window * (image - background)
4. Zero-pad by margin on each side and run FFT-based RL on the padded volume.
5. Crop out the margin from the output; boundary region is discarded.
"""
import numpy as np
import nrrd

# ── parameters ────────────────────────────────────────────────────────────────
IMAGE_PATH    = "/nfs/data28/chuyu/data/20260109_095628/reference_stack_epi.nrrd"
PSF_PATH      = "/nfs/data26/chuyu/data/20250505_181137/20250505_181137_avg_bead_volume.nrrd"
OUTPUT_PREFIX = "/nfs/data28/chuyu/data/20260109_095628/reference_stack_epi_RL_deconv_apodized_iter"
CHECKPOINTS   = [5, 10, 20, 30, 50]   # save at these iteration counts
EPS           = 1e-7
BG_PERCENTILE = 5      # percentile used to estimate background level
TAPER_RATIO   = 0.1    # fraction of each axis used for the cosine roll-off
# ──────────────────────────────────────────────────────────────────────────────


def _pad_psf_to_image(psf, image_shape):
    psf_padded = np.zeros(image_shape, dtype=np.float32)
    psf_shape  = np.array(psf.shape)
    img_shape  = np.array(image_shape)
    start  = (img_shape - psf_shape) // 2
    end    = start + psf_shape
    slices = tuple(slice(s, e) for s, e in zip(start, end))
    psf_padded[slices] = psf
    psf_padded = np.fft.ifftshift(psf_padded)
    return psf_padded


def _fft_convolve(image, psf_fft):
    return np.fft.ifftn(np.fft.fftn(image) * psf_fft).real


def _tukey_window_1d(n, margin):
    """1-D window: ones in the centre, cosine taper over `margin` samples at each end."""
    w = np.ones(n, dtype=np.float32)
    if margin == 0:
        return w
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(margin) / margin)).astype(np.float32)
    w[:margin]  = ramp
    w[-margin:] = ramp[::-1]
    return w


def _tukey_window_3d(shape, taper_ratio):
    """Separable 3-D Tukey window."""
    margins = [max(1, int(s * taper_ratio)) for s in shape]
    w = np.ones(shape, dtype=np.float32)
    for axis, (s, m) in enumerate(zip(shape, margins)):
        w1d = _tukey_window_1d(s, m)
        # broadcast along the correct axis
        idx = [np.newaxis] * 3
        idx[axis] = slice(None)
        w *= w1d[tuple(idx)]
    return w


def apodize(image, bg_percentile=BG_PERCENTILE, taper_ratio=TAPER_RATIO):
    """Return apodized image and the background level used."""
    background = float(np.percentile(image, bg_percentile))
    window     = _tukey_window_3d(image.shape, taper_ratio)
    return background + window * (image - background), background


# ── load ──────────────────────────────────────────────────────────────────────
print("Loading image and PSF...")
image, image_header = nrrd.read(IMAGE_PATH)
image = np.transpose(image, (2, 1, 0))   # swap x/z to (z, y, x)

psf, _ = nrrd.read(PSF_PATH)
psf    = psf[::2, :, :]
print(f"  image shape: {image.shape},  PSF shape: {psf.shape}")

# ── pre-process ───────────────────────────────────────────────────────────────
image = image.astype(np.float32)
psf   = psf.astype(np.float32)

psf_sum = np.sum(psf)
if psf_sum <= 0:
    raise ValueError("PSF sum must be positive")
psf = psf / psf_sum

image    = np.clip(image, 0, None)
orig_max = image.max()
if orig_max > 0:
    image = image / orig_max   # work in [0, 1]

# apodize: taper towards background near edges
image_apo, bg = apodize(image, BG_PERCENTILE, TAPER_RATIO)
margin = [max(1, int(s * TAPER_RATIO)) for s in image.shape]
print(f"  background estimate: {bg:.5f},  taper margins (voxels): {margin}")

# zero-pad by the taper margin so the padded region matches the tapered edge
image_padded = np.pad(image_apo, [(m, m) for m in margin], mode='constant', constant_values=bg)
crop = tuple(slice(m, m + s) for m, s in zip(margin, image.shape))

# ── precompute FFTs ───────────────────────────────────────────────────────────
print("Precomputing PSF FFTs...")
psf_fft        = np.fft.fftn(_pad_psf_to_image(psf,                    image_padded.shape))
psf_mirror_fft = np.fft.fftn(_pad_psf_to_image(psf[::-1, ::-1, ::-1], image_padded.shape))

estimate = np.full_like(image_padded, np.maximum(image_padded.mean(), EPS))

# ── run RL with checkpointing ─────────────────────────────────────────────────
checkpoints = sorted(CHECKPOINTS)
max_iter    = checkpoints[-1]
ckpt_iter   = iter(checkpoints)
save_at     = next(ckpt_iter)

print(f"Running apodized FFT RL up to {max_iter} iterations, checkpoints: {checkpoints}")
for i in range(1, max_iter + 1):
    conv          = _fft_convolve(estimate, psf_fft)
    relative_blur = image_padded / (conv + EPS)
    estimate     *= _fft_convolve(relative_blur, psf_mirror_fft)
    print(f"  iter {i}/{max_iter}", flush=True)

    if i == save_at:
        # crop out the apodized margin — these voxels are unreliable
        result = estimate[crop]
        if orig_max > 0:
            result = result * orig_max
        out_path = f"{OUTPUT_PREFIX}{i}.nrrd"
        nrrd.write(out_path, np.transpose(result, (2, 1, 0)).astype(np.float32))
        print(f"  --> Saved: {out_path}", flush=True)
        try:
            save_at = next(ckpt_iter)
        except StopIteration:
            break

print("Done.")
