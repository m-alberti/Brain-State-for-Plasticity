# DEWEY_v7 - July 2026 SWEEP2 Full dMRI Preprocessing Pipeline

## Overview

All steps are executed via bash scripts generated in Python and logged per subject/session. Intermediate QA images (SlicesDir) are produced at each major stage.
DEWEY generates a bash script and log file specific to each participant session - can be parallelised via PAR_POOL.sh 

---

## Pipeline at a Glance

```
Raw BIDS (mag + phase, PA + AP)
        │
        ▼
[01] MP-PCA Denoising           complex domain (MRtrix3 dwidenoise)
        │
        ▼
[02] Gibbs Ringing Correction   mrdegibbs (MRtrix3)
        │
        ▼
[03] B0 Field Estimation        topup (FSL)
        │
        ▼
[04] Eddy + Motion Correction   eddy / eddy_cuda (FSL)
        │
        ▼
[05] Signal Drift Correction    custom sgbak/drift module
        │
        ▼
[06] Bias Field Correction      N4BiasFieldCorrection (ANTs)
        │
        ▼
[07] Brain Extraction           bet (FSL)
        │
        ▼
[08] Model Fitting              DKI / DTI / NODDI
        │
        ▼
[09] Coregistration → MNI152    ANTs SyN (nonlinear)
        │
        ▼
[10] Smoothing                  fslmaths -s (Gaussian kernel)
```

---

## Step 01 — MP-PCA Denoising (Complex Domain)

**Tool:** MRtrix3 `dwidenoise`, `mrcalc`  
**Input:** `part-mag_dwi`, `part-phase_dwi`  
**Output:** `part-mag_dwi_denoised`, `part-noise`, `part-residue`, `part-sqr_wei_noise`

Denoising is performed in the **complex domain** to avoid the Rician noise floor bias present in magnitude-only images. This is especially important for low-SNR shells and for subsequent kurtosis and NODDI fitting.

> **Optional:** Rician noise floor rectification is available as a post-denoising step for magnitude-only acquisitions where phase data are unavailable.

### Substeps

**1. Phase → radians**  
Scanner phase is in arbitrary integer units. Convert to radians:

```
phase_rad = phase × π / phase_range
```

Values should lie in [−π, +π]. Verified via `mrstats` logged to file.

**2. Construct complex image**

```bash
mrcalc mag phase_rad -polar complex.nii.gz
```

**3. MP-PCA denoising**

```bash
dwidenoise complex.nii.gz denoised.nii.gz -noise noise.nii.gz -nthreads 12
```

Marchenko-Pastur PCA operates on the local patch covariance structure. Using the complex signal doubles the number of real-valued components (real + imaginary), which stabilises the MP threshold and improves rank estimation.

**4. Recover magnitude**

```bash
mrcalc denoised.nii.gz -abs denoised.nii.gz
```

**5. Residuals and QA maps**

```bash
fslmaths mag_original -sub mag_denoised residue          # raw residuals
fslmaths noise -sqrt sqrt_noise                          # noise sigma
fslmaths residue -div sqrt_noise sqr_wei_noise           # squared weighted residuals
```

The squared weighted residual map should be spatially uniform white noise if denoising is well-conditioned.

**6. Cleanup**  
Phase-in-radians and complex intermediate files are removed after completion.

**QA outputs:** SlicesDir PNGs for denoised image, noise map, and squared weighted residual map.

---

## Step 02 — Gibbs Ringing Correction

**Tool:** MRtrix3 `mrdegibbs`  
**Input:** `part-mag_dwi_denoised`  
**Output:** `part-mag_dwi_degibbs`

Gibbs ringing arises from k-space truncation (finite matrix acquisition). `mrdegibbs` uses the local subvoxel-shift method to suppress ringing without blurring edges.

> **Order matters:** Gibbs correction must be applied *before* any interpolation (topup, eddy resampling) to avoid corrupting the subvoxel frequency structure on which the correction depends. It is applied *after* denoising to avoid amplifying noise-induced pseudo-ringing.

---

## Step 03 — B0 Field Estimation (`topup`)

**Tool:** FSL `topup`  
**Input:** b=0 volumes from PA and AP acquisitions, `.acqp` file  
**Output:** B0 field estimate (`_fieldmap`), corrected b=0 images

PA and AP b=0 volumes have **opposite susceptibility-induced distortions**. `topup` estimates the underlying B0 inhomogeneity field by modelling the warp that maps one encoding direction to the other.

**Inputs required:**
- Extracted b=0 volumes from both phase-encoding directions (concatenated)
- Acquisition parameter file specifying phase-encode vectors and total readout time

The estimated field coefficients are passed directly to `eddy` and are not applied as a standalone warp at this stage.

**QA:** Visual check of corrected b=0 images — anatomical distortions (frontal lobe, cerebellum, EPI ghosting) should be substantially reduced.

---

## Step 04 — Eddy Current & Motion Correction (`eddy`)

**Tool:** FSL `eddy` / `eddy_cuda` (GPU)  
**Input:** Gibbs-corrected DWI, `topup` field, bvals, bvecs, brain mask  
**Output:** Corrected DWI, rotated bvecs, motion/outlier QC files

`eddy` simultaneously corrects for:

- **Eddy current distortions** — gradient-induced field changes, direction-dependent
- **Susceptibility distortions** — via the `topup` field estimate
- **Subject motion** — rigid-body, 6 DOF, estimated per volume
- **Outlier slices** — signal dropout detection and replacement via Gaussian process

bvecs are rotated to match the corrected gradient orientations. The rotated bvecs **must** be used for all downstream model fitting.

**Key options:**

```
--repol        outlier slice replacement
--cnr_maps     contrast-to-noise ratio maps per shell (QA)
--residuals    voxelwise residuals (QA)
--slm=linear   second-level model for eddy field across directions
```

**QA outputs:** `eddy_qc` report — motion parameters, outlier fraction per volume/slice, SNR per shell.

---

## Step 05 — Signal Drift Correction

**Tool:** Custom module (`sgbak` / `FF_DWI_Drift`)  
**Input:** Eddy-corrected DWI, interleaved b=0 volumes  
**Output:** Drift-corrected DWI

Long multi-shell acquisitions are susceptible to **spatio-temporal B0 signal drift** caused by gradient coil heating. This manifests as a progressive change in signal intensity across the time series that is not captured by `eddy` (which models spatial, not temporal, field changes).

**Approach:**

1. Extract the time series of interleaved b=0 volumes distributed throughout the acquisition
2. Model the temporal drift as a polynomial or spline fitted to b=0 signal evolution
3. Normalise each DWI volume by the drift factor interpolated at its acquisition time

MD (mean diffusivity) is the metric most sensitive to uncorrected drift, as it scales linearly with overall signal level. Drift correction is therefore critical before DKI and DTI fitting.

---

## Step 06 — Bias Field Correction

**Tool:** ANTs `N4BiasFieldCorrection`  
**Input:** Drift-corrected DWI (b=0 volume used to estimate field)  
**Output:** Bias-corrected DWI

Slowly-varying B1 receive inhomogeneity creates a smooth spatial intensity gradient across the volume. N4 iteratively estimates and removes this field using B-spline regularisation.

The bias field is estimated from the b=0 image (highest SNR) and applied to the full DWI series to preserve relative signal ratios between shells.

**QA:** Comparison of mean b=0 intensity profiles before and after correction; histogram of b=0 signal within the brain mask should become more uniform.

---

## Step 07 — Brain Extraction

**Tool:** FSL `bet`  
**Input:** Mean b=0 image (after bias correction)  
**Output:** Brain mask (`_mask.nii.gz`)

A robust brain mask is generated from the mean b=0 image. The mask is applied at multiple stages:

- At the `eddy` stage (required input)
- Before model fitting (restrict fitting to brain voxels)
- Before coregistration (improves ANTs registration by excluding skull)
- As the analysis mask for voxel-wise statistics

**Typical parameters:** `-f 0.25 -g 0 -m` — fractional intensity threshold may need per-subject adjustment. Visual QC of mask boundaries is mandatory.

---

## Step 08 — Model Fitting

All model fitting is performed on the eddy-corrected, drift-corrected, bias-corrected DWI using the **rotated bvecs** from `eddy`.

### 8a — Diffusion Kurtosis Imaging (DKI)

**Tool:** dipy (`dki.py`)  
**Shells used:** b = 0, 1000, 2000 s/mm² (all shells)

The kurtosis tensor extends the standard diffusion tensor with fourth-order terms describing non-Gaussian diffusion behaviour. A weighted least-squares fit is applied.

| Output map | Description |
|---|---|
| FA | Fractional Anisotropy |
| MD | Mean Diffusivity |
| AD | Axial Diffusivity |
| RD | Radial Diffusivity |
| MK | Mean Kurtosis |
| AK | Axial Kurtosis |
| RK | Radial Kurtosis |

### 8b — Diffusion Tensor Imaging (DTI)

**Tool:** FSL `dtifit`  
**Shells used:** b = 0 + b = 1000 s/mm² (single-shell fit)

Standard tensor model used for comparison and to drive the ANTs registration (FA map as moving image).

| Output map | Description |
|---|---|
| FA | Fractional Anisotropy |
| MD | Mean Diffusivity |
| AD | Axial Diffusivity (L1) |
| RD | Radial Diffusivity ((L2+L3)/2) |
| V1 | Principal eigenvector |

### 8c — NODDI

**Tool:** AMICO (`amico.py`)  
**Shells used:** All shells

The NODDI model decomposes the DWI signal into three tissue compartments using a fixed-radius stick-and-ball model convolved with an orientation distribution function.

| Output map | Description |
|---|---|
| NDI | Neurite Density Index |
| ODI | Orientation Dispersion Index |
| FWF | Free Water Fraction (isotropic compartment) |

> AMICO requires prior computation of a response kernel for each unique acquisition protocol (bvals, bvecs, TE, δ, Δ). The kernel is computed once and cached for reuse across subjects.

---

## Step 09 — Coregistration to MNI152 Space

**Tool:** ANTs `antsRegistration`, `antsApplyTransforms`  
**Reference:** FSL MNI152 1mm standard brain  
**Output:** Warp field (`_Warp.nii.gz`), affine matrix (`_0GenericAffine.mat`), MNI-space scalar maps

Each subject's DTI-FA map is registered to the MNI152 1mm template using a two-stage ANTs SyN nonlinear registration.

### Stage 1 — Rigid + Affine

```bash
antsRegistration \
  --dimensionality 3 \
  --initial-moving-transform [MNI_template, FA_subject, 1] \
  --transform Rigid[0.1] \
  --metric MI[MNI_template, FA_subject, 1, 32] \
  --transform Affine[0.1] \
  --metric MI[MNI_template, FA_subject, 1, 32]
```

### Stage 2 — Nonlinear SyN

```bash
  --transform SyN[0.1, 3, 0] \
  --metric CC[MNI_template, FA_subject, 1, 4]
```

Cross-correlation (CC) is preferred over MI for the nonlinear stage with FA images due to the structured spatial contrast in white matter.

### Applying Transforms to Scalar Maps

The combined warp (affine + SyN) is applied to all scalar maps from DKI, DTI, and NODDI using `antsApplyTransforms`:

```bash
antsApplyTransforms \
  -d 3 \
  -i subject_FA.nii.gz \
  -r MNI152_1mm.nii.gz \
  -o FA_MNI.nii.gz \
  -t subject_Warp.nii.gz \
  -t subject_0GenericAffine.mat \
  -n Linear
```

Use `-n BSpline[3]` for smoother interpolation of non-FA maps. Use `-n NearestNeighbor` for label/atlas images.

**QA:** Overlay of warped FA on MNI template; verify alignment of corpus callosum, internal capsule, and cerebellar tracts.

---

## Step 10 — Spatial Smoothing

**Tool:** FSL `fslmaths -s`  
**Applied to:** All scalar maps in MNI152 space  
**Kernel:** Gaussian (σ in mm; FWHM = σ × 2.355)

Smoothing is applied **after** warping to MNI space to operate in isotropic voxel dimensions. Two kernel sizes are used:

| FWHM | σ (mm) | Use |
|---|---|---|
| 4 mm | 1.70 | Conservative / fine-grained WM structures |
| 6 mm | 2.55 | Standard |

```bash
fslmaths FA_MNI.nii.gz -s <sigma> FA_MNI_s<fwhm>mm.nii.gz
```

Results are considered robust if they replicate across both kernel sizes.

---

## QA Checkpoints Summary

| Step | QA method |
|---|---|
| Denoising | `mrstats` on phase_rad (verify range −π to +π); SlicesDir PNGs for noise map and weighted residuals |
| Gibbs | Visual inspection at tissue boundaries before vs. after |
| topup | Overlay corrected b=0 PA vs. AP; check distortion symmetry |
| eddy | `eddy_qc` report: motion timeseries, outlier fraction, CNR per shell |
| Drift | b=0 signal timeseries plot; MD map before vs. after |
| Bias field | b=0 intensity histogram within brain mask before vs. after |
| Brain mask | Overlay on b=0; verify frontal and cerebellar coverage |
| Coregistration | Warped FA overlaid on MNI template; check CC/MI metric convergence |
| Model fitting | FA ∈ [0, 1]; MD ≈ 0.7×10⁻³ mm²/s in WM; NDI ∈ [0, 1] |
| Smoothing | Effective FWHM verification via `smoothest` |

---
---

*Last updated: June 2026 — SWEEP2 / DEWEY v7*
