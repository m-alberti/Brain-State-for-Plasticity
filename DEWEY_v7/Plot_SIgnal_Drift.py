#!/usr/bin/env python3
"""
Plot b0 signal drift before and after correction.

Usage:
    python plot_signal_drift.py \
        --pre  sub-16_ses-01_dwi_eddy_corrected_ABC_noPA.nii \
        --post sub-16_ses-01_eddy-current_signal-drift_ABC_corr.nii \
        --bval sub-16_ses-01_dwi_eddy_corrected_ABC_noPA.bval \
        --bvec sub-16_ses-01_dwi_eddy_corrected_ABC_noPA.eddy_rotated_bvecs \
        --mask sub-16_ses-01_dwi_eddy_corrected_ABC_mask.nii \
        --out  /path/to/output/sub-16_ses-01_drift.png \
        --subjid sub-16_ses-01
"""

import argparse
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def pct_change(arr):
    """Percent change relative to the first volume."""
    arr = np.array(arr, dtype=float)
    return (arr / arr[0] - 1) * 100


def extract_b0s(dwi_path, bval_path, bvec_path, out_dir, b0_thresh=50):
    """Extract b0 volumes from a 4-D NIfTI and save them. Returns output path."""
    bvals = np.loadtxt(bval_path)
    b0_idx = np.where(bvals <= b0_thresh)[0]

    img  = nib.load(dwi_path)
    data = img.get_fdata()

    b0_data = data[..., b0_idx]
    os.makedirs(out_dir, exist_ok=True)

    basename = os.path.basename(dwi_path).replace(".nii.gz", "").replace(".nii", "")
    out_path = os.path.join(out_dir, f"{basename}_b0s.nii.gz")
    nib.save(nib.Nifti1Image(b0_data, img.affine, img.header), out_path)
    return out_path


def mean_in_mask(nii_path, mask):
    """Return per-volume mean inside mask, NaNs → 0."""
    data = nib.load(nii_path).get_fdata()
    data[np.isnan(data)] = 0
    return np.array([data[..., i][mask].mean() for i in range(data.shape[-1])])


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot b0 signal drift before / after correction."
    )
    p.add_argument("--pre",    required=True,  help="Pre-correction DWI NIfTI (.nii / .nii.gz)")
    p.add_argument("--post",   required=True,  help="Post-correction DWI NIfTI (.nii / .nii.gz)")
    p.add_argument("--bval",   required=True,  help=".bval file")
    p.add_argument("--bvec",   required=True,  help=".bvec / .eddy_rotated_bvecs file")
    p.add_argument("--mask",   required=True,  help="Brain mask NIfTI")
    p.add_argument("--out",    required=True,  help="Output path for the figure (e.g. /path/fig.png)")
    p.add_argument("--subjid", default="",     help="Subject ID label for plot title")
    p.add_argument("--vlines", nargs="*", type=int, default=[],
                   help="X positions for vertical reference lines (default: 21 42)")
    p.add_argument("--ylim",   nargs=2, type=float, default=[-3, 3],
                   help="Y-axis limits (default: -3 3)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

args   = parse_args()
tmpdir = os.path.join(os.path.dirname(args.out), "tmp_b0s")

# Load mask
mask = nib.load(args.mask).get_fdata().astype(bool)

# Extract b0s for pre and post
print(f"Extracting b0s  PRE  → {args.pre}")
pre_b0s  = extract_b0s(args.pre,  args.bval, args.bvec, tmpdir)

print(f"Extracting b0s  POST → {args.post}")
post_b0s = extract_b0s(args.post, args.bval, args.bvec, tmpdir)

# Mean signal per b0 volume
pre_means  = mean_in_mask(pre_b0s,  mask)
post_means = mean_in_mask(post_b0s, mask)

print(f"  PRE  b0 volumes : {len(pre_means)}")
print(f"  POST b0 volumes : {len(post_means)}")

# Percent change relative to first volume
pre_pc  = pct_change(pre_means)
post_pc = pct_change(post_means)

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(25, 5))

ax.plot(np.arange(len(pre_pc)),  pre_pc,
        marker="o", linewidth=2, markersize=4,
        color="#0E141F", linestyle="-",
        label=f"{args.subjid}  pre-correction")

ax.plot(np.arange(len(post_pc)), post_pc,
        marker="^", linewidth=2, markersize=4,
        color="#C44E52", linestyle=":",
        label=f"{args.subjid}  post-correction")

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

for x in args.vlines:
    ax.axvline(x=x, color="gray", linestyle="--", linewidth=1, alpha=0.7)

ax.set_xlabel("b0 Volume index", fontsize=13)
ax.set_ylabel("Signal change (%)", fontsize=13)
#ax.set_ylim(args.ylim)
ax.set_title(
    f"{args.subjid} — b0 Signal Drift  |  Pre vs Post correction\n"
    f"Normalised to first volume",
    fontsize=14
)
ax.legend(fontsize=10, framealpha=0.9, loc="upper left",
            bbox_to_anchor=(1.01, 1), borderaxespad=0)
ax.grid(True, alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
plt.savefig(args.out, dpi=250, bbox_inches="tight")
