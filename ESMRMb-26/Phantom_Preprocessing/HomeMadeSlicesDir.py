import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

## DIPY
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs


def do_HomeMadeSlicesDir(dwi, png_dir, outpng, bval=None, bvec=None):
    """
    Create slice collages (every 2 slices) for each b-shell defined in unique_bvals.
    Groups volumes based on b-value within ±50 tolerance.
    Only saves slices that have >100 non-zero voxels.
    Intensity is scaled based on the lowest and highest value of each volume.

    If bval/bvec are not provided, treats the input as a single 3D volume (or
    processes the first volume of a 4D image without b-shell filtering).
    """
    # --- load data ---
    print(f"Loading: {dwi}")
    data, affine = load_nifti(dwi)
    data = np.nan_to_num(data)
    step = 2

    outpath = os.path.join(png_dir, "HomeMadeSlicesDir")
    os.makedirs(outpath, exist_ok=True)

    # ------------------------------------------------------------------
    # CASE 1 — no bval/bvec: treat as single volume (or 4D without shells)
    # ------------------------------------------------------------------
    if bval is None or bvec is None:
        print("No bval/bvec provided — treating input as a single volume image.")

        # If 4D, use the first volume
        if data.ndim == 4:
            print(f"  4D image detected ({data.shape[3]} volumes) — using first volume.")
            vol = data[..., 0]
        else:
            vol = data  # already 3D

        _save_collage(vol, outpath, outpng, label="vol", step=step)

    # ------------------------------------------------------------------
    # CASE 2 — bval/bvec provided: filter by b-shell
    # ------------------------------------------------------------------
    else:
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        unique_bvals = [0, 500, 1000, 1500, 2000, 2500, 2800, 3000]  # Hard coded

        for val in unique_bvals:
            for dim in range(data.shape[3]):
                if val - 50 < bvals[dim] < val + 50:
                    vol = data[..., dim]
                    _save_collage(vol, outpath, outpng, label=f"b{int(val)}", step=step)
                    break  # stop after the first matching volume


def _save_collage(vol, outpath, outpng, label, step=2):
    """Build and save a slice collage for a single 3D volume."""

    # select slices with enough signal
    slice_idx = [
        i for i in range(0, vol.shape[2], step)
        if np.count_nonzero(vol[:, :, i]) > 100
    ]
    slices = [vol[:, :, i] for i in slice_idx]

    if not slices:
        print(f"  [{label}] No slices with sufficient signal — skipping.")
        return

    # grid layout
    n_slices = len(slices)
    cols = 6
    rows = math.ceil(n_slices / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.ravel()

    for i in range(rows * cols):
        ax = axes[i]
        if i < n_slices:
            ax.imshow(slices[i].T, cmap="gray", origin="lower")
            ax.set_title(f"z={slice_idx[i]}", fontsize=8)
        ax.axis("off")

    plt.tight_layout(pad=0.5)

    out_file = os.path.join(outpath, f"{outpng}_{label}.png")
    plt.savefig(out_file, bbox_inches="tight", pad_inches=0, dpi=800)
    plt.close()
    print(f"  Saved: {out_file}")


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate slice collage PNGs for each b-shell of a DWI dataset, "
            "or for a single structural/functional volume (no bval/bvec needed)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--dwi",
        required=True,
        metavar="DWI",
        help="Path to the input NIfTI file (.nii / .nii.gz).",
    )
    parser.add_argument(
        "-bval",
        default=None,
        metavar="BVAL",
        help="Path to the b-values file (.bval). Optional for single-volume images.",
    )
    parser.add_argument(
        "-bvec",
        default=None,
        metavar="BVEC",
        help="Path to the b-vectors file (.bvec). Optional for single-volume images.",
    )
    parser.add_argument(
        "-out", "--png_dir",
        required=True,
        metavar="PNG_DIR",
        help="Output directory where the 'HomeMadeSlicesDir' folder will be created.",
    )
    parser.add_argument(
        "-outpng",
        required=True,
        metavar="OUTPNG",
        help="Base name for output PNG files (e.g. 'sub-01' → sub-01_vol.png or sub-01_b0.png …).",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Warn if only one of bval/bvec is provided
    if (args.bval is None) != (args.bvec is None):
        parser.error("Please provide both -bval and -bvec, or neither.")

    do_HomeMadeSlicesDir(
        dwi=args.dwi,
        bval=args.bval,
        bvec=args.bvec,
        png_dir=args.png_dir,
        outpng=args.outpng,
    )
    print("Done.")


if __name__ == "__main__":
    main()