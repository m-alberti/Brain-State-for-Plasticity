#!/usr/bin/env python3
"""
dki.py — MP-PCA denoising + optional Rician noise rectification using DIPY

Usage:
    # With noise rectification (requires phase):
    python dki.py -i <dwi.nii.gz> -m <mask.nii.gz> \
                  --bvec <file.bvec> --bval <file.bval> \
                  -p <phase.nii.gz> --do_rect \
                  -o <output_prefix>

    # Without noise rectification (magnitude only):
    python dki.py -i <dwi.nii.gz> -m <mask.nii.gz> \
                  --bvec <file.bvec> --bval <file.bval> \
                  -o <output_prefix>

Outputs (all .nii.gz):
    <prefix>_denoised.nii.gz        — MP-PCA denoised magnitude DWI
                                      (Rician-rectified if --do_rect and -p provided)
    <prefix>_sigma.nii.gz           — Noise sigma map from highest b-shell MP-PCA
                                      (only saved when --do_rect is active)
"""

import argparse
import numpy as np
from numpy import pi

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.denoise.localpca import mppca
from time import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="MP-PCA denoising with optional Rician noise rectification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-i", "--input",   required=True,              help="Input DWI NIfTI file (.nii or .nii.gz)")
    parser.add_argument("-p", "--phase",   required=True,             help="Input phase NIfTI file (.nii or .nii.gz) — required for --do_rect")
    parser.add_argument("-m", "--mask",    required=False,             help="Binary brain mask NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--bval",          required=True,              help="b-values file (FSL format)")
    parser.add_argument("--bvec",          required=True,              help="b-vectors file (FSL format)")
    parser.add_argument("-o", "--output",  required=True,              help="Output prefix (e.g. /out/sub-01_ses-01_AP)")
    parser.add_argument("-s", "--suffix",  required=False,              help="Output prefix (e.g. /out/sub-01_ses-01_AP)")
    parser.add_argument("--do_rect",       action="store_true",        help="Perform Rician noise rectification (requires -p/--phase)")
    return parser.parse_args()


def main():
    args = parse_args()

    #  Load data  
    print("\nLoading data...")
    data, affine = load_nifti(args.input)
    data = np.nan_to_num(data, nan=0.0).astype(np.float32)

    mask = None
    if args.mask:
        mask_data, _ = load_nifti(args.mask)
        mask = (mask_data > 0).astype(np.uint8)

    if args.phase:
        phase, _ = load_nifti(args.phase)
        rad_phase = phase * (pi / 4096)   # Siemens [-4096, 4096] → radians


    bvals, bvecs = read_bvals_bvecs(args.bval, args.bvec)

    print(f"  DWI shape : {data.shape}")
    if mask is not None:
        print(f"  Mask shape: {mask.shape}")
    
    print(f"  b-values  : {np.unique(bvals)}")
    print(f"  Rectification: {'Y' if args.do_rect else 'N'}\n")


    patch_size = 3 if data.shape[-1] > 343 else 2   # radius → 7*7*7 or 5*5*5 patch size heuristic (matches MRtrix3 convention)

    #  Noise rectification 
    # ================================================================== #
    if args.do_rect:

        # #stimate sigma from highest b-shell
        b_max = np.max(bvals)
        b_range = 20
        sel_b = np.logical_or(bvals == 0, np.logical_and(bvals > b_max - b_range, bvals < b_max + b_range))
        data_highest_shell = data[..., sel_b]
        
        print(f"  Highest b-val shell: {b_max}  ({sel_b.sum()} volumes)")
        print("  Estimating sigma from highest shell (MP-PCA)...")
        
        t = time()
        _, sigma = mppca(data_highest_shell, patch_radius=2, return_sigma=True)
        
        print(f"  Sigma estimated in {time() - t:.1f}s\n")

        # MP-PCA on full complex data
        print("  Building complex data and running MP-PCA...")
        complex_data = data * np.exp(1j * rad_phase)
        t = time()
        denoised_complex, sigma_ = mppca(complex_data, patch_radius=patch_size, return_sigma=True)
        print(f"  Complex MP-PCA done in {time() - t:.1f}s")

        # Back to magnitude
        denoised_mag = np.abs(denoised_complex)

        # Rician rectification: S = sqrt( max(S²  - π/2·σ², 0) )
        print("  Performing Rician noise rectification...")
        S_out = np.sqrt(np.maximum(denoised_mag**2 - (pi / 2) * sigma**2, 0))


    #  Magnitude-only MP-PCA, no rectification
    # ================================================================== #
    else:
        
        # MP-PCA on full complex data
        print("  Building complex data and running MP-PCA...")
        complex_data = data * np.exp(1j * rad_phase)
        t = time()
        denoised_complex, sigma_ = mppca(complex_data, patch_radius=patch_size, return_sigma=True)
        print(f"  Complex MP-PCA done in {time() - t:.1f}s")

        # Back to magnitude
        S_out = np.abs(denoised_complex)


    # Save sigma map
    sigma_path = f"{args.output}_noise_nii.gz"
    save_nifti(sigma_path, sigma_.astype(np.float32), affine)
    print(f"  Saved sigma map : {sigma_path}")


    #  Save denoised output                                                #
    # ================================================================== #
    out_path = f"{args.output}_denoised.nii.gz"
    save_nifti(out_path, S_out.astype(np.float32), affine)
    print(f"  Saved denoised  : {out_path}\n")


if __name__ == "__main__":
    main()