#!/usr/bin/env python3.12
"""
@Manfre - Jun2026

dki.py — MP-PCA denoising (via MRtrix dwidenoise) + optional Rician noise rectification

Usage:
    # With noise rectification (requires phase):
    python dki.py -i <dwi.nii.gz> -m <mask.nii.gz> \
                  --bvec <file.bvec> --bval <file.bval> \
                  -p <phase.nii.gz> --do_rect \
                  -o <output_prefix>

    # Without noise rectification (complex denoising, magnitude output):
    python dki.py -i <dwi.nii.gz> -m <mask.nii.gz> \
                  --bvec <file.bvec> --bval <file.bval> \
                  -p <phase.nii.gz> \
                  -o <output_prefix>

Outputs (all .nii.gz):
    <prefix>_denoised.nii.gz        — MP-PCA denoised magnitude DWI
                                      (Rician-rectified if --do_rect)
    <prefix>_noise.nii.gz           — Noise sigma map from full dwidenoise
    <prefix>_noise_floor_est.nii.gz — Noise sigma map from highest b-shell only
                                      (only saved when --do_rect is active)
"""

import argparse
import os
import tempfile
import subprocess
import numpy as np
from numpy import pi

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from time import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="MRtrix dwidenoise + optional  noise rectification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-i", "--input",   required=True,       help="Input DWI NIfTI file (.nii or .nii.gz)")
    parser.add_argument("-p", "--phase",   required=True,       help="Input phase NIfTI file (.nii or .nii.gz)")
    parser.add_argument("-m", "--mask",    required=False,      help="Binary brain mask NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--bval",          required=True,       help="b-values file (FSL format)")
    parser.add_argument("--bvec",          required=True,       help="b-vectors file (FSL format)")
    parser.add_argument("-o", "--output",  required=True,       help="Output prefix (e.g. /out/sub-01_ses-01_AP)")
    parser.add_argument("--do_rect",       action="store_true", help="Perform noise rectification")
    parser.add_argument("--nthreads",      type=int, default=18,help="Threads for dwidenoise (default: 18)")
    return parser.parse_args()


def run(cmd, label=""):
    print(f"\n  [{label}] {cmd}" if label else f"\n  {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    #  Load data + gradients                                               #
    # ------------------------------------------------------------------ #
    print("\nLoading data...")
    data, affine = load_nifti(args.input)
    data = np.nan_to_num(data, nan=0.0).astype(np.float32)

    mask = None
    if args.mask:
        mask_data, _ = load_nifti(args.mask)
        mask = (mask_data > 0).astype(np.uint8)

    bvals, bvecs = read_bvals_bvecs(args.bval, args.bvec)

    print(f"  DWI shape    : {data.shape}")
    print(f"  b-values     : {np.unique(bvals)}")
    print(f"  Rectification: {'Y' if args.do_rect else 'N'}\n")

    out_denoised        = f"{args.output}_denoised.nii.gz"
    out_noise           = f"{args.output}_noise.nii.gz"
    out_noise_floor_est = f"{args.output}_noise_floor_est.nii.gz"

    mask_flag = f"-mask {args.mask}" if args.mask else ""
    
    b_range      = 20
    b_max        = np.max(bvals)
    floor_est_shells = np.logical_or( bvals == 0,np.logical_and(bvals < b_max - b_range, bvals < b_max + b_range)) #  Identify highest b-shell indices (for rectification sigma)         
  
    floor_est_indices = np.where(floor_est_shells)[0]
    print(f"  Highest b-shell: {b_max} | Floor est. on ({floor_est_shells.sum()} volumes, indices: {floor_est_indices.tolist()})")


    with tempfile.TemporaryDirectory() as tmp: #  All processing in a temp directory 

        phase_rad_tmp        = os.path.join(tmp, "phase_rad.nii.gz")
        complex_tmp          = os.path.join(tmp, "complex.nii.gz")
        denoised_tmp         = os.path.join(tmp, "denoised_complex.nii.gz")
        noise_tmp            = os.path.join(tmp, "noise.nii.gz")
        denoised_mag_tmp     = os.path.join(tmp, "denoised_mag.nii.gz")

        # ── Phase → radians ────────────────────────────────────────────
        run(f"mrcalc {args.phase} pi 4096 -div -mul {phase_rad_tmp} -force", label="phase→rad")

        # ── Magnitude + phase → complex ────────────────────────────────
        run(f"mrcalc {args.input} {phase_rad_tmp} -polar {complex_tmp} -force", label="mag+phase→complex")

        # ── dwidenoise on full complex data ────────────────────────────
        print("\n  Running dwidenoise on full complex data...")
        t = time()
        
        run(f"dwidenoise {complex_tmp} {denoised_tmp} -noise {noise_tmp} {mask_flag} -nthreads {args.nthreads} -force -debug", label="dwidenoise (full)")
        print(f"\n  dwidenoise done in {time() - t:.1f}s")

        # ── Complex → magnitude ────────────────────────────────────────
        run(f"mrcalc {denoised_tmp} -abs {denoised_mag_tmp} -force",
            label="complex→magnitude")

        # Load full denoised magnitude + full sigma
        denoised_mag, _ = load_nifti(denoised_mag_tmp)
        sigma_full, _   = load_nifti(noise_tmp)

        # ── Optional: highest-shell sigma for rectification ────────────
        if args.do_rect:

            # Extract highest-shell volumes (b0 + b_max) from complex image
            floor_est_tmp        = os.path.join(tmp, "complex_floor_est.nii.gz")
            denoised_hs_tmp      = os.path.join(tmp, "denoised_complex_floor_est.nii.gz")
            noise_hs_tmp         = os.path.join(tmp, "noise_floor_est.nii.gz")

            # mrconvert extracts volumes by index
            indices_str = ",".join(str(i) for i in floor_est_indices)
            run(f"mrconvert {complex_tmp} {floor_est_tmp} " f"-coord 3 {indices_str} -force", label="extract floor_est shell")

            print("\n  Running dwidenoise on floor_est b-shell (sigma estimation)...")
            t = time()
            run(f"dwidenoise {floor_est_tmp} {denoised_hs_tmp} -noise {noise_hs_tmp} {mask_flag} -nthreads {args.nthreads} -force", label="dwidenoise (floor_est shell)")
            print(f"\n  dwidenoise (floor_est shell) done in {time() - t:.1f}s")

            sigma_rect, _ = load_nifti(noise_hs_tmp)
            S_out = np.zeros_like(denoised_mag, dtype=np.float32)

            for i in range(denoised_mag.shape[3]):          # iterate over volume index
                vol = denoised_mag[..., i]                  # 3D volume (x, y, z) 
                S_out[..., i] = np.sqrt(np.maximum(vol**2 - (pi / 2) * sigma_rect**2, 0)).astype(np.float32)  

            # Save highest-shell sigma map
            save_nifti(out_noise_floor_est, sigma_rect.astype(np.float32), affine)
            print(f"  Saved floor_est sigma : {out_noise_floor_est}")

        else:
            S_out = denoised_mag.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Save final outputs                                                  #
    # ------------------------------------------------------------------ #
    save_nifti(out_denoised, S_out, affine)
    print(f"\n  Saved denoised  : {out_denoised}")

    save_nifti(out_noise, sigma_full.astype(np.float32), affine)
    print(f"  Saved noise map : {out_noise}\n")


if __name__ == "__main__":
    main()