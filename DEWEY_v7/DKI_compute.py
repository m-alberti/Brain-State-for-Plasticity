#!/usr/bin/env python3
"""
dki.py — DKI + MSDKI scalar map estimator using DIPY

Usage:
    python dki.py -i <dwi.nii.gz> -m <mask.nii.gz> \
                  --bvec <file.bvec> --bval <file.bval> \
                  -o <output_prefix>

Outputs (all .nii.gz):
    MSDKI:
        <prefix>_MSD.nii.gz    — Mean Signal Diffusivity
        <prefix>_MSK.nii.gz    — Mean Signal Kurtosis
        <prefix>_SMT2f.nii.gz  — SMT2 intra-neurite volume fraction
        <prefix>_SMT2Di.nii.gz — SMT2 intrinsic diffusivity
        <prefix>_SMT2uFA.nii.gz— SMT2 microscopic FA
    DKI:
        <prefix>_MK.nii.gz     — Mean Kurtosis
        <prefix>_MDk.nii.gz    — Mean Diffusivity (from DKI fit)
        <prefix>_FAk.nii.gz    — Fractional Anisotropy (from DKI fit)
"""

import argparse
import os
import sys
import numpy as np

from dipy.io.image import load_nifti, save_nifti
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
import dipy.reconst.msdki as msdki
import dipy.reconst.dki as dki_module


python_func = "/home/malberti/wks14/temp/FF_DWI_Drift/script/python_func"

sys.path.append(python_func)

from Tensor_plot import do_tensor_plot
from HomeMadeSlicesDir import do_HomeMadeSlicesDir 

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate MSDKI (and optionally DKI) scalar maps from DWI data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input DWI NIfTI file (.nii or .nii.gz)"
    )
    parser.add_argument(
        "-m", "--mask", required=True,
        help="Binary brain mask NIfTI file (.nii or .nii.gz)"
    )
    parser.add_argument(
        "--bvec", required=True,
        help="b-vectors file (FSL format)"
    )
    parser.add_argument(
        "--bval", required=True,
        help="b-values file (FSL format)"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output prefix (directory + basename, e.g. /out/sub-01_ses-01_AP)"
    )
    parser.add_argument(
        "--dki-fit-method", default="CWLS", choices=["OLS", "WLS", "UWLS", "NLLS", "CWLS"],
        help="Fitting method for DKI (default: C)"
    )
    return parser.parse_args()



def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    #  Load data                                                           #
    # ------------------------------------------------------------------ #
    print("\nLoading data...")
    data, affine = load_nifti(args.input)
    mask, _      = load_nifti(args.mask)

    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)

    bvals, bvecs = read_bvals_bvecs(args.bval, args.bvec)
    gtab = gradient_table(bvals, bvecs)

    print(f"  DWI shape : {data.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  b-values  : {np.unique(bvals)}")

    # Replace NaNs (common after eddy/signal-drift correction)
    data = np.nan_to_num(data, nan=0.0)

    # ------------------------------------------------------------------ #
    #  MSDKI & SMT2 fit                                                           #
    # ------------------------------------------------------------------ #
    print("\nFitting MSDKI & SMT2f model...")
    msdki_model = msdki.MeanDiffusionKurtosisModel(gtab)
    msdki_fit   = msdki_model.fit(data, mask=mask)

    metrics_msdki = {
        "MSD"     : msdki_fit.msd,
        "MSK"     : msdki_fit.msk,
       # "SMT2f"   : msdki_fit.smt2f,
       # "SMT2Di"  : msdki_fit.smt2di,
       # "SMT2uFA" : msdki_fit.smt2uFA,
    }

    print("Saving MSDKI maps:")
    for name, vol in metrics_msdki.items():
        out_path = f"{args.output}_{name}.nii.gz"
        save_nifti(out_path, vol.astype(np.float32), affine)
        print(f"  {out_path}")

          ## Save Q&A
        out_png  = os.path.dirname(args.output)   
        out_PNG = os.path.join(out_png, f"{os.path.basename(args.output)}_{name}")
        do_HomeMadeSlicesDir(out_path, out_png, out_PNG)
        do_tensor_plot(out_path, out_png, out_PNG) 

    # ------------------------------------------------------------------ #
    #  DKI fit                                                             #
    # ------------------------------------------------------------------ #
    print(f"\nFitting DKI model (method={args.dki_fit_method})...")
    dki_model = dki_module.DiffusionKurtosisModel(gtab, fit_method=args.dki_fit_method)
    dki_fit   = dki_model.fit(data, mask=mask)

    metrics_dki = {
        "MK" : dki_fit.mk(0, 3),   # clipped to [0, 3]
        "MDk": dki_fit.md,
        "FAk": dki_fit.fa,
    }

    print("Saving DKI maps:")
    for name, vol in metrics_dki.items():
        out_path = f"{args.output}_{name}.nii.gz"
       
        save_nifti(out_path, vol.astype(np.float32), affine)
        ## Save Q&A
        out_png  = os.path.dirname(args.output)   
        out_PNG = os.path.join(out_png, f"{os.path.basename(args.output)}_{name}")
        do_HomeMadeSlicesDir(out_path, out_png, out_PNG)
        do_tensor_plot(out_path, out_png, out_PNG) 

    print(f"\nDone: {os.path.basename(args.output)}.\n")


if __name__ == "__main__":
    main()