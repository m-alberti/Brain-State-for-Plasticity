#!/usr/bin/env python3
"""
noddi.py — NODDI scalar map estimator using AMICO

Usage:
    python noddi.py -i <dwi.nii.gz> -m <mask.nii.gz> \
                    --bvec <file.bvec> --bval <file.bval> \
                    -o <output_prefix> \
                    [--dPar 0.0011] [--dIso 0.003]

Outputs (all .nii.gz, renamed from AMICO defaults):
    <prefix>_NDI.nii.gz           — Neurite Density Index
    <prefix>_ODI.nii.gz           — Orientation Dispersion Index
    <prefix>_FWF.nii.gz           — Free Water Fraction
    <prefix>_NDI_modulated.nii.gz — NDI modulated by (1 - FWF)
    <prefix>_ODI_modulated.nii.gz — ODI modulated by (1 - FWF)
    <prefix>_NRMSE.nii.gz         — Normalised Root Mean Square Error
    <prefix>_dir.nii.gz           — Fibre orientation (peaks)

Notes:
    - dPar: intrinsic parallel diffusivity [mm²/s]  (WM ≈ 0.0017, GM ≈ 0.0011)
    - dIso: isotropic (CSF) diffusivity [mm²/s]     (default 0.003)
    - AMICO requires CWD = output directory; the script handles this automatically.
    - The AMICO AMICO/ subfolder is removed after renaming outputs.
"""

import argparse
import os
import sys
import subprocess
import shutil

import amico


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate NODDI scalar maps from DWI data using AMICO.", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,)
    
    parser.add_argument("-i", "--input",    required=True,      help="Input DWI NIfTI file (.nii or .nii.gz)")
    parser.add_argument("-m", "--mask",     required=True,      help="Binary brain mask NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--bvec",           required=True,      help="b-vectors file (FSL format)")
    parser.add_argument("--bval",           required=True,      help="b-values file (FSL format)")
    parser.add_argument("-o", "--output",   required=True,      help="Path + prefix output")
    parser.add_argument("--dPar", type=float, default=0.0011,   help="Intrinsic parallel diffusivity [mm²/s] (default: 0.0011 for GM; use 0.0017 for WM)")
    parser.add_argument("--dIso", type=float, default=0.003,    help="Isotropic (CSF) diffusivity [mm²/s] (default: 0.003)")
    parser.add_argument("--bStep", type=float, default=0,       help="b-value rounding step for scheme file (default: 0 = no rounding)")
    parser.add_argument("--b0-thr", type=float, default=0, dest="b0_thr",       help="b-value threshold to identify b0 volumes (default: 0)")
    parser.add_argument("--replace-bad-voxels", type=float, default=0.0, dest="replace_bad_voxels",     help="Value used to replace bad voxels in AMICO (default: 0.0)")
    parser.add_argument("--no-nrmse", action="store_true",      help="Disable NRMSE computation (faster fitting)")
    parser.add_argument("--no-modulated", action="store_true",  help="Disable saving of modulated maps (NDI_modulated, ODI_modulated)")
    return parser.parse_args()


# AMICO always names its internal result folder after the *basename* of the
# DWI file (minus extension).  We need to know that name to rename outputs.
def _amico_result_dir(out_dir: str, prefix: str) -> str:
    """Return the path AMICO creates: <out_dir>/AMICO/NODDI_<prefix>/"""
    return os.path.join(out_dir, "AMICO", f"NODDI_{prefix}")


def _rename_outputs(amico_result_dir: str, out_dir: str, prefix: str, suffixes: list[str]) -> None:  #Amico gives random names, so i just rename them
    """Move AMICO fit_*.nii.gz to <out_dir>/<prefix>_*.nii.gz."""
    for suffix in suffixes:
        src = os.path.join(amico_result_dir, f"fit_{suffix}.nii.gz")
        dst = os.path.join(out_dir, f"{prefix}_{suffix}.nii.gz")
        shutil.move(src, dst)
        print(f"  saved → {dst}")

#  Main


def main():
    args = parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────
    inDWIs   = os.path.abspath(args.input)
    in_mask  = os.path.abspath(args.mask)
    in_bvec  = os.path.abspath(args.bvec)
    in_bval  = os.path.abspath(args.bval)

    out_dir    = os.path.abspath(os.path.dirname(args.output))
    out_prefix = os.path.basename(args.output)          # e.g. sub-01_ses-01_AP

    schemefile = os.path.join(out_dir, "NODDI_protocol.scheme")

    os.makedirs(out_dir, exist_ok=True)

    # ── Sanity checks ──────────────────────────────────────────────────────
    for label, path in [("DWI", inDWIs), ("mask", in_mask),
                        ("bvec", in_bvec), ("bval", in_bval)]:
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {label} file not found: {path}")

    print("=" * 60)
    print(f"  NODDI fitting — {out_prefix}")
    print("=" * 60)
    print(f"  Input DWI   : {inDWIs}")
    print(f"  Mask        : {in_mask}")
    print(f"  BVECs       : {in_bvec}")
    print(f"  BVALs       : {in_bval}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Prefix      : {out_prefix}")
    print(f"  dPar        : {args.dPar} mm²/s")
    print(f"  dIso        : {args.dIso} mm²/s")
    print("=" * 60)

    # ── Binarise mask (AMICO / DIPY requirement) ───────────────────────────
    print("\nBinarising mask...")
    subprocess.run(f"fslmaths {in_mask} -bin {in_mask}", shell=True, check=True)

    # ── AMICO setup ────────────────────────────────────────────────────────
    amico.setup()
    ae = amico.Evaluation()

    ae.set_config("doComputeNRMSE",       not args.no_nrmse)
    ae.set_config("doSaveModulatedMaps",  not args.no_modulated)

    # ── AMICO requires CWD = output directory ──────────────────────────────
    os.chdir(out_dir)

    
    # ── Build scheme file ──────────────────────────────────────────────
    print("\nBuilding scheme file...")
    amico.util.fsl2scheme(in_bval, in_bvec, schemefile, bStep=args.bStep)

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading DWI data into AMICO...")
    ae.load_data(inDWIs,schemefile,mask_filename=in_mask,replace_bad_voxels=args.replace_bad_voxels,b0_thr=args.b0_thr)

    # ── Configure NODDI model ──────────────────────────────────────────
    ae.set_model("NODDI")
    ae.model.dPar = args.dPar
    ae.model.dIso = args.dIso

    # ── Fit ────────────────────────────────────────────────────────────
    print("Generating kernels...")
    ae.generate_kernels(regenerate=True)
    ae.load_kernels()

    print("Fitting NODDI model...")
    ae.fit()
    ae.save_results(path_suffix=out_prefix)


    # ── Rename outputs ─────────────────────────────────────────────────────
    # Decide which suffixes to expect based on config flags
    NODDI_suffixes = ["dir", "FWF", "NDI", "NRMSE", "ODI"]
    
    if not args.no_modulated:
        NODDI_suffixes += ["NDI_modulated", "ODI_modulated"]
    
    if args.no_nrmse:
        NODDI_suffixes.remove("NRMSE")

    amico_result_dir = _amico_result_dir(out_dir, out_prefix)

    print("\nRenaming AMICO outputs...")
    _rename_outputs(amico_result_dir, out_dir, out_prefix, NODDI_suffixes)

    print(f"\nDone: {out_prefix}\n")

if __name__ == "__main__":
    main()