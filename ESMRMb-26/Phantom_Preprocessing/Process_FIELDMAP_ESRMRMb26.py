import subprocess
import os
import getpass
from datetime import datetime
import sys
import socket
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from time import time
import nibabel as nib
import glob as gl
import math
import time

# ──────────────────────────────────────────────────────────────────────────────
# SET WORKING DIRECTORY
# ──────────────────────────────────────────────────────────────────────────────
prj_path = r"/home/malberti/wks14/temp/FF_DWI_Drift"
project  = "sub-"
script   = r'/home/malberti/wks14/temp/FF_DWI_Drift/Script'
vps      = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

DELTA_TE_MS = 2.65          # ΔTE in milliseconds (TE2 - TE1)
MAG_THRESHOLD = 150         # Intensity threshold for brain mask from magnitude

# ── Toggle this to open fsleyes for mask QC before continuing ─────────────────
SHOW_MASK_QC = False         # Set to False to skip the visual check

METRICS = {
    "Tensor":    ["FA", "MD"],
    "Kurtosis":  ["MK", "kMD", "kFA"],
    "NODDI":     ["NDI", "ODI", "FWF"],
    "FreeWater": ["fw", "fwMD", "fwFA"],
}

# ──────────────────────────────────────────────────────────────────────────────
def run(cmd: str):
    """Run a shell command, print it, raise on failure."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}):\n  {cmd}")

def mask_qc(mag: str, mask: str):
    """
    Open fsleyes to inspect the mask overlaid on the magnitude image.
    Pauses execution until the user confirms to continue.
    Remove this function (or set SHOW_MASK_QC=False) to skip.
    """
    print("\n  🔍 Opening fsleyes for mask QC — close the window or press Enter to continue...")
    proc = subprocess.Popen(
        f"fsleyes {mag} {mask} -cm red -a 50",
        shell=True,
    )
    input("     Press Enter to continue after reviewing the mask... ")
    proc.terminate()

# ──────────────────────────────────────────────────────────────────────────────
for vp in vps:
    id_str = f"sub-{vp:02d}"
    raw_sub_path = os.path.join(prj_path, "derivatives", id_str)
    sessions = sorted([
        s for s in os.listdir(raw_sub_path)
        if s.startswith("ses-") and os.path.isdir(os.path.join(raw_sub_path, s))
    ])

    for session in sessions:
        subjid      = f"{id_str}_{session}"
        LOGsOut     = os.path.join(script, "LOGs", f"{id_str}-LOGs")
        LOGs_file   = os.path.join(script, "LOGs", f"{id_str}-LOGs", f"{subjid}-LOGs.txt")
        derivatives = os.path.join(prj_path, "derivatives", id_str, session)
        rawdata     = os.path.join(prj_path, "rawdata",     id_str, session)
        fmap        = os.path.join(rawdata,  "fmap")
        fmap_out    = os.path.join(derivatives, "fmap")

        os.makedirs(fmap_out, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"  {subjid}  —  fieldmap preparation")
        print(f"{'='*65}")

        # ── Input files ───────────────────────────────────────────────────────
        raw_phase = os.path.join(fmap, f"{subjid}_phasediff.nii.gz")
        mag_te1   = os.path.join(fmap, f"{subjid}_magnitude1.nii.gz")
        mag_te2   = os.path.join(fmap, f"{subjid}_magnitude2.nii.gz")

        # ── Output files ──────────────────────────────────────────────────────
        phase_rad        = os.path.join(fmap_out, f"{subjid}_phasediff-rad.nii.gz")
        mask             = os.path.join(fmap_out, f"{subjid}_magnitude1-mask.nii.gz")
        phase_masked     = os.path.join(fmap_out, f"{subjid}_phasediff-rad-masked.nii.gz")
        phase_unwrapped  = os.path.join(fmap_out, f"{subjid}_phasediff-unwrapped.nii.gz")
        fieldmap_rads    = os.path.join(fmap_out, f"{subjid}_fieldmap-rads.nii.gz")

        # ── 1. Convert phase from [-4096, 4096] to radians [-π, π] ───────────
        print("\n  [1/5] Converting phase to radians...")
        run(f"fslmaths {raw_phase} -div 4096 -mul 3.14159265 {phase_rad}")

        # ── 2. Create brain mask from magnitude TE1 ───────────────────────────
        print("\n  [2/5] Creating brain mask from magnitude TE1...")
        run(f"fslmaths {mag_te1} -thr {MAG_THRESHOLD} -bin {mask}")

        # ── Optional: QC mask in fsleyes before continuing ────────────────────
        if SHOW_MASK_QC:
            mask_qc(mag_te1, mask)

        # ── 3. Apply mask to phase ────────────────────────────────────────────
        print("\n  [3/5] Applying mask to phase...")
        run(f"fslmaths {phase_rad} -mas {mask} {phase_masked}")

        # ── 4. Unwrap phase with PRELUDE ──────────────────────────────────────
        # Use prelude directly since phase is already in radians.
        # fsl_prepare_fieldmap SIEMENS would double-scale the data.
        print("\n  [4/5] Unwrapping phase with prelude...")
        run(f"prelude -a {mag_te2} -p {phase_masked} -m {mask} -o {phase_unwrapped}")

        # ── 5. Convert unwrapped phase [rad] → fieldmap [rad/s] ──────────────
        delta_te_s = DELTA_TE_MS / 1000.0
        print(f"\n  [5/5] Computing fieldmap (÷ ΔTE={DELTA_TE_MS}ms = {delta_te_s}s)...")
        run(f"fslmaths {phase_unwrapped} -div {delta_te_s} {fieldmap_rads}")

        print(f"\n  ✅  Fieldmap written → {fieldmap_rads}")

print(f"\n{'='*65}")
print("✅  All done.")
print(f"{'='*65}\n")
