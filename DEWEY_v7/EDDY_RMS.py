#!/usr/bin/env python3
"""
EDDY_RMS.py — Plot eddy current movement RMS values

Usage:
    python3.12 EDDY_RMS.py -i <eddy_movement_rms_file> -o <output_folder> [options]

Examples:
    python3.12 EDDY_RMS.py -i sub-01_ses-01.eddy_movement_rms -o /derivatives/sub-01/ses-01/dwi/eddy/plots/
    python3.12 EDDY_RMS.py -i sub-01_ses-01.eddy_restricted_movement_rms -o ./plots/ -subjid sub-01_ses-01
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Core function
# ============================================================
def do_EDDY_RMS(Mrms_file: str, outpath: str, subjid: str = "") -> None:
    """
    Plot movement RMS values from eddy current output.

    Arguments:
        Mrms_file: path to .eddy_movement_rms or .eddy_restricted_movement_rms file
        outpath:   output folder where the plot is saved
        subjid:    subject identifier used in plot title (optional)
    """

    os.makedirs(outpath, exist_ok=True)

    # --- Output filename mirrors input filename ---
    base   = os.path.basename(Mrms_file)
    no_ext = os.path.splitext(base)[0]       # strips last extension only
    out    = os.path.join(outpath, f"{no_ext}_RMS.png")

    title_prefix = f"{subjid} — " if subjid else ""

    # --- Load & plot ---
    Mrms = np.loadtxt(Mrms_file)
    trend_line = np.poly1d(
        np.polyfit(np.arange(len(Mrms)), Mrms[:, 0], deg=1)
    )(np.arange(len(Mrms)))

    plt.figure(figsize=(25, 5))
    plt.plot(np.arange(len(Mrms)), trend_line,  color='black', linestyle='--', linewidth=2,  label='Trend (Abs RMS)')
    plt.plot(Mrms[:, 1], marker='o', linestyle='-', color='red',  alpha=0.6, label='Δ RMS (between volumes)')
    plt.plot(Mrms[:, 0], marker='o', linestyle='-', color='blue', alpha=0.6, label='Abs RMS')

    plt.title(f"{title_prefix}Movement RMS Over Time")
    plt.xlabel('Volume Index')
    plt.ylabel('mm')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    print(f"  ✅ Saved: {out}")


# ============================================================
# CLI
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="Plot eddy current movement RMS values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i",      "--input",   required=True, help="Eddy movement RMS file (.eddy_movement_rms or .eddy_restricted_movement_rms)")
    parser.add_argument("-o",      "--output",  required=True, help="Output folder for the plot")
    parser.add_argument("-subjid", "--subjid",  default="",    help="Subject ID for plot title (e.g. sub-01_ses-01)")
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    print(f"→ Input:   {args.input}")
    print(f"→ Output:  {args.output}")
    print(f"→ Subject: {args.subjid or '(not set)'}\n")

    do_EDDY_RMS(
        Mrms_file = args.input,
        outpath   = args.output,
        subjid    = args.subjid
    )


if __name__ == "__main__":
    main()