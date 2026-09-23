#!/usr/bin/env python3
"""
RMS_movement.py — Plot eddy current motion parameters (translation & rotation)

Usage:
    python3.12 RMS_movement.py -i <eddy_parameters_file> -o <output_folder> [options]

Examples:
    python3.12 RMS_movement.py -i sub-01_ses-01.eddy_parameters -o /derivatives/sub-01/ses-01/dwi/eddy/
    python3.12 RMS_movement.py -i sub-01_ses-01.eddy_parameters -o ./plots/ -subjid sub-01_ses-01 -fps 20
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# Core function
# ============================================================
def do_RMS_movement_plots(path2rms: str, outpath: str, subjid: str = "", fps: int = 20) -> None:
    """
    Plot eddy current motion parameters (translation and rotation).

    Arguments:
        path2rms: path to eddy_parameters file (Nx16 whitespace-separated text)
        outpath:  output folder where plots/GIFs are saved
        subjid:   subject identifier used in plot titles (optional)
        fps:      frames per second for the animated GIFs
    """

    os.makedirs(outpath, exist_ok=True)
    if not outpath.endswith("/"):
        outpath += "/"

    title_prefix = f"{subjid} — " if subjid else ""

    # --- Load data ---
    rms = np.loadtxt(path2rms)

    # First 6 columns: translation (0-2) and rotation (3-5)
    trn = rms[:, 0:3]
    rtn = rms[:, 3:6]

    xtrn, ytrn, ztrn = trn[:, 0], trn[:, 1], trn[:, 2]
    xrtn, yrtn, zrtn = rtn[:, 0], rtn[:, 1], rtn[:, 2]

    n_frames = rms.shape[0]
    time     = np.arange(n_frames)

    # --------------------------------------------------------
    # Animated 3D GIFs
    # --------------------------------------------------------
    def _make_3d_anim(x, y, z, title, color, outfile):
        fig = plt.figure(figsize=(15, 7))
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_title(f"{title_prefix}{title}")
        ax.set_xlabel('X', labelpad=15)
        ax.set_ylabel('Y', labelpad=15)
        ax.set_zlabel('Z', labelpad=15)

        line,  = ax.plot([], [], [], color=color, alpha=0.8, label='Trajectory')
        point, = ax.plot([], [], [], 'ro')

        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_zlim(np.min(z), np.max(z))
        ax.legend()

        def update(frame):
            line.set_data(x[:frame],  y[:frame])
            line.set_3d_properties(z[:frame])
            point.set_data([x[frame]], [y[frame]])
            point.set_3d_properties([z[frame]])
            return line, point

        anim = FuncAnimation(fig, update, frames=n_frames, interval=800, blit=False)
        anim.save(outfile, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  ✅ Saved: {outfile}")

    _make_3d_anim(xtrn, ytrn, ztrn, "Translation Trajectory", "blue",    outpath + "translation_trajectory.gif")
    _make_3d_anim(xrtn, yrtn, zrtn, "Rotation Trajectory",    "magenta", outpath + "rotation_trajectory.gif")

    # --------------------------------------------------------
    # 2D line plots
    # --------------------------------------------------------
    def _make_2d_plot(t, channels, labels, colors, title, ylabel, outfile):
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_title(f"{title_prefix}{title}")
        ax.set_xlabel('Volume')
        ax.set_ylabel(ylabel)
        for ch, lbl, col in zip(channels, labels, colors):
            ax.plot(t, ch, label=lbl, color=col, alpha=0.8, linewidth=1.5)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print(f"  ✅ Saved: {outfile}")

    _make_2d_plot(
        time,
        [xtrn, ytrn, ztrn],
        ['X', 'Y', 'Z'],
        ['#6EB86E', '#008000', '#808000'],
        "Translation Over Volumes",
        "Translation (mm)",
        outpath + "translation_plot.png"
    )

    _make_2d_plot(
        time,
        [xrtn, yrtn, zrtn],
        ['X', 'Y', 'Z'],
        ['#33B9DE', '#1E7180', '#002D80'],
        "Rotation Over Volumes",
        "Rotation (degrees)",
        outpath + "rotation_plot.png"
    )


# ============================================================
# CLI
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="Plot eddy current motion parameters (translation & rotation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i",       "--input",   required=True,          help="Eddy parameters file (.eddy_parameters or similar)")
    parser.add_argument("-o",       "--output",  required=True,          help="Output folder for plots and GIFs")
    parser.add_argument("-subjid",  "--subjid",  default="",             help="Subject ID for plot titles (e.g. sub-01_ses-01)")
    parser.add_argument("-fps",     "--fps",     default=20, type=int,   help="Frames per second for animated GIFs")
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    print(f"→ Input:   {args.input}")
    print(f"→ Output:  {args.output}")
    print(f"→ Subject: {args.subjid or '(not set)'}")
    print(f"→ FPS:     {args.fps}\n")

    do_RMS_movement_plots(
        path2rms = args.input,
        outpath  = args.output,
        subjid   = args.subjid,
        fps      = args.fps
    )

    print("\n✅ All plots saved.")


if __name__ == "__main__":
    main()