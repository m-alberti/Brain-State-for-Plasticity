import os
import numpy as np
import matplotlib.pyplot as plt
import math
## DIPY 
# nifti manipualtion 
from dipy.io.image import load_nifti, save_nifti

""" 
This function is used to read a scalar map. (I generally use the MD for this) and create a slicesdir-like output and a histogram of voxel intensity. This is done without a range; you may need to adjust it.
The function only takes into account voxels > 0 and slices with more than 100 voxels > 0.
SlicesDire: select one slice for each two volumes.
"""

def do_tensor_plot(scalar_maps, outdir, outpng): 
    
    scalar, affine=load_nifti(scalar_maps)
    data = np.nan_to_num(scalar)

    step =2 # <------ Here to change the number of slice
    outpath=os.path.join(outdir, 'Tensor_png')
    os.makedirs(outpath, exist_ok=True)

    slice_idx = [i for i in range(0, data.shape[2], step) if np.count_nonzero(data[:, :, i]) > 100]   # 100 is a random voxel
    if not slice_idx:
        print("⚠️ No valid slices found (less than 100 nonzero voxels per slice).")
        return

    slices = [data[:, :, i] for i in slice_idx]
    
    nonzero_data = data[data > 0]
    vmin, vmax=np.percentile(nonzero_data, [1, 99])  # robust scaling
    # "collage set up" - just change the number of columns to change the organization 
    # --> ASK chat gpt to do it

    n_slices = len(slices)
    cols = 6
    rows = math.ceil(n_slices / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2),facecolor='white')
    axes = axes.ravel()

    for i in range(rows * cols):
        ax = axes[i]
        if i < n_slices:
            ax.imshow(slices[i].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(f"z={slice_idx[i]}", fontsize=8)
        ax.axis('off')

    plt.tight_layout(pad=0.5)
    out_file_slices = os.path.join(outpath,f"{outpng}_slices.png")
    plt.savefig(out_file_slices, bbox_inches='tight', pad_inches=0, dpi=800)
    plt.close()

    # PLot values ....... 

 
    plt.figure(figsize=(8, 6))
    plt.hist(nonzero_data, bins=5000, alpha=0.6, color='teal', density=True) # Teal ==> light blue/green roughly 
    plt.xlabel("DWI idx")
    plt.ylabel("Frequency")
    plt.title("MD")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_file_hist = os.path.join(outpath,f"{outpng}_hist.png")
    plt.savefig(out_file_hist, bbox_inches='tight', pad_inches=0, dpi=800)
    plt.close()
