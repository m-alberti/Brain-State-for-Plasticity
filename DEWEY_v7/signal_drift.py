#!/usr/bin/env python3.12
"""
signal_drift.py — Spatiotemporal signal drift correction (Hansen et al. 2018)
https://doi.org/10.1016/j.mri.2018.11.009

Usage:
    python3.12 signal_drift.py -i <dwi.nii.gz> -bval <file.bval> -o <outbase> -roi <mask.nii.gz> [options]

Examples:
    python3.12 signal_drift.py -i eddy.nii.gz -bval dwi.bval -o drift/sub-01_drift -roi mask.nii.gz
    python3.12 signal_drift.py -i eddy.nii.gz -bval dwi.bval -o drift/sub-01_drift -roi mask.nii.gz -order 1
"""

import argparse
import os
import sys
import numpy as np
import json
import numpy.typing as npt
import nibabel as nib


# =============================================================
# IVIM fucntions
# =============================================================


def data_from_file(
    im_file: str,
    bval_file: str,
    bvec_file: str | None = None,
    cval_file: str | None = None,
    roi_file: str | None = None,
) ->tuple[
    npt.NDArray[np.float64],        # data  (always present)
    npt.NDArray[np.float64],        # bvals (always present)
    npt.NDArray[np.float64] | None, # bvecs (None if bvec_file not given)
    npt.NDArray[np.float64] | None, # cvals (None if cval_file not given)
    npt.NDArray[np.float64] | None, # roi   (None if roi_file  not given)
]: # type: ignore
    """
    Load image data (optionally from an ROI/mask) into a 2D array along with b-values, b-vectors, c-values.

    Arguments:
        im_file:   path to nifti image file
        bval_file: path to .bval file
        bvec_file: (optional) path to .bvec file
        cval_file: (optional) path to .cval file
        roi_file:  (optional) path to nifti file defining a region-of-interest (ROI) from with data is extracted

    Output:
        Y:         2D array containing the image data [number of voxels x number of b-values]
        b:         vector containing the b-values
        v:         matrix containing the diffusion encoding directions
        c:         vector containgin the c-values (flow encoding)
    """

    im = read_im(im_file)

    if im.ndim == 3:
        nb = 1
    elif im.ndim != 4:
        raise ValueError(
            f"The image must have four dimensions, or in expections three, but loaded image had {im.ndim}."
        )
    else:
        nb = im.shape[3]

    b = read_bval(bval_file)
    check_vector(b, nb, "b-values")

    if cval_file is not None:
        c = read_cval(cval_file)
        check_vector(c, nb, "c-values")

    if bvec_file is not None:
        v = read_bvec(bvec_file)
        check_2dmatrix(v, nb, "b-vectors")

    if roi_file is not None:
        roi = nib.load(roi_file).get_fdata().astype(bool)
        check_roi(roi, im.shape[:3])
    else:
        roi = np.full(im.shape[:-1], True)

    Y = im[roi, :]

    if cval_file is None:
        if bvec_file is None:
            return Y, b
        else:
            return Y, b, v
    else:
        if bvec_file is None:
            return Y, b, c
        else:
            return Y, b, v, c


def file_from_data(
    filename: str,
    data: npt.NDArray[np.float64],
    roi: npt.NDArray[np.bool_] | None = None,
    imref_file: str | None = None,
) -> None:
    """
    Save image data in Nifti format based on input 2D array originating from optional roi.

    Arguments:
        filename:   path to Nifti file
        data:       image data in v x n array (v is number of voxels, n gives size of 4th dimension in final image)
        roi:        (optional) region-of-interest from which data is assumed to originate. The number of True elements must match first dimension of data
        imref_file: (optional) path to nifti file from which header info is obtained
    """

    if imref_file is not None:
        nii_ref = nib.load(imref_file).get_fdata()
        if roi is None:
            roi = np.full(nii_ref.shape[:3], True)
    else:
        if roi is None:
            raise ValueError(
                "Either roi or imref_file must be specified to derive the size of the image."
            )
    roi = roi.astype(bool)
    sz = roi.shape

    if data.ndim > 1:
        im = np.full(list(sz) + [data.shape[1]], np.nan)
        im[roi, :] = data
    else:
        im = np.full(sz, np.nan)
        im[roi] = data

    write_im(filename, im, imref_file=imref_file)


def read_im(filename: str | None) -> npt.NDArray[np.float64] | None:
    """
    Load image in Nifti format.

    Arguments:
        filename: path to image file

    Output:
        im:       image (or None if filename is None)
    """

    if filename is None:
        im = None
    else:
        im = nib.load(filename).get_fdata()
    return im


def read_bval(filename: str) -> npt.NDArray[np.float64]:
    """
    Load b-values from file in FSL format.

    Arguments:
        filename: path to cval file

    Output:
        b:        b-values
    """

    b = np.atleast_1d(np.loadtxt(filename))
    return b


def read_cval(filename: str) -> npt.NDArray[np.float64]:
    """
    Load c-values from file in format similar to FSL bval format.

    Arguments:
        filename: path to cval file

    Output:
        c:        c-values
    """

    c = np.atleast_1d(np.loadtxt(filename))
    return c


def read_time(filename: str) -> npt.NDArray[np.float64]:
    """
    Load time parameter from file in format similar to FSL bval format.

    Arguments:
        filename: path to time parameter file

    Output:
        t:        time parameter
    """

    t = np.atleast_1d(np.loadtxt(filename))
    return t


def read_k(filename: str) -> npt.NDArray[np.float64]:
    """
    Load k (for intermediate regime) from file in format similar to FSL bval format.

    Arguments:
        filename: path to k file

    Output:
        k:        k (+/- 1)
    """

    k = np.atleast_1d(np.loadtxt(filename))
    return k


def read_bvec(filename: str) -> npt.NDArray[np.float64]:
    """
    Load encoding directions from file in FSL bvec format.

    Arguments:
        filename: path to bvec file

    Output:
        v:        encoding directions
    """

    v = np.loadtxt(filename)
    if v.ndim < 2:  # a single encoding direction
        v = v[:, np.newaxis]
    return v


def write_im(
    filename: str, im: npt.NDArray[np.float64], imref_file: str | None = None
) -> None:
    """
    Save image in Nifti format.

    Arguments:
        filename:   path to nifti file
        im:         image to save
        imref_file: (optional) path to nifti file from which header info is obtained
    """

    if imref_file is not None:
        nii_ref = nib.load(imref_file)
        nii = nib.Nifti1Image(im, affine=nii_ref.affine, header=nii_ref.header)
    else:
        nii = nib.Nifti1Image(im, affine=np.eye(4))
    nib.save(nii, filename)


def write_bval(filename: str, b: npt.NDArray[np.float64]) -> None:
    """
    Save encoding directions to file in FSL bval format.

    Arguments:
        filename: path to file
        b:        b-values
    """

    np.savetxt(filename, b, fmt="%.1f", newline=" ")


def write_cval(filename: str, c: npt.NDArray[np.float64]) -> None:
    """
    Save encoding directions to file in format similar to FSL bval.

    Arguments:
        filename: path to file
        c:        c-values
    """

    np.savetxt(filename, c, fmt="%.3f", newline=" ")


def write_time(filename: str, t: npt.NDArray[np.float64]) -> None:
    """
    Save time parameter to file in format similar to FSL bval.

    Arguments:
        filename: path to file
        t:        time parameter
    """

    np.savetxt(filename, t, fmt="%.5f", newline=" ")


def write_k(filename: str, k: npt.NDArray[np.float64]) -> None:
    """
    Save k to file in format similar to FSL bval.

    Arguments:
        filename: path to file
        k:        k
    """

    np.savetxt(filename, k, fmt="%.0f", newline=" ")


def write_bvec(filename: str, v: npt.NDArray[np.float64]) -> None:
    """
    Save encoding directions to file in FSL bvec format.

    Arguments:
        filename: path to file
        v:        3 x n array with diffusion encoding directions
    """
    if v.ndim < 2:
        v = v[:, np.newaxis]
    check_2dmatrix(v, v.size // 3, "bvec")
    np.savetxt(filename, v, fmt="%.3f")


def check_vector(v: npt.NDArray[np.float64], n: int, name: str) -> None:
    if v.ndim != 1:
        raise ValueError(f"The {name} must be ordered as a 1-dimensional vector.")
    if v.size != n:
        raise ValueError(
            f"The number of {name} must match the size of the fourth dimension of the nifti file."
        )


def check_2dmatrix(M: npt.NDArray[np.float64], n: int, name: str) -> None:
    if M.ndim != 2:
        raise ValueError(f"The {name} must be ordered as a 2-dimensional vector.")
    if M.shape[1] != n:
        raise ValueError(
            f"The size of the 2nd dimesion of the {name} must match the size of the fourth dimension of the nifti file."
        )
    if M.shape[0] != 3:
        raise ValueError(f"The size of the 1st dimesion of the {name} must be three.")


def check_roi(roi: npt.NDArray[np.float64], sz: npt.NDArray[np.int_]) -> None:
    if roi.ndim != 3:
        raise ValueError("The ROI must be a 3D array.")
    if np.any(roi.shape != sz):
        raise ValueError(
            "The shape of the ROI must match that of the first three dimension of the image."
        )


# ============================================================
# Core function (from Hansen et al. 2018) 
# ============================================================
def spatiotemporal(im_file: str, bval_file: str, outbase: str, roi_file: str | None = None, order: int = 2) -> None:
    """
    Correct for spatiotemporal signal drift by fitting a polynomial in space and time.

    Arguments:
        im_file:   path to nifti image file
        bval_file: path to .bval file
        outbase:   basis for output filenames (no extension)
        roi_file:  path to nifti ROI/mask file (required)
        order:     order of estimated polynomial (1 or 2)
    """

    def _hansen_A(mask, n, order):
        if order not in [1, 2]:
            raise ValueError('order must be 1 or 2')
        sz = mask.shape
        coords = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        x, y, z = [coords[i][mask]/sz[i] for i in range(3)]
        xyz = np.vstack((np.ones_like(x), x, y, z, x*y, x*z, y*z, x*y*z)).T
        if order == 2:
            xyz = np.vstack((xyz.T,
                            x**2, y**2, z**2,
                            x*y**2, x*z**2,
                            x*y**2*z, x*y*z**2, x*y**2*z**2,
                            x**2*y, x**2*z, x**2*y*z, x**2*y**2, x**2*z**2,
                            x**2*y**2*z, x**2*y*z**2, x**2*y**2*z**2,
                            y**2*z, y*z**2, y**2*z**2)).T

        A = np.vstack((np.repeat(xyz, n.size, axis=0).T,
                       (np.tile(n, x.size)[:, np.newaxis] * np.repeat(xyz, n.size, axis=0)).T)).T
        if order == 2:
            A = np.vstack((A.T,
                           (np.tile(n**2, x.size)[:, np.newaxis] * np.repeat(xyz, n.size, axis=0)).T)).T
        return A

    def _hansen_fwd(p, A):
        return A @ p

    def _hansen_inv(Y, p, A, scale=100):
        return Y * scale / _hansen_fwd(p, A).reshape(Y.shape)

    # --- Load data ---
    Y, b = data_from_file(im_file, bval_file, roi_file=roi_file)

    if roi_file is not None:
        mask = read_im(roi_file).astype(bool)
    else:
        raise ValueError('A mask/ROI file is required for spatiotemporal drift correction.')

    # --- Fit ---
    n = (b == 0).nonzero()[0]
    zeromask = ~np.all(Y[:, n] == 0, axis=1)
    mask[mask] = zeromask
    Y = Y[zeromask, :]
    Y = Y / np.mean(Y[:, n], axis=1)[:, np.newaxis]
    y = Y[:, n].flatten()
    A = _hansen_A(mask, n, order)

    X = np.vstack((np.ones(A.shape[0]), A.T / b.size)).T
    m = np.mean(X, axis=0)
    C = X.T @ X / (Y.shape[0] - 1)
    h = 1/Y.shape[0] + np.einsum('ij,ji->i',
                                  (X - m[np.newaxis, :]),
                                  np.linalg.inv(C) @ (X - m[np.newaxis, :]).T) / (Y.shape[0] - 1)
    w = np.ones_like(h)
    Sold = np.inf

    for i in range(15):
        p = np.linalg.lstsq(np.sqrt(w[:, np.newaxis]) * A, np.sqrt(w) * y, rcond=None)[0]
        r = np.abs(y - _hansen_fwd(p, A))
        S = np.median(np.sort(r)[(p.size - 1):]) / 0.6745
        u = r / (4.685 * S * np.sqrt(1 - h))
        w[np.abs(u) > 1]  = 0
        w[np.abs(u) <= 1] = (1 - u[np.abs(u) <= 1]**2)**2
        print(f'  Iteration {i}, S = {S:.6f}')
        if np.abs(S - Sold) / S < 1e-3:
            break
        Sold = S

    # --- Save outputs ---
    Acorr = _hansen_A(mask, np.arange(Y.shape[1]), order)
    Ycorr = _hansen_inv(Y, p, Acorr)

    file_from_data(outbase + '_corr.nii.gz',      Ycorr,                                                              roi=mask, imref_file=im_file)
    Yhat = 1 / _hansen_inv(np.ones_like(Y), p, Acorr)
    file_from_data(outbase + '_corrfield.nii.gz', (Yhat - Yhat[:, 0][:, np.newaxis]) / Yhat[:, 0][:, np.newaxis],    roi=mask, imref_file=im_file)
    file_from_data(outbase + '_resid.nii.gz',     Y[:, n] - _hansen_fwd(p, A).reshape(Y[:, n].shape),                roi=mask, imref_file=im_file)

    with open(outbase + '.json', 'w') as f:
        json.dump({'p': list(p)}, f)


# ============================================================
# RUNNNNNN
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="Spatiotemporal signal drift correction (Hansen et al. 2018)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i",     "--input",   required=True,               help="Input DWI .nii.gz")
    parser.add_argument("-bval",  "--bval",    required=True,               help="Input .bval file")
    parser.add_argument("-o",     "--outbase", required=True,               help="Output base path (no extension)")
    parser.add_argument("-roi",   "--roi",   required=True,               help="ROI/mask .nii.gz (required by algorithm)")
    parser.add_argument("-order", "--order",   default=2, type=int,
                                               choices=[1, 2],              help="Polynomial order")
    return parser.parse_args()


def main():
    args = get_args()
    # --- Create output dir ---
    outdir = os.path.dirname(args.outbase)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # --- Print summary ---
    print(f"→ Input:   {args.input}")
    print(f"→ Bval:    {args.bval}")
    print(f"→ ROI:     {args.roi or 'None (will raise error)'}")
    print(f"→ Outbase: {args.outbase}")
    print(f"→ Order:   {args.order}\n")

    spatiotemporal(
        im_file   = args.input,
        bval_file = args.bval,
        outbase   = args.outbase,
        roi_file  = args.roi,
        order     = args.order
    )

if __name__ == "__main__":
    main()