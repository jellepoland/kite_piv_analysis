import numpy as np
from pathlib import Path
import pandas as pd
from scipy.signal import convolve2d
from utils import project_dir, interp2d_jelle
from defining_bound_volume import boundary_ellipse, boundary_rectangle


def conv2(x, y, mode="same"):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def cumsum(x):
    return np.cumsum(x, axis=0)


def gradient(u, dx=1, dy=1):
    """
    Calculate gradient with proper handling of small arrays

    Parameters:
    -----------
    u : ndarray
        Input array
    dx : float, optional
        Spacing in x direction
    dy : float, optional
        Spacing in y direction

    Returns:
    --------
    dudx : ndarray
        Gradient in x direction
    dudy : ndarray
        Gradient in y direction
    """
    # Ensure input is a NumPy array
    u = np.asarray(u)

    # Handle small arrays by checking dimensions
    if u.ndim == 1 and len(u) < 2:
        return np.zeros_like(u), np.zeros_like(u)
    elif u.ndim == 2 and min(u.shape) < 2:
        return np.zeros_like(u), np.zeros_like(u)

    # Compute gradient
    if u.ndim == 1:
        return np.gradient(u, dx)
    elif u.ndim == 2:
        return np.gradient(u, dx, dy)


def circshift(array, shift):
    """
    Circularly shift the elements of an array similar to MATLAB's circshift.

    Parameters:
    - array: np.ndarray
        The input array to be shifted.
    - shift: int
        The number of positions to shift. Positive values shift to the right,
        and negative values shift to the left.

    Returns:
    - np.ndarray: The circularly shifted array.
    """
    return np.roll(array, shift)


def reshape_data(df, x_col="x", y_col="y", *data_cols):
    """
    Reshapes columns of data into 2D arrays based on unique x and y values.

    Parameters:
    - df: DataFrame containing the data.
    - x_col: str, the name of the x-coordinate column.
    - y_col: str, the name of the y-coordinate column.
    - data_cols: str, names of the columns to be reshaped.

    Returns:
    - reshaped_data: dict, with each column reshaped into 2D form.
    """
    # Identify unique x and y values and create mesh grid
    x_unique = np.sort(df[x_col].unique())
    y_unique = np.sort(df[y_col].unique())
    x_mesh, y_mesh = np.meshgrid(x_unique, y_unique)

    # Reshape each data column
    # Add x and y grids to the output dictionary
    reshaped_data = {"x": x_mesh, "y": y_mesh}
    for col in data_cols:
        # Sort data by x and y columns to match the mesh grid
        sorted_data = df.sort_values(by=[y_col, x_col])[col].values
        reshaped_data[col] = sorted_data.reshape(len(y_unique), len(x_unique))

    return reshaped_data


def forceFromVelNoca2D_V3(
    d2x, d2y, d2u, d2v, d2vortZ, d2dudt, d2dvdt, d2curve, dmu, bcorMaxVort
):
    # Checking types
    # print(f"d2x {type(d2x)},  dim: {d2x.ndim}, shape: {d2x.shape}")
    # print(f"d2y {type(d2y)},  dim: {d2y.ndim}, shape: {d2y.shape}")
    # print(f"d2u {type(d2u)},  dim: {d2u.ndim}, shape: {d2u.shape}")
    # print(f"d2v {type(d2v)}, dim: {d2v.ndim}, shape: {d2v.shape}")
    # print(f"d2vortZ {type(d2vortZ)}, dim: {d2vortZ.ndim}, shape: {d2vortZ.shape}")
    # print(f"d2dudt {type(d2dudt)}, dim: {d2dudt.ndim}, shape: {d2dudt.shape}")
    # print(f"d2dvdt {type(d2dvdt)}, dim: {d2dvdt.ndim}, shape: {d2dvdt.shape}")
    # print(f"d2curve {type(d2curve)}, dim: {d2curve.ndim}, shape: {d2curve.shape}")
    # print(f"dmu {type(dmu)}")
    # print(f"bcorMaxVort {type(bcorMaxVort)}")

    # Initial value
    iN = 2

    # Data smoothing
    bsmooth = False
    ismooth = 9
    if bsmooth:
        d2u = conv2(d2u, np.ones((ismooth, ismooth)) / (ismooth**2), mode="same")
        d2v = conv2(d2v, np.ones((ismooth, ismooth)) / (ismooth**2), mode="same")
        d2vortZ = conv2(
            d2vortZ, np.ones((ismooth, ismooth)) / (ismooth**2), mode="same"
        )
        d2dudt = conv2(d2dudt, np.ones((ismooth, ismooth)) / (ismooth**2), mode="same")
        d2dvdt = conv2(d2dvdt, np.ones((ismooth, ismooth)) / (ismooth**2), mode="same")

    # Curve coordinate
    d1s = cumsum(
        np.concatenate(
            ([0], np.sqrt(np.sum((d2curve[1:] - d2curve[:-1]) ** 2, axis=1)))
        )
    )

    # Ensure inputs are NumPy arrays
    d2curve = np.asarray(d2curve)

    # Calculate curve derivatives
    d1nx = gradient(d2curve[:, 0])
    d1ny = gradient(d2curve[:, 1])

    # Normalize the vectors
    norm_factor = np.sqrt(d1ny**2 + d1nx**2)
    d1ny /= norm_factor
    d1nx /= norm_factor
    d1ny = -d1ny  # Negative as per original MATLAB code

    # Spatial gradients of first and second order
    ddx = d2x[1, 1] - d2x[0, 0]
    ddy = d2y[1, 1] - d2y[0, 0]

    d2dudx, d2dudy = gradient(d2u, ddx, ddy)
    d2dvdx, d2dvdy = gradient(d2v, ddx, ddy)

    d2d2udx2, d2d2udydx = gradient(d2dudx, ddx, ddy)
    d2d2vdx2, d2d2vdydx = gradient(d2dvdx, ddx, ddy)
    d2d2udxdy, d2d2udy2 = gradient(d2dudy, ddx, ddy)
    d2d2vdxdy, d2d2vdy2 = gradient(d2dvdy, ddx, ddy)

    # Vector fields interpolated along curve
    # Note: You'll need to implement interp2 equivalent function
    d1u = interp2d_jelle(d2x, d2y, d2u, d2curve)
    d1v = interp2d_jelle(d2x, d2y, d2v, d2curve)
    d1vortZ = interp2d_jelle(d2x, d2y, d2vortZ, d2curve)

    d1dudt = interp2d_jelle(d2x, d2y, d2dudt, d2curve)
    d1dvdt = interp2d_jelle(d2x, d2y, d2dvdt, d2curve)

    d1dudx = interp2d_jelle(d2x, d2y, d2dudx, d2curve)
    d1dudy = interp2d_jelle(d2x, d2y, d2dudy, d2curve)
    d1dvdx = interp2d_jelle(d2x, d2y, d2dvdx, d2curve)
    d1dvdy = interp2d_jelle(d2x, d2y, d2dvdy, d2curve)

    d1d2udx2 = interp2d_jelle(d2x, d2y, d2d2udx2, d2curve)
    # d1d2udydx = interp2d_jelle(d2x, d2y, d2d2udydx, d2curve)
    d1d2vdx2 = interp2d_jelle(d2x, d2y, d2d2vdx2, d2curve)
    # d1d2vdydx = interp2d_jelle(d2x, d2y, d2d2vdydx, d2curve)
    d1d2udxdy = interp2d_jelle(d2x, d2y, d2d2udxdy, d2curve)
    d1d2udy2 = interp2d_jelle(d2x, d2y, d2d2udy2, d2curve)
    d1d2vdxdy = interp2d_jelle(d2x, d2y, d2d2vdxdy, d2curve)
    d1d2vdy2 = interp2d_jelle(d2x, d2y, d2d2vdy2, d2curve)

    # Test: Change coord frame to minimise impact of vorticity term
    if bcorMaxVort:
        # Combine current and shifted vorticity values
        vort_combined = np.abs(
            d1vortZ
            + circshift(d1vortZ, 1)
            + circshift(d1vortZ, 2)
            + circshift(d1vortZ, 3)
            + circshift(d1vortZ, 4)
        )
        imaxVortZ = np.argmax(vort_combined)
        d2curve = d2curve - d2curve[imaxVortZ, :]

    # Calculate the various normal force contributions
    d2Fn = np.zeros((len(d1s), 11))

    # Inviscid terms
    d2Fn[:, 1] = 0.5 * d1nx * (d1u**2 + d1v**2)
    d2Fn[:, 2] = -(d1nx * d1u**2 + d1ny * d1v * d1u)
    d2Fn[:, 3] = (
        -1
        / (iN - 1)
        * (d1nx * d1u * d2curve[:, 1] * d1vortZ + d1ny * d1v * d2curve[:, 1] * d1vortZ)
    )
    d2Fn[:, 4] = 0

    # Time dependent terms
    d2Fn[:, 5] = (
        -1 / (iN - 1) * d1nx * (d2curve[:, 0] * d1dudt + d2curve[:, 1] * d1dvdt)
    )
    d2Fn[:, 6] = (
        1 / (iN - 1) * (d1nx * d2curve[:, 0] * d1dudt + d1ny * d2curve[:, 1] * d1dudt)
    )
    d2Fn[:, 7] = -(d1nx * d1dudt * d2curve[:, 0] + d1ny * d1dvdt * d2curve[:, 0])

    # Viscous terms
    d1nablaTau1 = 2 * d1d2udx2 + d1d2vdxdy + d1d2udy2
    d1nablaTau2 = d1d2udxdy + d1d2vdx2 + 2 * d1d2vdy2

    d2Fn[:, 8] = (
        1
        / (iN - 1)
        * dmu
        * (d1nx * (d2curve[:, 0] * d1nablaTau1 + d2curve[:, 1] * d1nablaTau2))
    )
    d2Fn[:, 9] = (
        -1
        / (iN - 1)
        * dmu
        * (d1nx * d2curve[:, 0] * d1nablaTau1 + d1ny * d2curve[:, 1] * d1nablaTau1)
    )
    d2Fn[:, 10] = dmu * (d1nx * 2 * d1dudx + d1ny * (d1dvdx + d1dudy))

    # Ensure total force is calculated in the first column of d2Fn
    d2Fn[:, 0] = np.sum(d2Fn[:, 1:], axis=1)

    # Integrate each column across rows (axis=0), producing 11 values
    d1Fn = np.trapz(d2Fn, d1s, axis=0)

    # Calculate the various tangential force contributions
    d2Ft = np.zeros((len(d1s), 11))

    # Inviscid terms
    d2Ft[:, 1] = 0.5 * d1ny * (d1u**2 + d1v**2)
    d2Ft[:, 2] = -(d1nx * d1u * d1v + d1ny * d1v**2)
    d2Ft[:, 3] = (
        -1
        / (iN - 1)
        * (-d1nx * d1u * d2curve[:, 0] * d1vortZ - d1ny * d1v * d2curve[:, 0] * d1vortZ)
    )
    d2Ft[:, 4] = 0

    # Time dependent terms
    d2Ft[:, 5] = (
        -1 / (iN - 1) * d1ny * (d2curve[:, 0] * d1dudt + d2curve[:, 1] * d1dvdt)
    )
    d2Ft[:, 6] = (
        1 / (iN - 1) * (d1nx * d2curve[:, 0] * d1dvdt + d1ny * d2curve[:, 1] * d1dvdt)
    )
    d2Ft[:, 7] = -(d1nx * d1dudt * d2curve[:, 1] + d1ny * d1dvdt * d2curve[:, 1])

    # Viscous terms
    d2Ft[:, 8] = (
        1
        / (iN - 1)
        * dmu
        * (
            d1ny
            * (
                d2curve[:, 0] * (2 * d1d2udx2 + d1d2vdxdy + d1d2udy2)
                + d2curve[:, 1] * (2 * d1d2udxdy + d1d2vdx2 + d1d2vdy2)
            )
        )
    )
    d2Ft[:, 9] = (
        -1
        / (iN - 1)
        * dmu
        * (d1nx * d2curve[:, 0] * d1nablaTau2 + d1ny * d2curve[:, 1] * d1nablaTau2)
    )
    d2Ft[:, 10] = dmu * (d1nx * (d1dudy + d1dvdx) + d1ny * 2 * d1dvdy)

    # Ensure total force is calculated in the first column of d2Ft
    d2Ft[:, 0] = np.sum(d2Ft[:, 1:], axis=1)

    # Integrate each column across rows (axis=0), producing 11 values
    d1Ft = np.trapz(d2Ft, d1s, axis=0)

    return d1Fn, d1Ft


def main(
    alpha: int,
    y_num: int,
    is_CFD: bool = True,
    is_ellipse: bool = True,
    d1centre: np.ndarray = np.array([0.27, 0.13]),
    drot: float = 0,
    dLx: float = 0.8,
    dLy: float = 0.4,
    iP: int = 50,
    mu: float = 1.8e-5,
    is_with_maximim_vorticity_location_correction: bool = True,
    alpha_d_rod: float = 7.25,
):

    # create d2curve
    if is_ellipse:
        d2curve = boundary_ellipse(d1centre, drot, dLx, dLy, iP)
        print(f"Running NOCA on Ellipse, will take a while...")
    else:
        d2curve = boundary_rectangle(d1centre, drot, dLx, dLy, iP)
        print(f"Running NOCA on Rectangle, will take a while...")

    # load data
    if is_CFD:
        csv_path = csv_path = (
            Path(project_dir)
            / "processed_data"
            / "CFD"
            / f"alpha_{alpha}"
            / f"Y{y_num}_paraview_corrected.csv"
        )
    else:
        csv_path = (
            Path(project_dir)
            / "processed_data"
            / "stichted_planes_erik"
            / f"alpha_{alpha}"
            / f"aoa_{int(alpha+alpha_d_rod)}_Y{y_num}_stichted.csv"
        )

    df_1D = pd.read_csv(csv_path)

    # reshape df
    df_2D = reshape_data(df_1D, "x", "y", "u", "v", "vort_z", "dudx", "dvdx")

    d1Fn, d1Ft = forceFromVelNoca2D_V3(
        d2x=df_2D["x"],
        d2y=df_2D["y"],
        d2u=df_2D["u"],
        d2v=df_2D["v"],
        d2vortZ=df_2D["vort_z"],
        d2dudt=df_2D["dudx"],
        d2dvdt=df_2D["dvdx"],
        d2curve=d2curve,
        dmu=mu,
        bcorMaxVort=is_with_maximim_vorticity_location_correction,
    )

    print(f"Normal force: {d1Fn[0]}")
    print(f"Tangential force: {d1Ft[0]}")

    return d1Fn, d1Ft


if __name__ == "__main__":
    main(
        alpha=6,
        y_num=1,
        is_CFD=True,
        is_ellipse=True,
        is_with_maximim_vorticity_location_correction=True,
    )
