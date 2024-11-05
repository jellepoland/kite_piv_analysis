from pathlib import Path
import numpy as np
from scipy.interpolate import (
    griddata,
    interp2d,
    RectBivariateSpline,
    RegularGridInterpolator,
)

project_dir = Path(__file__).resolve().parent.parent


def interp2d_batch(d2x, d2y, d2_values, points, kind="linear"):
    """Perform 2D interpolation at multiple points using scipy's interp2d."""
    # print(f"\ninput: d2x: {d2x}")
    # print(f"\ninput: d2y: {d2y}")
    # print(f"\ninput: d2_values: {d2_values}")
    # print(f"\ninput: points: {points}")
    # print(f"dim: d2x: {d2x.shape}, d2y: {d2y.shape}, d2_values: {d2_values.shape}")

    # Create the interpolator
    x = np.sort(np.unique(d2x.ravel()))
    y = np.sort(np.unique(d2y.ravel()))
    ip = interp2d(x, y, d2_values, kind=kind)

    ip_out_list = []
    for point in points:
        # Perform interpolation for each point
        xi, yi = point
        ip_out = ip(xi, yi)
        ip_out_list.append(ip_out)
    print(f"dim input: points: {np.array(points).shape}")
    print(f"dim: ip_out: {np.array(ip_out_list).shape}")
    return np.array(ip_out_list).flatten()


# def interp2d_scipy(x, y, z, xi, yi, kind="linear"):
#     """Interpolation using scipy's interp2d."""
#     ip = interp2d(x, y, z, kind=kind)
#     return ip(xi, yi)[0]  # Return the first element to match output shape


# def interp2d_fast(d2x, d2y, d2_variable, d2curve, method="linear"):
#     interpolator = RegularGridInterpolator((d2x, d2y), d2_variable, method=method)
#     return interpolator(d2curve)


# def interp2d_jelle(
#     d2x: np.ndarray,
#     d2y: np.ndarray,
#     d2_variable: np.ndarray,
#     d2curve: np.ndarray,
#     method="linear",
# ):
#     interpolated_points = []
#     for point in d2curve:
#         interpolated_point = interp2d_scipy(
#             d2x.ravel(), d2y.ravel(), d2_variable, point[0], point[1], method
#         )
#         interpolated_points.append(interpolated_point)
#     return np.array(interpolated_points)


def prepare_grid_inputs(d2x, d2y, d2_variable):
    """
    Prepare inputs for RegularGridInterpolator by reshaping and sorting d2x and d2y.

    Parameters:
    - d2x: Array of x-coordinates.
    - d2y: Array of y-coordinates.
    - d2_variable: 2D array of values on the (d2x, d2y) grid.

    Returns:
    - d2x_sorted, d2y_sorted: 1D arrays sorted in ascending order.
    - d2_variable_sorted: 2D array of values sorted according to the sorted grid.
    """
    # Ensure d2x and d2y are 1D by taking unique values along each axis
    d2x_unique = np.unique(d2x.ravel())
    d2y_unique = np.unique(d2y.ravel())

    # Sort d2x and d2y and ensure d2_variable aligns with sorted axes
    if not np.all(np.diff(d2x_unique) > 0):  # Ascending check
        d2x_unique = d2x_unique[::-1]  # Reverse order if needed
        d2_variable = d2_variable[::-1, :]

    if not np.all(np.diff(d2y_unique) > 0):  # Ascending check
        d2y_unique = d2y_unique[::-1]
        d2_variable = d2_variable[:, ::-1]

    return d2x_unique, d2y_unique, d2_variable


# def interp2d_batch(d2x, d2y, d2_variable, d2curve, method="linear"):
#     """
#     Perform 2D interpolation on multiple points in batch mode.

#     Parameters:
#     - d2x: 2D array or 1D array of x-coordinates.
#     - d2y: 2D array or 1D array of y-coordinates.
#     - d2_variable: 2D array of values on the (d2x, d2y) grid.
#     - d2curve: Nx2 array of [x, y] points for interpolation.
#     - method: Interpolation method ('linear' or 'nearest').

#     Returns:
#     - Interpolated values for each point in `d2curve`.
#     """

#     print(
#         f"before: dim: d2x: {d2x.shape}, d2y: {d2y.shape}, d2_variable: {d2_variable.shape}"
#     )
#     # print(f"d2x: {d2x}")
#     # print(f"d2y: {d2y}")

#     import matplotlib.pyplot as plt

#     plt.scatter(d2x, d2y, label="Grid Points", color="blue")
#     plt.scatter(d2curve[:, 0], d2curve[:, 1], label="Curve Points", color="red")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.legend()
#     plt.show()

#     d2x = np.sort(np.unique(d2x.ravel()))
#     d2y = np.sort(np.unique(d2y.ravel()))

#     print(
#         f"after: dim: d2x: {d2x.shape}, d2y: {d2y.shape}, d2_variable: {d2_variable.shape}"
#     )
#     print(
#         f"(d2y, d2x): {d2y.shape}, {d2x.shape}, x_bounds: [{d2x.min():.2f},{d2x.max():.2f}], y_bounds: [{d2y.min():.2f},{d2y.max():.2f}]"
#     )
#     print(
#         f"d2curve: {d2curve.shape}, print x_bounds: [{d2curve[:, 0].min():.2f},{d2curve[:, 0].max():.2f}], y_bounds: [{d2curve[:, 1].min():.2f},{d2curve[:, 1].max():.2f}]"
#     )
#     # print(f"d2x: {d2x}")
#     # print(f"d2y: {d2y}")

#     # Create the interpolator object
#     interpolator = RegularGridInterpolator((d2y, d2x), d2_variable, method=method)

#     # Perform interpolation for all clamped points in `d2curve`
#     return interpolator(d2curve)


def interp2d_jelle(
    d2x: np.ndarray,
    d2y: np.ndarray,
    d2_variable: np.ndarray,
    d2curve: np.ndarray,
    method="cubic",
):
    points = np.column_stack((d2x.ravel(), d2y.ravel()))

    # Interpolate u and v values at each boundary point on the curve
    d1_variable = griddata(points, d2_variable.ravel(), d2curve, method=method)

    return d1_variable


import numpy as np
from scipy.interpolate import griddata, interp2d, RectBivariateSpline


def interp_griddata(
    d2x: np.ndarray,  # 1D array of x-coordinates (grid points)
    d2y: np.ndarray,  # 1D array of y-coordinates (grid points)
    d2_variable: np.ndarray,  # 2D array of values on the (d2x, d2y) grid
    xi: float,  # X-coordinate for interpolation
    yi: float,  # Y-coordinate for interpolation
    method="cubic",
):
    # Ensure method is valid
    valid_methods = {"linear", "cubic", "nearest"}
    if method not in valid_methods:
        print(f"Warning: Unknown method '{method}', defaulting to 'linear'")
        method = "linear"

    # Create meshgrid from 1D x and y arrays
    mesh_x, mesh_y = np.meshgrid(d2x, d2y)

    # Stack meshgrid coordinates into points array
    points = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))

    # Flatten the variable data
    values = d2_variable.ravel()

    # Prepare the input points for interpolation
    d2curve = np.array([[xi, yi]])  # Create a 2D array for a single (x, y) point

    # Interpolate at points defined by d2curve
    d1_variable = griddata(points, values, d2curve, method=method)

    return (
        d1_variable[0] if d1_variable is not None else None
    )  # Return the interpolated value


def interp2d_scipy(x, y, z, xi, yi, kind="linear"):
    """Interpolation using scipy's interp2d."""
    ip = interp2d(x, y, z, kind=kind)
    return ip(xi, yi)[0]  # Return the first element to match output shape


def rect_bivariate_spline(x, y, z, xi, yi, kx=3, ky=3, s=0):
    """Interpolation using RectBivariateSpline."""
    spline = RectBivariateSpline(x, y, z, kx=kx, ky=ky, s=s)
    return spline(xi, yi)[0, 0]  # Return the first element to match output shape


def nearest_neighbor(x, y, z, x0, y0):
    """Nearest-neighbor interpolation."""
    xi = np.abs(x - x0).argmin()
    yi = np.abs(y - y0).argmin()
    return z[yi, xi]


# Example testing script
if __name__ == "__main__":
    # Define the variables in MATLAB
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x, y = np.meshgrid(x, y)

    z = [
        [
            4.99094441e-01,
            6.67687865e-03,
            9.97170957e-01,
            9.43441206e-01,
            1.22367273e-01,
            8.86458365e-01,
            7.63812442e-01,
            5.63590418e-01,
            1.43184492e-01,
            5.48576248e-01,
            7.07253628e-01,
        ],
        [
            4.79997275e-01,
            8.78071662e-01,
            9.78347148e-01,
            5.97270338e-01,
            6.26036810e-01,
            3.75892029e-01,
            6.57586840e-01,
            2.00312974e-01,
            1.00088015e-01,
            9.75218532e-01,
            8.81224488e-01,
        ],
        [
            2.71540970e-01,
            1.34981750e-01,
            8.32040126e-01,
            7.19896800e-01,
            1.25012017e-01,
            2.56944303e-01,
            4.81102089e-01,
            9.32831864e-01,
            2.91856730e-01,
            8.31558647e-01,
            6.59688417e-01,
        ],
        [
            1.79409421e-01,
            3.01616017e-01,
            7.79765655e-02,
            2.54064911e-01,
            6.96664927e-01,
            1.68309157e-01,
            9.57384112e-01,
            5.06814941e-01,
            9.54026043e-01,
            5.25217837e-01,
            2.30496192e-01,
        ],
        [
            9.37771756e-01,
            3.30902883e-01,
            2.73433253e-01,
            3.07858166e-05,
            4.45931386e-01,
            7.52509391e-01,
            6.79676912e-01,
            6.99464097e-01,
            9.95797800e-01,
            3.48875774e-02,
            7.60978053e-01,
        ],
        [
            1.10499542e-01,
            7.17070042e-01,
            7.52048646e-02,
            4.42908486e-01,
            3.56518279e-01,
            4.53238967e-01,
            1.20260298e-01,
            6.15431587e-01,
            5.81692030e-01,
            7.84128830e-01,
            9.99602496e-05,
        ],
        [
            9.55527067e-01,
            2.98949357e-01,
            4.62574203e-01,
            3.67438919e-01,
            9.56842478e-01,
            8.76941590e-02,
            4.11407656e-01,
            7.59039508e-01,
            7.56549378e-01,
            5.84336849e-01,
            1.57019120e-01,
        ],
        [
            5.76708427e-01,
            8.88165050e-01,
            8.13778773e-01,
            9.36522875e-01,
            1.36936928e-01,
            4.07681681e-01,
            2.63256927e-01,
            1.47661850e-01,
            4.67681511e-01,
            1.85775776e-01,
            5.66794997e-01,
        ],
        [
            3.92494267e-01,
            7.44367580e-01,
            6.70636726e-01,
            9.41462324e-01,
            4.78051542e-01,
            7.60055913e-01,
            9.82771047e-01,
            7.39040133e-01,
            8.50156893e-01,
            4.02678604e-01,
            5.76719946e-01,
        ],
        [
            2.05921488e-01,
            5.83107374e-01,
            8.99466888e-01,
            2.61753916e-01,
            9.08406687e-01,
            7.68948549e-01,
            2.60630542e-01,
            8.44102023e-01,
            3.60279286e-01,
            5.13798819e-01,
            6.30330308e-02,
        ],
        [
            1.60516593e-01,
            2.39367317e-01,
            4.56652271e-01,
            8.36040469e-01,
            8.90584930e-01,
            9.02960708e-01,
            8.32226329e-01,
            2.06588074e-01,
            3.93270749e-01,
            4.14984536e-01,
            1.11003538e-02,
        ],
    ]
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    print("Input data:")
    print("x:", x)
    print("y:", y)
    print("z:", z)

    # Interpolation points
    # xi, yi = 5.5, 5.5
    d2curve = np.array([[5.5, 5.5], [5.5, 5.5], [5.5, 5.5]])

    # Test each function
    # results_interp_griddata = interp_griddata(x, y, z, xi, yi)
    # result_interp2d = interp2d_scipy(x, y, z, xi, yi)
    # result_spline = rect_bivariate_spline(x, y, z, xi, yi)
    # result_nearest = nearest_neighbor(x, y, z, xi, yi)
    # interp2d_jelle(x, y, z, np.array([[xi, yi]]))
    interp2d_batch(x, y, z, d2curve)

    # Print results
    print("\nInterpolated values: MATLAB result == 0.2682")
    # print("griddata result:", results_interp_griddata)
    # print("interp2d result:", result_interp2d)
    # print("RectBivariateSpline result:", result_spline)
    # print("Nearest-neighbor result:", result_nearest)
    # print("Jelle's interp2d result:", interp2d_jelle(x, y, z, np.array([[xi, yi]]))[0])
    print("Batch interp2d result:", interp2d_batch(x, y, z, d2curve))
