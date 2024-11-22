from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import (
    interp2d,
)

project_dir = Path(__file__).resolve().parent.parent


def reshape_remove_nans(col, n_rows, n_cols):
    return col.fillna(0).values.reshape(n_rows, n_cols)


def interp2d_batch(d2x, d2y, d2_values, points, kind="linear"):
    """Perform 2D interpolation at multiple points using scipy's interp2d."""
    # print(f"\ninput: d2x: {d2x}")
    # print(f"\ninput: d2y: {d2y}")
    # print(f"\ninput: d2_values: {d2_values}")
    # print(f"\ninput: points: {points}")
    # print(f"dim: d2x: {d2x.shape}, d2y: {d2y.shape}, d2_values: {d2_values.shape}")

    # Create the interpolator
    d2x_flat = np.sort(np.unique(d2x.ravel()))
    d2y_flat = np.sort(np.unique(d2y.ravel()))
    ip = interp2d(d2x_flat, d2y_flat, d2_values, kind=kind)

    x_points, y_points = zip(*points)

    ip_out_list = []
    for xi, yi in zip(x_points, y_points):
        # Perform interpolation for each point
        ip_out = ip(xi, yi)
        ip_out_list.append(ip_out)
    # print(f"dim input: points: {np.array(points).shape}")
    # print(f"dim: ip_out: {np.array(ip_out_list).shape}")
    return np.array(ip_out_list).flatten()


def csv_reader(
    is_CFD: bool = True, alpha: float = 6, y_num: int = 1, alpha_d_rod: float = 7.25
):

    if is_CFD:
        csv_path = csv_path = (
            Path(project_dir)
            / "processed_data"
            / "CFD"
            / f"alpha_{int(alpha)}"
            / f"Y{y_num}_paraview_corrected.csv"
        )
    else:
        csv_path = (
            Path(project_dir)
            / "processed_data"
            / "stichted_planes_erik"
            / f"aoa_{int(alpha+alpha_d_rod)}"
            / f"aoa_{int(alpha+alpha_d_rod)}_Y{y_num}_stichted.csv"
        )

    df_1D = pd.read_csv(csv_path)
    return df_1D


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
    # print("Input data:")
    # print("x:", x)
    # print("y:", y)
    # print("z:", z)

    # Interpolation points
    points = np.array([[5.5, 5.5], [5.5, 5.5], [5.5, 5.5]])
    results = interp2d_batch(x, y, z, points)

    # Print results
    print(f"\nTesting part 1")
    print("MATLAB result == 0.2682")
    print("Batch interp2d result:", results)

    ## TESTING ##
    print(f"\nTesting part 2")
    # Example input
    d2x = np.array([0, 1, 2, 3])
    d2y = np.array([0, 1, 2, 3])
    d2_values = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    points = [[1.5, 1.5], [2.5, 2.5]]

    result = interp2d_batch(d2x, d2y, d2_values, points)
    print(f"Matlab result = [3, 5]")
    print(f"Python result = {result}")

    # Example 2D data grid (d2x, d2y)
    d2x = np.array([0, 3.4, 6.7, 8.56])  # X values
    d2y = np.array([0, 3.4, 6.7])  # Y values

    # Example 2D values corresponding to the (x, y) grid
    d2_values = np.array(
        [[1.2342, 2.11, 1.00, 23], [-0.2, 34.5, 22.3, 11], [21.6, 7.5, 6.9, 6]]
    )

    # Points to interpolate (xi, yi)
    points = [[0.5, 0.5], [1.5, 1.5], [2.0, 2.0]]

    # Interpolated values
    interpolated_values = interp2d_batch(d2x, d2y, d2_values, points)
    print(f"\nTesting part 3")
    print(f"Matlab results:   1.883575e+00, 7.571277e+00, 1.260960e+01")
    print("Interpolated values (Python):", interpolated_values)
