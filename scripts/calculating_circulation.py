import numpy as np
import pandas as pd
from scipy.interpolate import interp2d, griddata
from pathlib import Path
import time
from utils import project_dir, interp2d_jelle
from defining_bound_volume import boundary_ellipse, boundary_rectangle


def calculate_circulation(
    df: pd.DataFrame,
    d2curve: np.ndarray,
) -> float:
    """
    Calculates the circulation around a boundary curve using linear interpolation and trapezoidal integration.

    Parameters:
    - d2x (np.ndarray): Matrix of x-coordinates from data (e.g., grid of data points).
    - d2y (np.ndarray): Matrix of y-coordinates from data.
    - d2u (np.ndarray): Matrix of u-velocity components (aligned with d2x, d2y).
    - d2v (np.ndarray): Matrix of v-velocity components (aligned with d2x, d2y).
    - d2curve (np.ndarray): Matrix of x, y boundary points of the curve from `boundary_ellipse` or `boundary_rectangle`.
    - search_radius (float): Radius to consider for nearby points when interpolating.

    Returns:
    - float: Calculated circulation `dGamma` along the curve.
    """
    # Extract the relevant columns for x, y, u, and v, ignoring any NaN entries
    d2x = df["x"].values.reshape(-1, 1)
    d2y = df["y"].values.reshape(-1, 1)
    d2u = df["u"].values.reshape(-1, 1)
    d2v = df["v"].values.reshape(-1, 1)

    # Interpolate u and v values at each boundary point on the curve
    d1u = interp2d_jelle(d2x, d2y, d2u, d2curve)
    d1v = interp2d_jelle(d2x, d2y, d2v, d2curve)
    # Calculate circulation using trapezoidal integration over the boundary points
    dGamma = -np.trapz(d1u.ravel(), d2curve[:, 0]) - np.trapz(
        d1v.ravel(), d2curve[:, 1]
    )

    return dGamma


def main(
    csv_path: Path,
    is_ellipse: bool,
    d1centre: np.ndarray,
    drot: float,
    dLx: float,
    dLy: float,
    iP: int,
) -> float:
    """
    Defines an elliptical boundary and calculates the circulation using data from a CSV file.

    Parameters:
    - csv_path (Path): Path to the CSV file containing `x`, `y`, `u`, `v`, and other data.
    - d1centre (np.ndarray): Array specifying the (x, y) center coordinates of the ellipse boundary.
    - drot (float): Rotation angle in degrees; 0 corresponds to no rotation.
    - dLx (float): Length of the ellipse along the x-axis.
    - dLy (float): Length of the ellipse along the y-axis.
    - iP (int): Number of points to define the ellipse boundary curve; higher values increase resolution.

    Returns:
    - float: Calculated circulation `dGamma` for the defined boundary curve.

    Notes:
    - The function reads the CSV file to obtain `d2x`, `d2y`, `d2u`, and `d2v` arrays used in circulation calculations.
    - The `boundary_ellipse` function is used to define the elliptical boundary curve.
    - Interpolation and trapezoidal integration are applied within `calculate_circulation` to compute circulation.
    """

    # Load data from CSV into a DataFrame
    df = pd.read_csv(csv_path)

    if is_ellipse:
        d2curve = boundary_ellipse(d1centre, drot, dLx, dLy, iP)
    else:
        d2curve = boundary_rectangle(d1centre, drot, dLx, dLy, iP)

    # Calculate circulation using the previously defined function
    dGamma = calculate_circulation(df, d2curve)

    return dGamma


if __name__ == "__main__":
    # Example usage
    csv_path = (
        Path(project_dir)
        / "processed_data"
        / "CFD"
        / "alpha_6"
        / "Y1_paraview_corrected.csv"
    )
    d1centre = np.array([0.27, 0.13])
    drot = 0
    dLx = 0.8
    dLy = 0.4
    iP = 50

    circulation = main(
        csv_path,
        is_ellipse=True,
        d1centre=d1centre,
        drot=0,
        dLx=dLx + 0.1,
        dLy=dLy + 0.1,
        iP=iP,
    )
    print(f"Circulation around the elliptical boundary: {circulation}")
    rho = 1.225
    V_inf = 15
    lift_2D = rho * V_inf * circulation
    print(f"Lift force (2D) = {lift_2D} N (approx. using Kutta-Joukowski theorem)")

    print(f"\nRECTANGLE")
    circulation = main(
        csv_path,
        is_ellipse=False,
        d1centre=d1centre,
        drot=0,
        dLx=dLx,
        dLy=dLy,
        iP=iP,
    )
    print(f"Circulation around the elliptical boundary: {circulation}")
    rho = 1.225
    V_inf = 15
    lift_2D = rho * V_inf * circulation
    print(f"Lift force (2D) = {lift_2D} N (approx. using Kutta-Joukowski theorem)")
