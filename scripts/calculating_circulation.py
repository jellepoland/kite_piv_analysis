import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from pathlib import Path
import time
from utils import project_dir
from defining_bound_volume import boundary_ellipse, calculate_circulation


def main(
    csv_path: Path, d1centre: np.ndarray, drot: float, dLx: float, dLy: float, iP: int
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
    data = pd.read_csv(csv_path)

    # Extract the relevant columns for x, y, u, and v, ignoring any NaN entries
    d2x = data["x"].values.reshape(-1, 1)
    d2y = data["y"].values.reshape(-1, 1)
    d2u = data["u"].values.reshape(-1, 1)
    d2v = data["v"].values.reshape(-1, 1)

    # Define the elliptical boundary curve using boundary_ellipse
    d2curve = boundary_ellipse(d1centre, drot, dLx, dLy, iP)

    # Calculate circulation using the previously defined function
    dGamma = calculate_circulation(d2x, d2y, d2u, d2v, d2curve)

    return dGamma


if __name__ == "__main__":
    # Example usage
    csv_path = (
        Path(project_dir) / "processed_data" / "CFD" / "Y1_paraview_corrected.csv"
    )

    d1centre = np.array([0.3, 0.15])
    drot = 0
    dLx = 0.7
    dLy = 0.4
    iP = 20

    start_time = time.time()
    circulation = main(csv_path, d1centre, drot, dLx, dLy, iP)
    print(f"Circulation around the elliptical boundary: {circulation}")
    print(f"Time taken: {time.time() - start_time:.4f} seconds")

    rho = 1.225
    V_inf = 15
    lift_2D = rho * V_inf * circulation
    print(f"\n Lift force (2D) = {lift_2D} N (approx. using Kutta-Joukowski theorem)")
