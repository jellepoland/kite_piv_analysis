from pathlib import Path
import numpy as np
from scipy.interpolate import griddata

project_dir = Path(__file__).resolve().parent.parent


def interp2d_jelle(
    d2x: np.ndarray,
    d2y: np.ndarray,
    d2_variable: np.ndarray,
    d2curve: np.ndarray,
    method="linear",
):
    points = np.column_stack((d2x.ravel(), d2y.ravel()))

    # Interpolate u and v values at each boundary point on the curve
    d1_variable = griddata(points, d2_variable.ravel(), d2curve, method=method)

    return d1_variable
