import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from utils import project_dir


def main(
    alpha: int,
    y_num: int,
    translation_csv_path: Optional[Path] = Path(project_dir)
    / "data"
    / "airfoils"
    / "airfoil_translation_values.csv",
) -> Tuple[float, float]:
    """
    Calculate the geometric center of an airfoil, apply rotation and translation,
    and return the final translated center coordinates.

    Args:
        alpha (int): Angle of attack in degrees.
        y_num (int): Y-section number for the airfoil.
        translation_csv_path (Optional[Path]): Path to the CSV file containing translation values.

    Returns:
        Tuple[float, float]: The translated (x_center, y_center) coordinates of the airfoil center.
    """
    # Load translation data
    translation_data = pd.read_csv(translation_csv_path)
    # Filter for the current alpha and y_num
    translation_row = translation_data[
        (translation_data["alpha"] == alpha) & (translation_data["Y"] == y_num)
    ]
    if translation_row.empty:
        raise ValueError(f"No translation data found for alpha={alpha}, y_num={y_num}.")

    # Extract translation values
    x_translation = translation_row["x"].values[0]
    y_translation = translation_row["y"].values[0]

    # Path to the airfoil data file
    airfoil_file = Path(project_dir) / "data" / "airfoils" / f"y{y_num}.dat"
    airfoil_data = pd.read_csv(airfoil_file, header=None, skiprows=1, sep="\s+")

    # Extract x and y coordinates of the airfoil
    x_coords = airfoil_data.iloc[:, 0].values
    y_coords = airfoil_data.iloc[:, 1].values

    # Calculate the geometric center (midpoint between min and max)
    x_center = (np.max(x_coords) + np.min(x_coords)) / 2
    y_center = (np.max(y_coords) + np.min(y_coords)) / 2

    # Translate the center to the pre-defined location
    x_translated = x_center + x_translation
    y_translated = y_center + y_translation

    # Rotate the translated center by alpha
    alpha_rad = np.radians(-alpha)  # Negative for clockwise rotation
    x_final = x_translated * np.cos(alpha_rad) - y_translated * np.sin(alpha_rad)
    y_final = x_translated * np.sin(alpha_rad) + y_translated * np.cos(alpha_rad)

    return x_final, y_final


if __name__ == "__main__":
    alpha = 6
    y_num = 7
    x_center, y_center = main(alpha, y_num)
    print(
        f"Translated center for alpha {alpha}deg and Y{y_num} ({x_center:.6f}, {y_center:.6f})"
    )
