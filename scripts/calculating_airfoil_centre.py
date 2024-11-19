import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from utils import project_dir


def load_translation_data(csv_path: str) -> pd.DataFrame:
    """
    Load the translation data from CSV file.

    Args:
        csv_path: Path to the CSV file containing translation data

    Returns:
        pd.DataFrame: DataFrame containing translation data
    """
    return pd.read_csv(csv_path, index_col=0)


def translate_center(
    x: float, y: float, alpha: int, Y: int, translation_data: pd.DataFrame
) -> Tuple[float, float]:
    """
    Translate the center coordinates based on the case parameters.

    Args:
        x: Original x coordinate
        y: Original y coordinate
        alpha: Angle of attack
        Y: Y number
        translation_data: DataFrame containing translation information

    Returns:
        Tuple[float, float]: (x_translated, y_translated) coordinates
    """
    # Find the corresponding row in translation data
    mask = (translation_data["alpha"] == alpha) & (translation_data["Y"] == Y)
    if not mask.any():
        raise ValueError(f"No translation data found for alpha={alpha}, Y={Y}")

    translation = translation_data[mask].iloc[0]

    # Apply translation
    x_translated = x + translation["x"]
    y_translated = y + translation["y"]

    return x_translated, y_translated


def calculate_center(
    alpha: int,
    y_num: int,
    translation_csv_path: Optional[str] = Path(project_dir)
    / "data"
    / "airfoils"
    / "airfoil_translation_values.csv",
) -> Tuple[float, float]:
    """
    Calculate the geometric center of an airfoil as the midpoint between min and max coordinates,
    then translate based on case parameters.

    Args:
        path_to_airfoil_dat: Path to the airfoil data file
        alpha: Angle of attack
        Y: Y number
        translation_csv_path: Path to CSV file containing translation data (optional if translation_data provided)
        translation_data: DataFrame containing translation data (optional if translation_csv_path provided)

    Returns:
        Tuple[float, float]: (x_center, y_center) coordinates of the translated centroid

    Example:
        >>> x_center, y_center = calculate_center("naca0012.dat", alpha=6, Y=1, translation_csv_path="translations.csv")
        >>> print(f"Translated center at: ({x_center:.6f}, {y_center:.6f})")
    """

    translation_data = load_translation_data(translation_csv_path)

    # Read the airfoil data file
    path_to_airfoil_dat: Path = (
        Path(project_dir) / "data" / "airfoils" / f"y{y_num}.dat"
    )
    try:
        df = pd.read_csv(path_to_airfoil_dat, delimiter="\s+", skiprows=1)
    except Exception as e:
        print(f"Error reading airfoil file: {e}")
        raise

    # Get x and y columns (first two columns)
    x_coords = df.iloc[:, 0].values
    y_coords = df.iloc[:, 1].values

    # Calculate center as midpoint between min and max
    x_center = (np.max(x_coords) + np.min(x_coords)) / 2
    y_center = (np.max(y_coords) + np.min(y_coords)) / 2

    # Apply translation
    x_translated, y_translated = translate_center(
        x_center, y_center, alpha, y_num, translation_data
    )

    return x_translated, y_translated


def saving_centers_in_csv():
    """
    Save the translated centers for all airfoils to a CSV file.
    """
    # Define the path to the CSV file
    csv_path = Path(project_dir) / "data" / "airfoils" / "translated_centers.csv"

    # Initialize an empty DataFrame to store the results
    results = pd.DataFrame(columns=["alpha", "Y", "x_center", "y_center"])

    # Loop over all airfoils and angles of attack
    alpha = 6
    for y_num in [1, 2, 3, 4, 5, 6, 7]:
        x_center, y_center = calculate_center(alpha, y_num)
        results = results._append(
            {"alpha": alpha, "Y": y_num, "x_center": x_center, "y_center": y_center},
            ignore_index=True,
        )
    alpha = 16
    for y_num in [1, 2, 3, 4]:
        x_center, y_center = calculate_center(alpha, y_num)
        results = results._append(
            {"alpha": alpha, "Y": y_num, "x_center": x_center, "y_center": y_center},
            ignore_index=True,
        )

    # Save the results to a CSV file
    results.to_csv(csv_path, index=False)
    print(f"Translated centers saved to: {csv_path}")


def reading_center_from_csv(alpha: int, y_num: int) -> Tuple[float, float]:
    # Reading in the airfoil centers
    path_to_airfoil_centers = (
        Path(project_dir) / "data" / "airfoils" / "translated_centers.csv"
    )
    pd_airfoil_centers = pd.read_csv(path_to_airfoil_centers)
    mask = (pd_airfoil_centers["alpha"] == alpha) & (pd_airfoil_centers["Y"] == y_num)
    airfoil_center = pd_airfoil_centers.loc[mask, ["x_center", "y_center"]].values[0]
    return airfoil_center


def main(alpha: str, y_num: str):
    x_center, y_center = calculate_center(alpha=int(alpha), y_num=int(y_num))
    return x_center, y_center


if __name__ == "__main__":
    alpha = 6
    y_num = 7
    x_center, y_center = main(alpha, y_num)
    print(
        f"Translated center for alpha {alpha}deg and Y{y_num} ({x_center:.6f}, {y_center:.6f})"
    )

    saving_centers_in_csv()
