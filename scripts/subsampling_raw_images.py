import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from pathlib import Path
import pandas as pd
import os
from io import StringIO


def subsample_and_save_dat(dat_file_path, subsample_factor):
    """
    Read a .dat file, subsample it, and save as CSV in the same directory
    with '_subsampled' appended to the filename.

    Args:
        dat_file_path (str or Path): Path to the original .dat file
        subsample_factor (int): Factor by which to subsample the data

    Returns:
        Path: Path to the saved subsampled file
    """
    # Convert to Path object if it isn't already
    dat_file_path = Path(dat_file_path)

    # Read the original file
    df = pd.read_csv(
        dat_file_path,
        sep="\s+",
        skiprows=5,
        names=["x [mm]", "y [mm]", "Intensity [counts]", "isValid"],
        on_bad_lines="skip",
    )

    # Subsample the dataframe
    df_subsampled = df.iloc[::subsample_factor, :]

    # Create new filename
    # Replace .dat with _subsampled.csv
    new_filename = dat_file_path.stem + "_subsampled.csv"
    output_path = dat_file_path.parent / new_filename

    # Save as CSV
    df_subsampled.to_csv(output_path, index=False)


def main():
    project_dir = "/home/jellepoland/ownCloud/phd/code/kite_piv_analysis"
    aoa = 13
    y_num = 1
    x_num_list = [1, 2]
    subsample_factor = 75
    dat_file_dir = (
        Path(project_dir) / "data" / "raw_images" / f"aoa_{aoa}" / f"Y{y_num}"
    )

    for x_num in x_num_list:
        subsample_and_save_dat(
            Path(dat_file_dir) / f"flipped_Y{y_num}_X{x_num}" / "B0001.dat",
            subsample_factor,
        )
        subsample_and_save_dat(
            Path(dat_file_dir) / f"normal_Y{y_num}_X{x_num}" / "B0001.dat",
            subsample_factor,
        )


if __name__ == "__main__":
    main()
