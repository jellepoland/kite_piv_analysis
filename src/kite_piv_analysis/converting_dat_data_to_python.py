import numpy as np
import pandas as pd
import re
import os
import sys
import logging
import xarray as xr
from dataclasses import dataclass
from typing import List, Dict

logging.basicConfig(level=logging.INFO)

# Go back to root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)


@dataclass
class PIVRun:
    file_path: str
    file_name: str
    variable_names: List[str]
    i_value: int
    j_value: int
    data: xr.Dataset

def read_dat_file(file_path: str) -> PIVRun:

    # Extract the case name from the file path
    case_name = os.path.basename(os.path.dirname(file_path))

    # Read the header information (first 4 lines)
    with open(file_path, "r") as file:
        title = file.readline().strip()
        variables = file.readline().strip()
        zone = file.readline().strip()
        strand_solution = file.readline().strip()

    #Rename the variables
    file_name = title

    # Extract the variable names (skip the first 11 characters: 'VARIABLES =')
    variable_names = [v.strip().strip('"') for v in variables[11:].split(",")]

    # Extract the i and j values from the zone line
    zone_parts = zone.split(",")
    i_value = int(zone_parts[1].split("=")[1])
    j_value = int(zone_parts[2].split("=")[1])

    # Load the data (skip the first 4 lines of header information)
    data = np.genfromtxt(file_path, skip_header=4)

    # Reshape the data into matrix form (j by i)
    data_matrix = data.reshape((j_value, i_value, -1))

    # Create an xarray Dataset
    data_dict = {
        var: (("y", "x"), data_matrix[..., idx])
        for idx, var in enumerate(variable_names)
    }
    dataset = xr.Dataset(data_dict)

    # Print the dataset
    logging.info(dataset)

    # Access individual variables
    logging.info(dataset["Velocity u [m/s]"])

    breakpoint()

    return PIVRun(
        header=header,
        data=df,
        flow_info={},
        geometric_info={},
        additional_velocity=np.array([]),
        induction_velocity=np.array([]),
    )


read_dat_file(
    sys.path[0] + "/data/aoa_13/flipped_aoa_13_vw_15_H_918_Z1_Y1_X2_v5/B0001.dat"
)
breakpoint()


def read_xlsx_file(filename: str) -> pd.DataFrame:
    return pd.read_excel(filename)


def process_piv_data(dat_filename: str, xlsx_filename: str) -> PIVRun:
    piv_run = read_dat_file(dat_filename)
    lab_book = read_xlsx_file(xlsx_filename)

    # Find the corresponding row in the lab book
    dat_file_base = dat_filename.split("/")[-1].split(".")[0]
    lab_book_row = lab_book[
        lab_book["Filename"].str.contains(dat_file_base, na=False)
    ].iloc[0]

    # Add flow information
    piv_run.flow_info = {
        "config": lab_book_row["config"],
        "aoa": lab_book_row["aoa"],
        "sideslip": lab_book_row["sideslip"],
        "set_ws": lab_book_row["Set ws"],
        "actual_ws": lab_book_row["actual ws"],
        "pressure": lab_book_row["Pressure"],
        "temperature": lab_book_row["Temp"],
        "density": lab_book_row["Density"],
    }

    # Add geometric information
    piv_run.geometric_info = {
        "H_table": lab_book_row["H_table"],
        "X": lab_book_row["X"],
        "Y": lab_book_row["Y"],
        "Z": lab_book_row["Z"],
    }

    # Calculate additional velocity (example: resultant velocity)
    piv_run.additional_velocity = np.sqrt(
        piv_run.data["Velocity u [m/s]"] ** 2
        + piv_run.data["Velocity v [m/s]"] ** 2
        + piv_run.data["Velocity w [m/s]"] ** 2
    ).values.reshape(int(piv_run.header["J"]), int(piv_run.header["I"]))

    # Calculate induction velocity (subtracting mean stream)
    mean_velocity = (
        piv_run.data[["Velocity u [m/s]", "Velocity v [m/s]", "Velocity w [m/s]"]]
        .mean()
        .values
    )
    induction_velocity = (
        piv_run.data[
            ["Velocity u [m/s]", "Velocity v [m/s]", "Velocity w [m/s]"]
        ].values
        - mean_velocity
    )
    piv_run.induction_velocity = induction_velocity.reshape(
        int(piv_run.header["J"]), int(piv_run.header["I"]), 3
    )

    return piv_run


# Example usage
dat_file = "path/to/your/datfile.dat"
xlsx_file = "path/to/your/labbook.xlsx"
piv_data = process_piv_data(dat_file, xlsx_file)

# Access the processed data
print(piv_data.header)
print(piv_data.data.head())
print(piv_data.flow_info)
print(piv_data.geometric_info)
print(piv_data.additional_velocity.shape)
print(piv_data.induction_velocity.shape)
