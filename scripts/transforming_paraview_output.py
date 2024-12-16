import pandas as pd
import numpy as np
import os
import scipy.interpolate as interpolate
from scipy.interpolate import griddata
from pathlib import Path
from utils import project_dir
import matplotlib.pyplot as plt


def scaling_velocity(data_array, headers, vel_scaling=15):
    """Scale velocity components in the data array by the given factor, ignoring x, y, z columns."""
    # Find the indices of velocity-related columns (anything except 'x', 'y', 'z')
    for i, header in enumerate(headers):
        if header in ["x", "y", "z"]:
            continue
        elif header in ["pressure"]:
            continue
            # data_array[:, i] *= vel_scaling**2
        else:
            data_array[:, i] *= vel_scaling
    # velocity_indices = [
    #     i for i, header in enumerate(headers) if header not in ["x", "y", "z"]
    # ]
    # # Scale the velocity components by the given factor
    # data_array[:, velocity_indices] *= vel_scaling

    return data_array


def rotate_data(points, data_dict, angle_deg):
    """
    Rotate points and vector quantities by specified angle.

    Args:
        points (np.ndarray): Array of points with shape (n, 3)
        data_dict (dict): Dictionary containing vector quantities
        angle_deg (float): Rotation angle in degrees

    Returns:
        tuple: (rotated_points, rotated_data_dict)
    """

    # Convert angle to radians
    angle_rad = -np.radians(angle_deg)

    # Create rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ]
    )

    # Rotate points
    rotated_points = np.dot(points, rotation_matrix.T)

    # Initialize rotated data dictionary
    rotated_data_dict = data_dict.copy()

    # Define vector quantities to rotate
    vector_quantities = {
        "U": ["U:0", "U:1", "U:2"],
        "grad_U": [
            "grad(U):0",
            "grad(U):1",
            "grad(U):2",
            "grad(U):3",
            "grad(U):4",
            "grad(U):5",
            "grad(U):6",
            "grad(U):7",
            "grad(U):8",
        ],
        "vorticity": ["vorticity:0", "vorticity:1", "vorticity:2"],
        "wallShearStress": [
            "wallShearStress:0",
            "wallShearStress:1",
            "wallShearStress:2",
        ],
    }

    # Rotate velocity vectors
    if all(key in data_dict for key in vector_quantities["U"]):
        velocity_vectors = np.column_stack(
            [data_dict["U:0"], data_dict["U:1"], data_dict["U:2"]]
        )
        rotated_velocities = np.dot(velocity_vectors, rotation_matrix.T)
        rotated_data_dict["U:0"] = rotated_velocities[:, 0]
        rotated_data_dict["U:1"] = rotated_velocities[:, 1]
        rotated_data_dict["U:2"] = rotated_velocities[:, 2]

    # Rotate gradient tensors
    if all(key in data_dict for key in vector_quantities["grad_U"]):
        gradients = np.column_stack(
            [[data_dict[f"grad(U):{i}"] for i in range(9)]]
        ).reshape(-1, 3, 3)

        rotated_gradients = np.zeros_like(gradients)
        for i in range(len(gradients)):
            rotated_gradients[i] = np.dot(
                np.dot(rotation_matrix, gradients[i]), rotation_matrix.T
            )

        for i in range(9):
            rotated_data_dict[f"grad(U):{i}"] = rotated_gradients[:, i // 3, i % 3]

    # Rotate vorticity vectors
    if all(key in data_dict for key in vector_quantities["vorticity"]):
        vorticity_vectors = np.column_stack(
            [
                data_dict["vorticity:0"],
                data_dict["vorticity:1"],
                data_dict["vorticity:2"],
            ]
        )
        rotated_vorticity = np.dot(vorticity_vectors, rotation_matrix.T)
        rotated_data_dict["vorticity:0"] = rotated_vorticity[:, 0]
        rotated_data_dict["vorticity:1"] = rotated_vorticity[:, 1]
        rotated_data_dict["vorticity:2"] = rotated_vorticity[:, 2]

    # Rotate wall shear stress vectors
    if all(key in data_dict for key in vector_quantities["wallShearStress"]):
        wallShearStress_vectors = np.column_stack(
            [
                data_dict["wallShearStress:0"],
                data_dict["wallShearStress:1"],
                data_dict["wallShearStress:2"],
            ]
        )
        rotated_wallShearStress = np.dot(wallShearStress_vectors, rotation_matrix.T)
        rotated_data_dict["wallShearStress:0"] = rotated_wallShearStress[:, 0]
        rotated_data_dict["wallShearStress:1"] = rotated_wallShearStress[:, 1]
        rotated_data_dict["wallShearStress:2"] = rotated_wallShearStress[:, 2]

    return rotated_points, rotated_data_dict


def filter_data(points, data_dict, y_num, alpha):
    """Filter data based on specified x and y ranges, then translate back to reference x and y ranges."""

    # reading the csv file with the translation values as df
    df = pd.read_csv(
        Path(project_dir) / "data" / "CFD_slices" / "CFD_translation_values.csv",
        index_col=0,
    )
    # filter on alpha
    df = df[df["alpha"] == alpha]
    # filter on y_num
    df = df[df["Y"] == y_num]
    # Get out the x and y position
    x_offset = df["x"].values[0]
    y_offset = df["y"].values[0]

    # Rotate the data
    rotated_points, rotated_data_dict = rotate_data(points, data_dict, angle_deg=alpha)

    # Apply translation
    rotated_points[:, 0] += x_offset
    rotated_points[:, 1] += y_offset

    # Define reference ranges for filtering
    x_range = (-0.5, 1.1)  # 1.0
    y_range = (-0.5, 0.7)  # 0.6

    # Filter points based on ranges
    x_mask = (rotated_points[:, 0] >= x_range[0]) & (rotated_points[:, 0] <= x_range[1])
    y_mask = (rotated_points[:, 1] >= y_range[0]) & (rotated_points[:, 1] <= y_range[1])
    mask = x_mask & y_mask

    # Apply mask to points and data
    filtered_points = rotated_points[mask]
    filtered_data_dict = {key: data[mask] for key, data in rotated_data_dict.items()}

    # Create final DataFrame
    filtered_df = pd.DataFrame(
        {
            "Points:0": filtered_points[:, 0],
            "Points:1": filtered_points[:, 1],
            "Points:2": filtered_points[:, 2],
            **filtered_data_dict,
        }
    )

    return filtered_df


def process_csv(input_path, output_path, spatial_scale, velocity_scale, y_num, alpha):
    """Process CSV file with scaling, filtering, and header remapping."""

    ### ERIKs definition
    header_mapping = {
        "Points:0": "x",
        "Points:1": "y",
        "Points:2": "z",
        "Time": "time",
        "ReThetat": "ReTheta",
        "U:0": "u",
        "U:1": "v",
        "U:2": "w",
        "gammaInt": "gamma_int",
        "grad(U):0": "dudx",
        "grad(U):1": "dudy",
        "grad(U):2": "dudz",
        "grad(U):3": "dvdx",
        "grad(U):4": "dvdy",
        "grad(U):5": "dvdz",
        "grad(U):6": "dwdx",
        "grad(U):7": "dwdy",
        "grad(U):8": "dwdz",
        "vorticity:2": "vort_z",
        "k": "tke",
        "nut": "nu_t",
        "omega": "omega",
        "p": "pressure",
        "vorticity:0": "vort_x",
        "vorticity:1": "vort_y",
        # "vorticity:2": "vorticity_z",
        "wallShearStress:0": "tau_w_x",
        "wallShearStress:1": "tau_w_y",
        "wallShearStress:2": "tau_w_z",
        "yPlus": "y_plus",
    }

    try:
        # Read the CSV file
        df = pd.read_csv(input_path)

        # Extract spatial points
        points = df[["Points:0", "Points:1", "Points:2"]].values

        # Scale spatial dimensions
        points /= spatial_scale

        # Create a dictionary of all other data
        data_dict = {
            col: df[col].values
            for col in df.columns
            if col not in ["Points:0", "Points:1", "Points:2"]
        }

        # Filter data
        filtered_df = filter_data(points, data_dict, y_num, alpha)

        # Rename columns using the mapping
        filtered_df = filtered_df.rename(columns=header_mapping)

        # Convert DataFrame to numpy array for velocity scaling
        data_array = filtered_df.values
        headers = filtered_df.columns.tolist()
        print(f"headers: {headers}")

        # Scale velocities
        scaled_data = scaling_velocity(data_array, headers, velocity_scale)

        # Create final DataFrame with scaled data
        final_df = pd.DataFrame(scaled_data, columns=headers)

        # Calculate additional quantities
        if all(col in final_df.columns for col in ["u", "v", "w"]):
            final_df["V"] = (
                final_df["u"] ** 2 + final_df["v"] ** 2 + final_df["w"] ** 2
            ) ** 0.5

        if all(col in final_df.columns for col in ["vort_x", "vort_y", "vort_z"]):
            final_df["vorticity_mag"] = (
                final_df["vort_x"] ** 2
                + final_df["vort_y"] ** 2
                + final_df["vort_z"] ** 2
            ) ** 0.5

        ### ERIKs definition
        # Rearange headers to: u, v, w, V, dudx, dudy, dvdx, dvdy, dwdx, dwdy, vort_z, is_valid
        variable_list = [
            "x",
            "y",
            "u",
            "v",
            "w",
            "V",
            "dudx",
            "dudy",
            "dvdx",
            "dvdy",
            "dwdx",
            "dwdy",
            "vort_z",
            "pressure",
            "tau_w_x",
            "tau_w_y",
            # "is_valid",
        ]
        final_df = final_df[variable_list]

        # Adding an is_valid column for consistency with should just be True
        final_df["is_valid"] = True

        # Perform interpolation

        ## From mm to m
        x_global = np.arange(-210, 840, 2.4810164835164836) / 1000
        y_global = np.arange(-205, 405, 2.4810164835164836) / 1000
        print(f"grid-shape: x_global:{x_global.shape}, y_global:{y_global.shape}")
        x_meshgrid_global, y_meshgrid_global = np.meshgrid(x_global, y_global)
        # x_meshgrid_global *= 100
        # y_meshgrid_global *= 100

        # Extract data
        x_loaded = final_df["x"].values
        y_loaded = final_df["y"].values
        points_loaded = np.array([x_loaded, y_loaded]).T

        variable_list_interpolation = [
            # "x",
            # "y",
            "u",
            "v",
            "w",
            "V",
            "dudx",
            "dudy",
            "dvdx",
            "dvdy",
            "dwdx",
            "dwdy",
            "vort_z",
            "pressure",
            "tau_w_x",
            "tau_w_y",
        ]
        interpolated_df = pd.DataFrame()  # New DataFrame for interpolated values

        for var in variable_list_interpolation:
            # print(f"Interpolating {var}")
            # print(f"len(points_loaded): {points_loaded.shape}")
            # print(f"len(final_df[var].values): {len(final_df[var].values)}")
            # print(f"len(x_meshgrid_global): {x_meshgrid_global.shape}")

            interpolated_values = griddata(
                points_loaded,
                final_df[var].values,
                (x_meshgrid_global, y_meshgrid_global),
                method="linear",
            )

            # Store interpolated data in the new DataFrame
            interpolated_df[var] = interpolated_values.flatten()

        # Add x and y coordinates from the meshgrid to the interpolated DataFrame
        interpolated_df["x"] = x_meshgrid_global.flatten()
        interpolated_df["y"] = y_meshgrid_global.flatten()

        # Reorder
        interpolated_df = interpolated_df[variable_list]

        # Save to new CSV
        interpolated_df.to_csv(output_path, index=False)

        print(f"Successfully processed data and saved to {output_path}")
        print(f"Final columns: {', '.join(final_df.columns)}")

        return final_df, interpolated_df

    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main(alpha: int):

    import plotting

    slices_folder = Path(project_dir) / "data" / "CFD_slices" / f"alpha_{alpha}"
    # f"/home/jellepoland/ownCloud/phd/data/V3A/Lebesque_folder/results/1e6/{alpha}/slices"

    # List files in the specified directory
    for file in os.listdir(slices_folder):
        if file.endswith("_0.csv"):
            continue
        y_num = int(file.split("_")[0][1:])
        print(f"\nProcessing file: {file}")
        input_path = Path(slices_folder) / file
        output_path = (
            Path(project_dir)
            / "processed_data"
            / "CFD"
            / f"alpha_{alpha}"
            / f"Y{y_num}_paraview_corrected.csv"
        )
        non_interpolated_df, interpolated_df = process_csv(
            input_path,
            output_path,
            spatial_scale=2.584,
            velocity_scale=15,
            y_num=y_num,
            alpha=alpha,
        )


# Example usage
if __name__ == "__main__":
    main(alpha=6)
