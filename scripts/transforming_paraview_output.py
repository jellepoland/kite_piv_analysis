import pandas as pd
import numpy as np
import os
from pathlib import Path
from utils import project_dir


# def scale_data(points, scale_factor=0.1):
#     """Scale spatial dimensions by the given factor."""
#     scaled_points = points.copy()
#     scaled_points *= scale_factor
#     return scaled_points


def scaling_velocity(data_array, headers, vel_scaling=15):
    """Scale velocity components in the data array by the given factor, ignoring x, y, z columns."""
    # Find the indices of velocity-related columns (anything except 'x', 'y', 'z')
    velocity_indices = [
        i for i, header in enumerate(headers) if header not in ["x", "y", "z"]
    ]
    # Scale the velocity components by the given factor
    data_array[:, velocity_indices] *= vel_scaling
    return data_array


# def filter_data(points, data_dict, y_num):
#     """Filter data based on x and y ranges."""
#     if y_num in [1, 2, 3, 4]:
#         x_range = (-3, 10)
#         y_range = (-3.5, 3.5)
#     elif y_num == 5:
#         x_range = (-3, 10)
#         y_range = (-3.5, 3.5)
#     elif y_num == 6:
#         x_range = (-3, 10)
#         y_range = (-3.5, 3.5)
#     elif y_num == 7:
#         x_range = (-3, 10)
#         y_range = (-5, 1)

#     x_mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
#     y_mask = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
#     mask = x_mask & y_mask

#     filtered_points = points[mask]
#     filtered_data_dict = {key: data[mask] for key, data in data_dict.items()}

#     # Create a new dataframe with filtered data
#     filtered_df = pd.DataFrame(
#         {
#             "Points:0": filtered_points[:, 0],
#             "Points:1": filtered_points[:, 1],
#             "Points:2": filtered_points[:, 2],
#             **filtered_data_dict,
#         }
#     )

#     return filtered_df


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
    import numpy as np

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
    x_range = (-0.2, 0.8)  # 1.0
    y_range = (-0.2, 0.4)  # 0.6

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

    # Define the header mapping dictionary
    ### OLD Jelle Definition
    # header_mapping = {
    #     "Points:0": "x",
    #     "Points:1": "y",
    #     "Points:2": "z",
    #     "Time": "time",
    #     "ReThetat": "ReTheta",
    #     "U:0": "vel_u",
    #     "U:1": "vel_v",
    #     "U:2": "vel_w",
    #     "gammaInt": "gamma_int",
    #     "grad(U):0": "du_dx",
    #     "grad(U):1": "du_dy",
    #     "grad(U):2": "du_dz",
    #     "grad(U):3": "dv_dx",
    #     "grad(U):4": "dv_dy",
    #     "grad(U):5": "dv_dz",
    #     "grad(U):6": "dw_dx",
    #     "grad(U):7": "dw_dy",
    #     "grad(U):8": "dw_dz",
    #     "k": "tke",
    #     "nut": "nu_t",
    #     "omega": "omega",
    #     "p": "pressure",
    #     "vorticity:0": "vorticity_x",
    #     "vorticity:1": "vorticity_y",
    #     "vorticity:2": "vorticity_z",
    #     "wallShearStress:0": "tau_w_x",
    #     "wallShearStress:1": "tau_w_y",
    #     "wallShearStress:2": "tau_w_z",
    #     "yPlus": "y_plus",
    # }
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

        ### OLD Jelle Definition
        # Rearange headers to: x,y,vel_u,vel_v,vel_w,vel_mag,du_dx,du_dy,dv_dx,dv_dy,dw_dx,dw_dy,vorticity_jw_z
        # variable_list = [
        #     "x",
        #     "y",
        #     "vel_u",
        #     "vel_v",
        #     "vel_w",
        #     "vel_mag",
        #     "du_dx",
        #     "du_dy",
        #     "du_dz",
        #     "dv_dx",
        #     "dv_dy",
        #     "dv_dz",
        #     "dw_dx",
        #     "dw_dy",
        #     "vorticity_z",
        #     # "dw_dz",
        #     # "tke",
        #     # "nu_t",
        #     # "omega",
        #     # "pressure",
        #     # "vorticity_x",
        #     # "vorticity_y",
        #     # "tau_w_x",
        #     # "tau_w_y",
        #     # "tau_w_z",
        #     # "y_plus",
        #     # "z",
        # ]
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
            # "is_valid",
        ]
        final_df = final_df[variable_list]

        # Adding an is_valid column for consistency with should just be True
        final_df["is_valid"] = True

        # Perform interpolation
        from scipy.interpolate import griddata

        ## From m to mm
        x_global = np.arange(-210, 840, 2.4810164835164836) / 1000
        y_global = np.arange(-205, 405, 2.4810164835164836) / 1000
        x_meshgrid_global, y_meshgrid_global = np.meshgrid(x_global, y_global)
        # x_meshgrid_global *= 100
        # y_meshgrid_global *= 100

        # Extract data
        x_loaded = final_df["x"].values
        y_loaded = final_df["y"].values
        points_loaded = np.array([x_loaded, y_loaded]).T
        # print(f"x_loaded average: {np.mean(x_loaded)}")
        # print(f"y_loaded average: {np.mean(y_loaded)}")
        # Perform interpolation
        # vel_mag_interp = griddata(
        #     points_loaded, vel_mag, (x_mesh_reduced, y_mesh_reduced), method=method
        # )
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

        return final_df

    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main(alpha: int):

    import scripts.plotting_old as plotting_old

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
        processed_df = process_csv(
            input_path,
            output_path,
            spatial_scale=2.584,
            velocity_scale=15,
            y_num=y_num,
            alpha=alpha,
        )

        plotting_old.main(
            is_CFD=True,
            y_num=y_num,
            alpha=alpha,
            project_dir=project_dir,
            is_with_overlay=False,
            is_with_airfoil=True,
            airfoil_transparency=1.0,
        )


# Example usage
if __name__ == "__main__":
    main(alpha=16)
