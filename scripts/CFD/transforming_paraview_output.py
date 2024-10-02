import pandas as pd
import numpy as np
import os
from pathlib import Path


def scale_data(points, scale_factor=6.5):
    """Scale spatial dimensions by the given factor."""
    scaled_points = points.copy()
    scaled_points *= scale_factor  # Convert from mm to m
    return scaled_points


def scaling_velocity(data_array, headers, vel_scaling=15):
    """Scale velocity components in the data array by the given factor, ignoring x, y, z columns."""
    # Find the indices of velocity-related columns (anything except 'x', 'y', 'z')
    velocity_indices = [
        i for i, header in enumerate(headers) if header not in ["x", "y", "z"]
    ]
    # Scale the velocity components by the given factor
    data_array[:, velocity_indices] *= vel_scaling
    return data_array


def filter_data(points, data_dict, x_range=(-3, 10), y_range=(-3.5, 3.5)):
    """Filter data based on x and y ranges."""
    x_mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
    y_mask = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    mask = x_mask & y_mask

    filtered_points = points[mask]
    filtered_data_dict = {key: data[mask] for key, data in data_dict.items()}

    return filtered_points, filtered_data_dict


def process_csv(
    input_path,
    output_path,
    spatial_scale=6.5,
    velocity_scale=15,
    x_range=(-3, 10),
    y_range=(-3.5, 3.5),
):
    """Process CSV file with scaling, filtering, and header remapping."""

    # Define the header mapping dictionary
    header_mapping = {
        "Points:0": "x",
        "Points:1": "y",
        "Points:2": "z",
        "Time": "time",
        "ReThetat": "ReTheta",
        "U:0": "vel_u",
        "U:1": "vel_v",
        "U:2": "vel_w",
        "gammaInt": "gamma_int",
        "grad(U):0": "du_dx",
        "grad(U):1": "du_dy",
        "grad(U):2": "du_dz",
        "grad(U):3": "dv_dx",
        "grad(U):4": "dv_dy",
        "grad(U):5": "dv_dz",
        "grad(U):6": "dw_dx",
        "grad(U):7": "dw_dy",
        "grad(U):8": "dw_dz",
        "k": "tke",
        "nut": "nu_t",
        "omega": "omega",
        "p": "pressure",
        "vorticity:0": "vorticity_x",
        "vorticity:1": "vorticity_y",
        "vorticity:2": "vorticity_z",
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
        scaled_points = scale_data(points, spatial_scale)

        # Create a dictionary of all other data
        data_dict = {
            col: df[col].values
            for col in df.columns
            if col not in ["Points:0", "Points:1", "Points:2"]
        }

        # Filter data
        filtered_points, filtered_data_dict = filter_data(
            scaled_points, data_dict, x_range, y_range
        )

        # Create a new dataframe with filtered data
        filtered_df = pd.DataFrame(
            {
                "Points:0": filtered_points[:, 0],
                "Points:1": filtered_points[:, 1],
                "Points:2": filtered_points[:, 2],
                **filtered_data_dict,
            }
        )

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
        if all(col in final_df.columns for col in ["vel_u", "vel_v", "vel_w"]):
            final_df["vel_mag"] = (
                final_df["vel_u"] ** 2 + final_df["vel_v"] ** 2 + final_df["vel_w"] ** 2
            ) ** 0.5

        if all(
            col in final_df.columns
            for col in ["vorticity_x", "vorticity_y", "vorticity_z"]
        ):
            final_df["vorticity_mag"] = (
                final_df["vorticity_x"] ** 2
                + final_df["vorticity_y"] ** 2
                + final_df["vorticity_z"] ** 2
            ) ** 0.5

        # Rearange headers to: x,y,vel_u,vel_v,vel_w,vel_mag,du_dx,du_dy,dv_dx,dv_dy,dw_dx,dw_dy,vorticity_jw_z
        variable_list = [
            "x",
            "y",
            "vel_u",
            "vel_v",
            "vel_w",
            "vel_mag",
            "du_dx",
            "du_dy",
            "du_dz",
            "dv_dx",
            "dv_dy",
            "dv_dz",
            "dw_dx",
            "dw_dy",
            "vorticity_z",
            # "dw_dz",
            # "tke",
            # "nu_t",
            # "omega",
            # "pressure",
            # "vorticity_x",
            # "vorticity_y",
            # "tau_w_x",
            # "tau_w_y",
            # "tau_w_z",
            # "y_plus",
            # "z",
        ]
        final_df = final_df[variable_list]

        # Perform interpolation
        from scipy.interpolate import griddata

        x_global = np.arange(-210, 840, 2.4810164835164836)
        y_global = np.arange(-205, 405, 2.4810164835164836)
        x_meshgrid_global, y_meshgrid_global = np.meshgrid(x_global, y_global)
        # x_meshgrid_global *= 100
        # y_meshgrid_global *= 100

        # Extract data
        x_loaded = final_df["x"].values * 100
        y_loaded = final_df["y"].values * 100
        points_loaded = np.array([x_loaded, y_loaded]).T
        print(f"x_loaded average: {np.mean(x_loaded)}")
        print(f"y_loaded average: {np.mean(y_loaded)}")
        # Perform interpolation
        # vel_mag_interp = griddata(
        #     points_loaded, vel_mag, (x_mesh_reduced, y_mesh_reduced), method=method
        # )
        variable_list_interpolation = [
            # "x",
            # "y",
            "vel_u",
            "vel_v",
            "vel_w",
            "vel_mag",
            "du_dx",
            "du_dy",
            "du_dz",
            "dv_dx",
            "dv_dy",
            "dv_dz",
            "dw_dx",
            "dw_dy",
            "vorticity_z",
            # "dw_dz",
            # "tke",
            # "nu_t",
            # "omega",
            # "pressure",
            # "vorticity_x",
            # "vorticity_y",
            # "tau_w_x",
            # "tau_w_y",
            # "tau_w_z",
            # "y_plus",
            # "z",
        ]
        interpolated_df = pd.DataFrame()  # New DataFrame for interpolated values

        for var in variable_list_interpolation:
            print(f"Interpolating {var}")
            print(f"len(points_loaded): {points_loaded.shape}")
            print(f"len(final_df[var].values): {len(final_df[var].values)}")
            print(f"len(x_meshgrid_global): {x_meshgrid_global.shape}")

            interpolated_values = griddata(
                points_loaded,
                final_df[var].values,
                (x_meshgrid_global, y_meshgrid_global),
                method="linear",
            )

            # Store interpolated data in the new DataFrame
            interpolated_df[var] = interpolated_values.flatten()

        # Add x and y coordinates from the meshgrid to the interpolated DataFrame
        interpolated_df["x"] = x_meshgrid_global.flatten() / 1000
        interpolated_df["y"] = y_meshgrid_global.flatten() / 1000

        # Reorder
        interpolated_df = interpolated_df[variable_list]

        # Save to new CSV
        interpolated_df.to_csv(output_path, index=False)

        print(f"Successfully processed data and saved to {output_path}")
        print(f"Spatial scaling factor: {spatial_scale}")
        print(f"Velocity scaling factor: {velocity_scale}")
        print(f"Number of points after filtering: {len(filtered_points)}")
        print(f"Final columns: {', '.join(final_df.columns)}")

        return final_df

    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Example usage
if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    print(f"root_dir: {root_dir}")
    case = "y0"
    input_file = f"/home/jellepoland/ownCloud/phd/data/V3A/Lebesque_folder/results/1e6/6/{case}_1.csv"
    output_file = (
        Path(root_dir) / "processed_data" / "CFD" / f"{case}_paraview_corrected.csv"
    )

    processed_df = process_csv(
        input_file,
        output_file,
        spatial_scale=6.5,
        velocity_scale=15,
        x_range=(-3, 10),
        y_range=(-3.5, 3.5),
    )
