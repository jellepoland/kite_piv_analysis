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


def computing_force_from_surface_pressure_distribution(final_df, y_num):

    ### Calculating force produces
    # # Example filter for surface points based on a specific z-value or condition
    # df_surface = processed_df[
    #     processed_df["y"] < 0.41
    # ]  # Adjust condition as necessary

    # Step 1: Check for wallShearStress as a surface indicator
    # Compute the magnitude of wall shear stress

    final_df["tau_w_mag"] = np.sqrt(final_df["tau_w_x"] ** 2 + final_df["tau_w_y"] ** 2)

    # Filter points where wall shear stress is non-zero
    df_surface = final_df[final_df["tau_w_mag"] > 1e-6]

    print(f"df_surface[x]: {df_surface['x']}, y: {df_surface['y']}")

    # Replace with the actual number of rows and columns
    x = df_surface["x"].values
    y = df_surface["y"].values
    pressure = df_surface["pressure"]

    def calculate_pressure_forces(x, y, pressure):
        # Convert inputs to numpy arrays if they're pandas Series
        x = np.asarray(x)
        y = np.asarray(y)
        pressure = np.asarray(pressure)

        # Calculate segments between points
        dx = np.diff(x)  # Length will be n-1
        dy = np.diff(y)  # Length will be n-1

        # Calculate segment lengths
        ds = np.sqrt(dx**2 + dy**2)  # Length n-1

        # Calculate normal vectors
        nx = dy / ds  # Length n-1
        ny = -dx / ds  # Length n-1

        # Calculate pressure at segments (average of adjacent points)
        p_avg = 0.5 * (pressure[:-1] + pressure[1:])  # Length n-1

        # Calculate force components (all arrays now length n-1)
        fx = p_avg * ds * nx
        fy = p_avg * ds * ny

        # Sum up total forces
        total_fx = np.sum(fx)
        total_fy = np.sum(fy)

        return total_fx, total_fy

    # Calculate forces
    Fx, Fy = calculate_pressure_forces(x, y, pressure)
    print(f"Total force in x direction: {Fx:.2f}")
    print(f"Total force in y direction: {Fy:.2f}")

    # # Compute normals and net forces for the filtered surface points
    # nx, ny = compute_surface_normals_2d(x, y)
    # Fx, Fy = compute_net_force_2d(x, y, pressure)
    rho = 1.2
    U_inf = 15
    c = 0.37
    C_l = Fy / (0.5 * rho * U_inf**2 * c)
    C_d = Fx / (0.5 * rho * U_inf**2 * c)

    print(f"Net force on the surface: C_l = {C_l:.2f}N, C_d = {C_d:.2f}")

    import matplotlib.pyplot as plt

    x = final_df["x"]
    y = final_df["y"]
    pressure = final_df["pressure"]

    plt.scatter(x, y, c=pressure, cmap="viridis", marker="o")
    plt.colorbar(label="Pressure")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Y{y_num} C_l = {C_l:.2f}N, C_d = {C_d:.2f}")
    plt.axis("equal")
    plt.show()


def visualize_interpolated_pressures(df, surface_x, surface_y, interpolated_pressures):
    """
    Visualize interpolated pressures on the airfoil surface

    Parameters:
    -----------
    df : pandas.DataFrame
        Pressure field data with 'x', 'y', 'pressure' columns
    surface_x : numpy.ndarray
        X coordinates of airfoil surface points
    surface_y : numpy.ndarray
        Y coordinates of airfoil surface points

    Returns:
    --------
    numpy.ndarray
        Interpolated pressure values
    """
    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot airfoil surface with interpolated pressures
    scatter = plt.scatter(
        surface_x, surface_y, c=interpolated_pressures, cmap="coolwarm", s=50
    )
    plt.title("Interpolated Pressures on Airfoil Surface")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.colorbar(scatter, label="Interpolated Pressure")

    plt.tight_layout()
    plt.show()


# Compute surface normals
def compute_surface_normals(surface_points):
    """
    Compute surface normals with more robust method

    Parameters:
    -----------
    surface_points : numpy.ndarray
        Array of surface point coordinates

    Returns:
    --------
    numpy.ndarray of surface normals
    """
    normals = []
    for i in range(len(surface_points) - 1):
        # Compute tangent vector
        tangent = surface_points[i + 1] - surface_points[i]

        # Compute perpendicular vector (rotated 90 degrees)
        # Use right-hand rule to determine normal direction
        normal = np.array([-tangent[1], tangent[0]])

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        normals.append(normal)

    return np.array(normals)


def verify_surface_normals(surface_x, surface_y):
    surface_points = np.column_stack((surface_x, surface_y))
    normals = compute_surface_normals(surface_points)

    plt.figure(figsize=(10, 6))
    plt.plot(surface_x, surface_y, "b-", label="Airfoil Surface")

    # Plot a subset of normals for clarity
    skip = max(1, len(normals) // int(len(normals)))  # Plot about 20 normal vectors
    for i in range(0, len(normals), skip):
        midpoint = (surface_points[i] + surface_points[i + 1]) / 2
        plt.quiver(
            midpoint[0],
            midpoint[1],
            normals[i][0],
            normals[i][1],
            color="r",
            scale=10,
            width=0.003,
        )

    plt.title("Airfoil Surface with Normal Vectors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.show()


def compute_surface_forces(df, y_num, alpha):
    from scipy import interpolate
    import numpy as np
    import calculating_airfoil_centre

    # Get surface points
    surface_x = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )  # Example points
    surface_y = np.array(
        [0.01, 0.02, 0.03, 0.02, 0.01, -0.01, -0.02, -0.03, -0.02]
    )  # Example points

    # Flip point order
    surface_x = surface_x[::-1]
    surface_y = surface_y[::-1]

    # Combine surface points
    surface_points = np.column_stack((surface_x, surface_y))

    # Interpolate pressure
    interpolator = interpolate.griddata(
        (df["x"], df["y"]), df["pressure"], surface_points, method="linear"
    )
    surface_normals = compute_surface_normals(surface_points)

    # Force computation
    forces = []
    debug_info = {
        "segment_lengths": [],
        "avg_pressures": [],
        "segment_forces": [],
        "normals": [],
    }

    for i in range(len(surface_points) - 1):
        segment_length = np.linalg.norm(surface_points[i + 1] - surface_points[i])
        debug_info["segment_lengths"].append(segment_length)

        avg_pressure = -(interpolator[i] + interpolator[i + 1]) / 2
        debug_info["avg_pressures"].append(avg_pressure)

        if not np.isnan(avg_pressure):
            segment_force = avg_pressure * segment_length * surface_normals[i]
            forces.append(segment_force)
            debug_info["segment_forces"].append(segment_force)
            debug_info["normals"].append(surface_normals[i])

    total_force = np.sum(forces, axis=0)
    Fx = total_force[0]
    Fy = total_force[1]

    # Compute lift and drag coefficients
    rho = 1.2
    U_inf = 15
    _, _, chord = calculating_airfoil_centre.main(alpha, y_num, is_with_chord=True)
    C_l = Fy / (0.5 * rho * (U_inf**2) * chord)
    C_d = Fx / (0.5 * rho * (U_inf**2) * chord)

    return Fx, Fy, C_l, C_d, debug_info


def running_NOCA(
    df,
    alpha: int,
    y_num: int,
    mu: float = 1.7894e-5,
    is_with_maximim_vorticity_location_correction: bool = True,
    U_inf: float = 15,
):

    import calculating_airfoil_centre
    import force_from_noca
    from utils import reading_optimal_bound_placement
    from defining_bound_volume import boundary_ellipse, boundary_rectangle

    x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
        alpha, y_num, is_with_chord=True
    )
    d1centre = (x_airfoil, y_airfoil)
    drot = 0
    iP = 360

    # Run NOCA analysis
    if alpha == 6:
        rho = 1.20
    else:
        rho = 1.18
    dLx, dLy, iP = reading_optimal_bound_placement(
        alpha, y_num, is_with_N_datapoints=True
    )
    d2curve = boundary_ellipse(
        d1centre,
        drot,
        dLx,
        dLy,
        iP,
    )
    F_x, F_y, C_l, C_d = force_from_noca.main(
        df,
        d2curve,
        mu=mu,
        is_with_maximim_vorticity_location_correction=is_with_maximim_vorticity_location_correction,
        rho=rho,
        U_inf=U_inf,
        c=chord,
    )
    return F_x, F_y, C_l, C_d


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

        df = non_interpolated_df.copy()
        df2 = non_interpolated_df.copy()
        df3 = non_interpolated_df.copy()
        rho = 1
        U_inf = 15
        spatial_scale = 2.584

        ## NOCA
        Fx, Fy, C_l, C_d = running_NOCA(interpolated_df, alpha, y_num)
        print(f"NOCA \nFx: {Fx:.2f}, Fy: {Fy:.2f}, C_l: {C_l:.2f}, C_d: {C_d:.2f}")

        ## no scaling
        fx, fy, cl, cd, debug_info = compute_surface_forces(df, y_num, alpha)
        print(f"\n[p = p] \n Fx: {fx:.2f}, Fy: {fy:.2f}, C_l: {cl:.2f}, C_d: {cd:.2f}")

        ## rho scaling
        df["pressure"] *= rho
        fx, fy, cl, cd, debug_info = compute_surface_forces(df, y_num, alpha)
        print(
            f"\n[p = p/rho] \n Fx: {fx:.2f}, Fy: {fy:.2f}, C_l: {cl:.2f}, C_d: {cd:.2f}"
        )
        df["pressure"] /= rho

        ## Roland scaling
        df["pressure"] *= rho * U_inf**2
        fx, fy, cl, cd, debug_info = compute_surface_forces(df, y_num, alpha)
        print(
            f"\n[p = p * (0.5*rho*Uinf^2)] \n Fx: {fx:.2f}, Fy: {fy:.2f}, C_l: {cl:.2f}, C_d: {cd:.2f}"
        )
        df["pressure"] /= rho * U_inf**2

        ## Spatial scaling
        df["pressure"] *= spatial_scale
        fx, fy, cl, cd, debug_info = compute_surface_forces(df, y_num, alpha)
        print(
            f"\n[p = p * spatial_scale] \n Fx: {fx:.2f}, Fy: {fy:.2f}, C_l: {cl:.2f}, C_d: {cd:.2f}"
        )
        df["pressure"] /= spatial_scale

        ## Spatial^2 scaling
        df["pressure"] *= spatial_scale**2
        fx, fy, cl, cd, debug_info = compute_surface_forces(df, y_num, alpha)
        print(
            f"\n[p = p * spatial_scale^2] \n Fx: {fx:.2f}, Fy: {fy:.2f}, C_l: {cl:.2f}, C_d: {cd:.2f}"
        )
        df["pressure"] /= spatial_scale**2

        ## Spatial^2 / Velocity^2
        df["pressure"] /= (spatial_scale**2) / (U_inf**2)
        fx, fy, cl, cd, debug_info = compute_surface_forces(df, y_num, alpha)
        print(
            f"\n[p = p * spatial_scale^2 / U_inf^2] \n Fx: {fx:.2f}, Fy: {fy:.2f}, C_l: {cl:.2f}, C_d: {cd:.2f}"
        )
        df["pressure"] *= (spatial_scale**2) / (U_inf**2)

        ## Lebesque I scaling
        print(
            f'part 1: {np.sum(df["pressure"] / (0.5 * rho * (U_inf**2)))} part 2: {np.sum((df["V"] ** 2) / (U_inf**2))}'
        )
        df["pressure"] = df["pressure"] / (0.5 * rho * (U_inf**2)) + (
            (df["V"] ** 2) / (U_inf**2)
        )
        fx, fy, cl, cd, debug_info = compute_surface_forces(df, y_num, alpha)
        print(
            f"\n[Cp,t = (p/(0.5*rho*Uinf^2)) + ((V/Uinf)^2)] \n C_l: {fy:.2f}, C_d: {fx:.2f}"
        )

        ## Lebesque I * (1/rho) scaling
        df3["pressure"] = (1 / 1.2) * (
            df3["pressure"] / (0.5 * rho * (U_inf**2)) + ((df["V"] ** 2) / (U_inf**2))
        )
        fx, fy, cl, cd, debug_info = compute_surface_forces(df3, y_num, alpha)
        print(
            f"\n[Cp,t = (1/rho_windtunnel)*(p/(0.5*rho*Uinf^2)) + ((V/Uinf)^2)] \n C_l: {fy:.2f}, C_d: {fx:.2f}"
        )

        ## Lebesque II scaling
        df2["pressure"] = 2 * (df2["pressure"] + 0.5 * ((df2["V"] / U_inf) ** 2))
        fx, fy, cl, cd, debug_info = compute_surface_forces(df2, y_num, alpha)
        print(f"\n[Cp,t = 2*(p+0.5*(V/Uinf)^2)] \n C_l: {fy:.2f}, C_d: {fx:.2f}")

        # # computing_force_from_surface_pressure_distribution(non_interpolated_df, y_num)
        # Fx, Fy, C_l, C_d, debug_info = compute_surface_forces(
        #     interpolated_df, y_num, alpha
        # )
        # plt.figure(figsize=(12, 6))
        # plt.subplot(131)
        # plt.title(
        #     f"Segment Lengths, total_length:{sum(debug_info['segment_lengths']):.2f}m"
        # )
        # plt.plot(debug_info["segment_lengths"])

        # plt.subplot(132)
        # plt.title("Average Pressures")
        # plt.plot(debug_info["avg_pressures"])

        # plt.subplot(133)
        # plt.title("Segment Force Magnitudes")
        # plt.plot([np.linalg.norm(force) for force in debug_info["segment_forces"]])

        # plt.tight_layout()
        # plt.show()

        # print(f"\n NEW <> Fx: {Fx:.2f}, Fy: {Fy:.2f}, C_l: {C_l:.2f}, C_d: {C_d:.2f}")
        # Fx, Fy, C_l, C_d = running_NOCA(interpolated_df, alpha, y_num)
        # print(f"\n NOCA <> Fx: {Fx:.2f}, Fy: {Fy:.2f}, C_l: {C_l:.2f}, C_d: {C_d:.2f}")


# Example usage
if __name__ == "__main__":
    main(alpha=6)
