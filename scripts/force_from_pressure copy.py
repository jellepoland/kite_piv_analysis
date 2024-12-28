import numpy as np
from pathlib import Path
import pandas as pd
from scipy.signal import convolve2d
from utils import project_dir, reshape_remove_nans, interp2d_batch, csv_reader
from defining_bound_volume import boundary_ellipse, boundary_rectangle


def circshift(array, shift):
    """
    Circularly shift the elements of an array similar to MATLAB's circshift.

    Parameters:
    - array: np.ndarray
        The input array to be shifted.
    - shift: int
        The number of positions to shift. Positive values shift to the right,
        and negative values shift to the left.

    Returns:
    - np.ndarray: The circularly shifted array.
    """
    return np.roll(array, shift)


def conv2(x, y, mode="same"):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def smooth_data(data, ismooth):
    kernel = np.ones((ismooth, ismooth)) / (ismooth**2)
    return conv2(data, kernel, mode="same")


# def calculate_force_unsteady(
#     coordinates,
#     velocity,
#     pressure,
#     normal_direction,
#     velocity_tp1,
#     velocity_tm1,
#     dt,
#     stress_tensor,
#     miu,
#     velocity_derivatives,
#     rho,
# ):
#     """
#     Calculate forces for unsteady flow conditions.

#     Parameters:
#     -----------
#     coordinates : np.ndarray
#         Array of shape (3, n) containing x, y, z coordinates
#     velocity : np.ndarray
#         Array of shape (3, n) containing current velocity components
#     pressure : np.ndarray
#         Array of shape (n,) containing pressure values
#     normal_direction : np.ndarray
#         Array of shape (3, n) containing normal direction components
#     velocity_tp1 : np.ndarray
#         Array of shape (3, n) containing velocity at time t+1
#     velocity_tm1 : np.ndarray
#         Array of shape (3, n) containing velocity at time t-1
#     dt : float
#         Time step
#     stress_tensor : np.ndarray
#         Array of shape (4, n) containing stress tensor components
#     miu : float
#         Dynamic viscosity
#     velocity_derivatives : np.ndarray
#         Array of shape (4, n) containing velocity derivatives
#     rho : float
#         Fluid density

#     Returns:
#     --------
#     tuple
#         (force, force_components)
#     """
#     # Space dimensions
#     N = 3

#     # Extract normal direction components
#     n1 = normal_direction[0, :]
#     n2 = normal_direction[1, :]
#     n3 = normal_direction[2, :]

#     # Extract coordinates
#     x = coordinates[0, :]
#     y = coordinates[1, :]
#     z = coordinates[2, :]

#     # Extract velocity components at current time
#     u = velocity[0, :]
#     v = velocity[1, :]
#     w = velocity[2, :]

#     # Extract velocity components at t+1
#     u_tp1 = velocity_tp1[0, :]
#     v_tp1 = velocity_tp1[1, :]
#     w_tp1 = velocity_tp1[2, :]

#     # Extract velocity components at t-1
#     u_tm1 = velocity_tm1[0, :]
#     v_tm1 = velocity_tm1[1, :]
#     w_tm1 = velocity_tm1[2, :]

#     # Process pressure (delta pressure calculation)
#     p = pressure - 1020.56 * 100

#     # Extract stress tensor components (2D)
#     Txx = stress_tensor[0, :]
#     Txy = stress_tensor[1, :]
#     Tyy = stress_tensor[1, :]  # Note: Using same index as MATLAB code

#     # Extract velocity derivatives
#     dudx = velocity_derivatives[0, :]
#     dudy = velocity_derivatives[1, :]
#     dvdx = velocity_derivatives[2, :]
#     dvdy = velocity_derivatives[3, :]

#     # Calculate force components

#     # Mean convection term (f1)
#     print("n1 shape:", n1.shape)
#     print("u shape:", u.shape)
#     print("v shape:", v.shape)
#     print("w shape:", w.shape)
#     print("n2 shape:", n2.shape)
#     print("n3 shape:", n3.shape)
#     f1 = np.array(
#         [
#             -n1 * (u**2) - n2 * u * v - n3 * u * w,
#             -n1 * u * v - n2 * (v**2) - n3 * v * w,
#             -n1 * u * w - n2 * v * w - n3 * (w**2),
#         ]
#     )

#     # Pressure term (f2)
#     f2 = np.array([-n1 * p, -n2 * p, -n3 * p])

#     # Stress tensor term (f4) - 2D case
#     f4 = np.array([-n1 * Txx - n2 * Txy, -n1 * Txy - n2 * Tyy, np.zeros_like(Txy)])

#     # Viscous stress term (f5) - 2D case
#     f5 = np.array(
#         [
#             n1 * (2 * dudx) + n2 * (dudy + dvdx),
#             n1 * (dvdx + dudy) + n2 * (2 * dvdy),
#             np.zeros_like(dvdy),
#         ]
#     )

#     # Time derivative terms
#     f3_tp1 = np.array(
#         [
#             -n1 * (x * u_tp1) - n2 * x * v_tp1 - n3 * x * w_tp1,
#             -n1 * u_tp1 * y - n2 * (v_tp1 * y) - n3 * y * w_tp1,
#             -n1 * u_tp1 * z - n2 * v_tp1 * z - n3 * (w_tp1 * z),
#         ]
#     )

#     f3_tm1 = np.array(
#         [
#             -n1 * (x * u_tm1) - n2 * x * v_tm1 - n3 * x * w_tm1,
#             -n1 * u_tm1 * y - n2 * (v_tm1 * y) - n3 * y * w_tm1,
#             -n1 * u_tm1 * z - n2 * v_tm1 * z - n3 * (w_tm1 * z),
#         ]
#     )

#     f3 = (f3_tp1 - f3_tm1) / dt

#     # Calculate total force
#     force = (f1 + f4) * rho + f2 + f3 * rho + f5 * miu

#     # Calculate force components
#     TD_term = f3 * rho  # time derivative
#     MC_term = f1 * rho  # mean convection term
#     TMC_term = f4 * rho  # turbulent momentum transfer
#     p_term = f2  # pressure term
#     VS_term = f5 * miu  # mean viscous stress term

#     force_components = np.vstack([TD_term, MC_term, TMC_term, p_term, VS_term])

#     return force, force_components


def calculate_force_guanqun(
    d2x,
    d2y,
    d2u,
    d2v,
    d2dudx,
    d2dudy,
    d2dvdx,
    d2dvdy,
    d2curve,
    d2pressure,
    mu,
    rho=1.0,
    U_inf=1,
):
    # scaling back
    d2u = d2u
    d2v = d2v

    # Ensure inputs are NumPy arrays
    d2curve = np.asarray(d2curve)
    # Normal vector calculation - MATLAB style
    n1 = np.gradient(d2curve[:, 1]) / np.sqrt(
        np.gradient(d2curve[:, 0]) ** 2 + np.gradient(d2curve[:, 1]) ** 2
    )
    n2 = -np.gradient(d2curve[:, 0]) / np.sqrt(
        np.gradient(d2curve[:, 0]) ** 2 + np.gradient(d2curve[:, 1]) ** 2
    )
    n3 = np.zeros_like(n1)
    u = interp2d_batch(d2x, d2y, d2u, d2curve)
    v = interp2d_batch(d2x, d2y, d2v, d2curve)
    w = np.zeros_like(u)
    V = np.sqrt(u**2 + v**2)
    # Pressure
    d1p = interp2d_batch(d2x, d2y, d2pressure, d2curve)
    p = d1p - 0.5 * 15**2
    # p = (d1p / (0.5 * rho * U_inf**2)) + (V**2 / U_inf**2)
    ### Spatial gradients of first and second order
    ddx = d2x[1, 1] - d2x[0, 0]
    ddy = d2y[1, 1] - d2y[0, 0]
    d2dudx, d2dudy = np.gradient(d2u, ddx, ddy)
    d2dvdx, d2dvdy = np.gradient(d2v, ddx, ddy)
    dudx = interp2d_batch(d2x, d2y, d2dudx, d2curve)
    dudy = interp2d_batch(d2x, d2y, d2dudy, d2curve)
    dvdx = interp2d_batch(d2x, d2y, d2dvdx, d2curve)
    dvdy = interp2d_batch(d2x, d2y, d2dvdy, d2curve)
    # Tensors
    Txx = -mu * (2 * dudx)
    Txy = -mu * (dudy + dvdx)
    Tyy = -mu * (2 * dvdy)

    # Mean convection term (f1)
    # for var in [n1, n2, n3, u, v, w, p, Txx, Txy, Tyy, dudx, dudy, dvdx, dvdy]:
    # print(
    # f"mean: {np.mean(var):.2f}, min: {np.min(var):.2f}, max: {np.max(var):.2f}"
    # )

    f1 = np.array(
        [
            -n1 * (u**2) - n2 * u * v - n3 * u * w,
            -n1 * u * v - n2 * (v**2) - n3 * v * w,
            -n1 * u * w - n2 * v * w - n3 * (w**2),
        ]
    )

    # Pressure term (f2)
    f2 = np.array([-n1 * p, -n2 * p, -n3 * p])

    # Stress tensor term (f4) - 2D case
    f4 = np.array([-n1 * Txx - n2 * Txy, -n1 * Txy - n2 * Tyy, np.zeros_like(Txy)])

    # Viscous stress term (f5) - 2D case
    f5 = np.array(
        [
            n1 * (2 * dudx) + n2 * (dudy + dvdx),
            n1 * (dvdx + dudy) + n2 * (2 * dvdy),
            np.zeros_like(dvdy),
        ]
    )
    # Calculate total force
    force = (f1 + f4) * rho + f2 + f5 * mu

    # # Calculate force components
    # TD_term = np.zeros_like(f1)  # time derivative
    # MC_term = f1 * rho  # mean convection term
    # TMC_term = f4 * rho  # turbulent momentum transfer
    # p_term = f2  # pressure term
    # VS_term = f5 * mu  # mean viscous stress term

    f1_mean_convection = np.sum(f1, axis=1) * rho
    f2_pressure = np.sum(f2, axis=1)
    f3_unsteady = np.zeros_like(f1_mean_convection)
    f4_turbulent_moment_transfer = np.sum(f4, axis=1) * rho
    f5_viscous_stress_term = np.sum(f5, axis=1) * mu
    ftotal = (
        f1_mean_convection
        + f2_pressure
        + f3_unsteady
        + f4_turbulent_moment_transfer
        + f5_viscous_stress_term
    )
    print(f"f1_mean_convection    : {f1_mean_convection}")
    print(f"f2_pressure : {f2_pressure}")
    print(f"f3_unsteady : {f3_unsteady}")
    print(f"f4_turbulent_moment_transfer: {f4_turbulent_moment_transfer}")
    print(f"f5_viscous_stress_term  : {f5_viscous_stress_term}")
    print(f"Total force : {ftotal}")

    force_components = np.vstack(
        [
            f1_mean_convection,
            f2_pressure,
            f3_unsteady,
            f4_turbulent_moment_transfer,
            f5_viscous_stress_term,
        ]
    )

    return force, force_components


def main(
    df_1D: pd.DataFrame,
    d2curve: np.ndarray,
    mu: float = 1.7894e-5,
    is_with_maximim_vorticity_location_correction: bool = True,
    rho: float = 1,
    U_inf: float = 15,
    c: float = 0.398347,
):
    # Reshape data
    n_rows = len(np.unique(df_1D["y"].values))
    n_cols = len(np.unique(df_1D["x"].values))

    # Reshape all relevant fields
    d2x = reshape_remove_nans(df_1D["x"], n_rows, n_cols)
    d2y = reshape_remove_nans(df_1D["y"], n_rows, n_cols)
    d2u = reshape_remove_nans(df_1D["u"], n_rows, n_cols)
    d2v = reshape_remove_nans(df_1D["v"], n_rows, n_cols)
    d2dudx = reshape_remove_nans(df_1D["dudx"], n_rows, n_cols)
    d2dudy = reshape_remove_nans(df_1D["dudy"], n_rows, n_cols)
    d2dvdx = reshape_remove_nans(df_1D["dvdx"], n_rows, n_cols)
    d2dvdy = reshape_remove_nans(df_1D["dvdy"], n_rows, n_cols)
    d2vort_z = reshape_remove_nans(df_1D["vort_z"], n_rows, n_cols)

    # Get pressure
    d2pressure = reshape_remove_nans(df_1D["pressure"], n_rows, n_cols)

    force, force_components = calculate_force_guanqun(
        d2x,
        d2y,
        d2u,
        d2v,
        d2dudx,
        d2dudy,
        d2dvdx,
        d2dvdy,
        d2curve,
        d2pressure,
        mu=mu,
        rho=1,
        U_inf=U_inf,
    )
    total_force = np.sum(force, axis=1) / 100
    print(f"P-calc Fx: {total_force[0]:.3f} N")
    print(f"P-calc Fy: {total_force[1]:.3f} N")

    # print(f"Force: {force}")

    # Continue with your existing NOCA calculation
    from force_from_noca import forceFromVelNoca2D_V3

    d1Fn, d1Ft = forceFromVelNoca2D_V3(
        d2x=d2x,
        d2y=d2y,
        d2u=d2u,
        d2v=d2v,
        d2vortZ=d2vort_z,
        d2dudt=np.zeros_like(d2x),  # zero for steady flow
        d2dvdt=np.zeros_like(d2x),  # zero for steady flow
        d2curve=d2curve,
        dmu=mu,
        bcorMaxVort=is_with_maximim_vorticity_location_correction,
    )

    F_x = d1Fn[0]
    F_y = d1Ft[0]
    C_l = F_y / (0.5 * rho * U_inf**2 * c)
    C_d = F_x / (0.5 * rho * U_inf**2 * c)

    print(f"NOCA   Fx: {d1Fn[0]:.3f}N (F_n Normal force)")
    print(f"NOCA   Fy: {d1Ft[0]:.3f}N (F_t Tangential force)")

    q_infc = 0.5 * rho * U_inf**2 * c
    print(f"\nq_inf: {q_infc:.3f}")
    print(f"P-calc Cl: {total_force[1]/q_infc:.3f}")
    print(f"P-calc Cd: {total_force[0]/q_infc:.3f}")
    print(f"NOCA   Cl: {F_y/q_infc:.3f}")
    print(f"NOCA   Cd: {F_x/q_infc:.3f}")

    return F_x, F_y, C_l, C_d


def process_csv(input_path, output_path, spatial_scale, velocity_scale, y_num, alpha):
    """Process CSV file with scaling, filtering, and header remapping."""

    from scipy.interpolate import griddata
    from transforming_paraview_output import filter_data, scaling_CFD, rotate_data

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
        # print(f"headers: {headers}")

        # Scale velocities
        scaled_data = scaling_CFD(data_array, headers, velocity_scale)

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
        # print(f"grid-shape: x_global:{x_global.shape}, y_global:{y_global.shape}")
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
        return interpolated_df
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":

    spatial_scale = 2.584
    velocity_scale = 15
    Re = 1e6
    mu = ((1 / 2.584) * velocity_scale) / Re
    mu = 1e-6
    y_num = 1
    alpha = 6
    print(f"mu: {mu}")
    input_path = (
        Path(project_dir) / "data" / "CFD_slices" / "alpha_6" / f"Y{y_num}_1.csv"
    )
    output_path = " "
    interpolated_df = process_csv(
        input_path, output_path, spatial_scale, velocity_scale, y_num, alpha
    )
    df_1D = interpolated_df.copy()
    df_1D["pressure"] = df_1D["pressure"]  # * (velocity_scale**2) / (spatial_scale**2)

    # df_1D = csv_reader(is_CFD=True, alpha=6, y_num=1, alpha_d_rod=7.25)
    ### Running for Ellipse ###
    0.22117388, 0.159773525

    is_ellipse = True
    d1centre = np.array([0.22117388, 0.159773525])
    drot = 0
    dLx = 0.65
    dLy = 0.37
    iP = 360

    # create d2curve
    if is_ellipse:
        d2curve = boundary_ellipse(d1centre, drot, dLx, dLy, iP)
        # print(f"Running NOCA on Ellipse, will take a while...")
    else:
        d2curve = boundary_rectangle(d1centre, drot, dLx, dLy, iP)
        # print(f"Running NOCA on Rectangle, will take a while...")

    print(f"Running for Ellipse")
    main(
        df_1D,
        d2curve,
        mu=mu,
        is_with_maximim_vorticity_location_correction=True,
    )

    ### Running for Rectangle ###
    is_ellipse = False
    d1centre = np.array([0.22117388, 0.159773525])
    drot = 0
    dLx = 0.65
    dLy = 0.37
    iP = 360

    # create d2curve
    if is_ellipse:
        d2curve = boundary_ellipse(d1centre, drot, dLx, dLy, iP)
        # print(f"Running NOCA on Ellipse, will take a while...")
    else:
        d2curve = boundary_rectangle(d1centre, drot, dLx, dLy, iP)
        # print(f"Running NOCA on Rectangle, will take a while...")

    print(f"Running for Rectangle")
    main(
        df_1D,
        d2curve,
        # mu=1.7894e-5,
        mu=mu,
        is_with_maximim_vorticity_location_correction=True,
    )
