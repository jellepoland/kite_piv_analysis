import pandas as pd
import numpy as np
import os
import scipy.interpolate as interpolate
from scipy.interpolate import griddata
from pathlib import Path
from utils import project_dir, interp2d_batch, csv_reader
import matplotlib.pyplot as plt

# def computing_force_from_surface_pressure_distribution(final_df, y_num):

#     ### Calculating force produces
#     # # Example filter for surface points based on a specific z-value or condition
#     # df_surface = processed_df[
#     #     processed_df["y"] < 0.41
#     # ]  # Adjust condition as necessary

#     # Step 1: Check for wallShearStress as a surface indicator
#     # Compute the magnitude of wall shear stress

#     final_df["tau_w_mag"] = np.sqrt(final_df["tau_w_x"] ** 2 + final_df["tau_w_y"] ** 2)

#     # Filter points where wall shear stress is non-zero
#     df_surface = final_df[final_df["tau_w_mag"] > 1e-6]

#     print(f"df_surface[x]: {df_surface['x']}, y: {df_surface['y']}")

#     # Replace with the actual number of rows and columns
#     x = df_surface["x"].values
#     y = df_surface["y"].values
#     pressure = df_surface["pressure"]

#     def calculate_pressure_forces(x, y, pressure):
#         # Convert inputs to numpy arrays if they're pandas Series
#         x = np.asarray(x)
#         y = np.asarray(y)
#         pressure = np.asarray(pressure)

#         # Calculate segments between points
#         dx = np.diff(x)  # Length will be n-1
#         dy = np.diff(y)  # Length will be n-1

#         # Calculate segment lengths
#         ds = np.sqrt(dx**2 + dy**2)  # Length n-1

#         # Calculate normal vectors
#         nx = dy / ds  # Length n-1
#         ny = -dx / ds  # Length n-1

#         # Calculate pressure at segments (average of adjacent points)
#         p_avg = 0.5 * (pressure[:-1] + pressure[1:])  # Length n-1

#         # Calculate force components (all arrays now length n-1)
#         fx = p_avg * ds * nx
#         fy = p_avg * ds * ny

#         # Sum up total forces
#         total_fx = np.sum(fx)
#         total_fy = np.sum(fy)

#         return total_fx, total_fy

#     # Calculate forces
#     Fx, Fy = calculate_pressure_forces(x, y, pressure)
#     print(f"Total force in x direction: {Fx:.4f}")
#     print(f"Total force in y direction: {Fy:.4f}")

#     # # Compute normals and net forces for the filtered surface points
#     # nx, ny = compute_surface_normals_2d(x, y)
#     # Fx, Fy = compute_net_force_2d(x, y, pressure)
#     rho = 1.2
#     U_inf = 15
#     c = 0.37
#     C_l = Fy / (0.5 * rho * U_inf**2 * c)
#     C_d = Fx / (0.5 * rho * U_inf**2 * c)

#     print(f"Net force on the surface: C_l = {C_l:.4f}N, C_d = {C_d:.4f}")

#     import matplotlib.pyplot as plt

#     x = final_df["x"]
#     y = final_df["y"]
#     pressure = final_df["pressure"]

#     plt.scatter(x, y, c=pressure, cmap="viridis", marker="o")
#     plt.colorbar(label="Pressure")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title(f"Y{y_num} C_l = {C_l:.4f}N, C_d = {C_d:.4f}")
#     plt.axis("equal")
#     plt.show()


def visualize_interpolated_pressures(surface_x, surface_y, interpolated_pressures):
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


# # Compute surface normals
# def compute_surface_normals(surface_points):
#     """
#     Compute surface normals with more robust method

#     Parameters:
#     -----------
#     surface_points : numpy.ndarray
#         Array of surface point coordinates

#     Returns:
#     --------
#     numpy.ndarray of surface normals
#     """
#     normals = []
#     for i in range(len(surface_points) - 1):
#         # Compute tangent vector
#         tangent = surface_points[i + 1] - surface_points[i]

#         # Compute perpendicular vector (rotated 90 degrees)
#         # Use right-hand rule to determine normal direction
#         normal = np.array([-tangent[1], tangent[0]])

#         # Normalize the normal vector
#         normal = normal / np.linalg.norm(normal)

#         normals.append(normal)

#     return np.array(normals)


def verify_surface_normals(surface_x, surface_y):
    surface_points = np.column_stack((surface_x, surface_y))
    normals, _ = compute_surface_normals(surface_points)

    plt.figure(figsize=(10, 6))
    plt.plot(surface_x, surface_y, "b-", label="Airfoil Surface")

    # Plot a subset of normals for clarity
    skip = max(1, len(normals) // int(len(normals)))
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


# def compute_surface_forces(df, surface_x, surface_y, y_num, alpha, is_plot=False):
#     from scipy import interpolate
#     import numpy as np
#     import calculating_airfoil_centre

#     # Flip point order
#     surface_x = surface_x[::-1]
#     surface_y = surface_y[::-1]

#     # Combine surface points
#     surface_points = np.column_stack((surface_x, surface_y))

#     # Interpolate pressure
#     interpolator = interpolate.griddata(
#         (df["x"], df["y"]), df["pressure"], surface_points, method="linear"
#     )
#     surface_normals = compute_surface_normals(surface_points)

#     # Visualising
#     if is_plot:
#         verify_surface_normals(surface_x, surface_y)
#         visualize_interpolated_pressures(surface_x, surface_y, interpolator)

#     # Force computation
#     forces = []
#     debug_info = {
#         "segment_lengths": [],
#         "avg_pressures": [],
#         "segment_forces": [],
#         "normals": [],
#     }

#     for i in range(len(surface_points) - 1):
#         segment_length = np.linalg.norm(surface_points[i + 1] - surface_points[i])
#         debug_info["segment_lengths"].append(segment_length)

#         avg_pressure = -(interpolator[i] + interpolator[i + 1]) / 2
#         debug_info["avg_pressures"].append(avg_pressure)

#         if not np.isnan(avg_pressure):
#             segment_force = avg_pressure * segment_length * surface_normals[i]
#             forces.append(segment_force)
#             debug_info["segment_forces"].append(segment_force)
#             debug_info["normals"].append(surface_normals[i])

#     total_force = np.sum(forces, axis=0)
#     Fx = total_force[0]
#     Fy = total_force[1]

#     # Compute lift and drag coefficients
#     rho = 1.2
#     U_inf = 15
#     x_airfoil_center, z_airfoil_center, chord = calculating_airfoil_centre.main(
#         alpha, y_num, is_with_chord=True
#     )
#     C_l = Fy / (0.5 * rho * (U_inf**2) * chord)
#     C_d = Fx / (0.5 * rho * (U_inf**2) * chord)

#     return Fx, Fy, C_l, C_d, debug_info, chord, x_airfoil_center, z_airfoil_center


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


def compute_surface_normals(surface_points):
    """
    Compute surface normals with a more robust method.

    Parameters:
    -----------
    surface_points : numpy.ndarray
        Array of surface point coordinates.

    Returns:
    --------
    numpy.ndarray of surface normals and segment lengths.
    """
    normals = []
    segment_lengths = []
    for i in range(len(surface_points) - 1):
        # Compute tangent vector
        tangent = surface_points[i + 1] - surface_points[i]

        # Compute perpendicular vector (rotated 90 degrees)
        # Use right-hand rule to determine normal direction
        normal = np.array([-tangent[1], tangent[0]])

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Append normal and segment length
        normals.append(normal)
        segment_lengths.append(np.linalg.norm(tangent))

    return np.array(normals), np.array(segment_lengths)


def compute_surface_forces(
    df,
    mu,
    density,
    x_surface,
    y_surface,
    p_ref=0,
    is_plot=False,
    chord=0.39,
):
    """
    Compute aerodynamic forces on a 2D airfoil surface from CFD data.
    """
    surface_points = np.column_stack((x_surface, y_surface))

    # Precompute normals and segment lengths
    normals, segment_lengths = compute_surface_normals(surface_points)

    # Interpolate grid data onto surface nodes
    pressure_surface = griddata(
        (df["x"], df["y"]), df["pressure"], surface_points, method="linear"
    )
    dudx_surface = griddata(
        (df["x"], df["y"]), df["dudx"], surface_points, method="linear"
    )
    dudy_surface = griddata(
        (df["x"], df["y"]), df["dudy"], surface_points, method="linear"
    )
    dvdx_surface = griddata(
        (df["x"], df["y"]), df["dvdx"], surface_points, method="linear"
    )
    dvdy_surface = griddata(
        (df["x"], df["y"]), df["dvdy"], surface_points, method="linear"
    )

    # Initialize force components
    Fx_p, Fy_p = 0, 0  # Pressure forces
    Fx_v, Fy_v = 0, 0  # Viscous forces

    # Loop through surface nodes
    for i in range(len(normals)):
        normal = normals[i]
        length = segment_lengths[i]

        # Pressure force contribution
        pressure_difference = -pressure_surface[i] - p_ref
        Fx_p += density * normal[0] * pressure_difference * length
        Fy_p += density * normal[1] * pressure_difference * length

        # Deviatoric stress tensor components (approximated)
        R_dev_x = 2 * mu * dudx_surface[i]
        R_dev_y = 2 * mu * dvdy_surface[i]
        shear_xy = mu * (dudy_surface[i] + dvdx_surface[i])

        # Viscous force contribution
        Fx_v += (R_dev_x * normal[0] + shear_xy * normal[1]) * length
        Fy_v += (shear_xy * normal[0] + R_dev_y * normal[1]) * length

    # Combine contributions

    Fx_total = Fx_p + Fx_v
    Fy_total = Fy_p + Fy_v
    q_infc = 0.5 * density * (15**2) * chord
    # print(f"Fx_p: {Fx_p:.2f}, Fy_p: {Fy_p:.2f}, Fx_v: {Fx_v:.4f}, Fy_v: {Fy_v:.4f}")
    print(
        f"P-integration --- Fx: {Fx_total:.2f}, Fy: {Fy_total:.2f}, C_l: {Fy_total/q_infc:.2f}, C_d: {Fx_total/q_infc:.2f}"
    )

    # Visualizing
    if is_plot:
        verify_surface_normals(x_surface, y_surface)
        visualize_interpolated_pressures(x_surface, y_surface, pressure_surface)

        surface_x_airfoil = x_surface
        surface_y_airfoil = y_surface

        # extract surface from a filter based on tau_w
        df["tau_w_mag"] = np.sqrt(df["tau_w_x"] ** 2 + df["tau_w_y"] ** 2)
        df_surface = df[df["tau_w_mag"] > 1e-3]
        surface_x_tau_w = df_surface["x"].values
        surface_y_tau_w = df_surface["y"].values
        plt.plot(
            surface_x_airfoil,
            surface_y_airfoil,
            "r",
            label="Airfoil Surface",
        )
        plt.scatter(surface_x_tau_w, surface_y_tau_w, c="b", label="Airfoil Surface")
        plt.axis("equal")
        plt.show()

    return Fx_total, Fy_total


def main():

    import plotting
    from transforming_paraview_output import process_csv
    import calculating_airfoil_centre

    # slices_folder = Path(project_dir) / "data" / "CFD_slices" / f"alpha_{alpha}"
    # f"/home/jellepoland/ownCloud/phd/data/V3A/Lebesque_folder/results/1e6/{alpha}/slices"

    spatial_scale = 2.584
    velocity_scale = 15
    rho_scaling = 1.2
    rho = 1.2
    U_inf = 15
    Re = 1e6
    p_ref = 0
    is_with_plot = False

    # # List files in the specified directory
    # for file in os.listdir(slices_folder):
    #     if file.endswith("_0.csv"):
    #         continue
    #     # if not file.endswith("Y1_1.csv"):
    #     #     continue
    #     y_num = int(file.split("_")[0][1:])
    #     print(f"\nProcessing file: {file}")
    #     input_path = Path(slices_folder) / file
    #     output_path = None
    #     # interpolated df
    #     df = process_csv(
    #         input_path,
    #         output_path,
    #         spatial_scale=spatial_scale,
    #         velocity_scale=velocity_scale,
    #         rho_scaling=rho_scaling,
    #         y_num=y_num,
    #         alpha=alpha,
    #     )
    alpha = 6
    for y_num in [1, 2, 3, 4, 5]:
        df = csv_reader(is_CFD=True, alpha=alpha, y_num=y_num, alpha_d_rod=7)
        # Reading in the airfoil centers
        x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
            alpha, y_num, is_with_chord=True
        )
        mu = (rho * U_inf * chord) / Re
        ## Use actual airfoils as surface points
        surface_x, surface_y = plotting.plot_airfoil(
            None, {"y_num": y_num, "alpha": alpha}, is_return_surface_points=True
        )
        # Flip point order
        surface_x = surface_x[::-1]
        surface_y = surface_y[::-1]
        ## NOCA
        Fx, Fy, C_l, C_d = running_NOCA(df, alpha, y_num)
        print(
            f"NOCA          --- Fx: {Fx:.2f}, Fy: {Fy:.2f}, C_l: {C_l:.2f}, C_d: {C_d:.2f}"
        )

        ### Compute surface forces
        Fx_total, Fy_total = compute_surface_forces(
            df,
            mu=mu,
            density=rho,
            x_surface=surface_x,
            y_surface=surface_y,
            p_ref=p_ref,
            is_plot=is_with_plot,
        )
    alpha = 16
    for y_num in [1]:
        df = csv_reader(is_CFD=True, alpha=alpha, y_num=y_num, alpha_d_rod=7)
        # Reading in the airfoil centers
        x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
            alpha, y_num, is_with_chord=True
        )
        mu = (rho * U_inf * chord) / Re
        ## Use actual airfoils as surface points
        surface_x, surface_y = plotting.plot_airfoil(
            None, {"y_num": y_num, "alpha": alpha}, is_return_surface_points=True
        )
        # Flip point order
        surface_x = surface_x[::-1]
        surface_y = surface_y[::-1]
        ## NOCA
        Fx, Fy, C_l, C_d = running_NOCA(df, alpha, y_num)
        print(
            f"NOCA          --- Fx: {Fx:.2f}, Fy: {Fy:.2f}, C_l: {C_l:.2f}, C_d: {C_d:.2f}"
        )

        ### Compute surface forces
        Fx_total, Fy_total = compute_surface_forces(
            df,
            mu=mu,
            density=rho,
            x_surface=surface_x,
            y_surface=surface_y,
            p_ref=p_ref,
            is_plot=is_with_plot,
        )

        # ## no scaling
        # fx, fy, cl, cd, debug_info, chord, x_airfoil_center, z_airfoil_center = (
        #     compute_surface_forces(
        #         df, surface_x, surface_y, y_num, alpha, is_plot=is_with_plot
        #     )
        # )
        # print(f"\n[p = p] \n Fx: {fx:.4f}, Fy: {fy:.4f}, C_l: {cl:.4f}, C_d: {cd:.4f}")

        # ## rho scaling
        # df["pressure"] *= rho
        # fx, fy, cl, cd, debug_info, chord, x_airfoil_center, z_airfoil_center = (
        #     compute_surface_forces(df, surface_x, surface_y, y_num, alpha)
        # )
        # print(
        #     f"\n[p = p/rho] \n Fx: {fx:.4f}, Fy: {fy:.4f}, C_l: {cl:.4f}, C_d: {cd:.4f}"
        # )
        # df["pressure"] /= rho

        # ## Roland scaling
        # df["pressure"] *= rho * U_inf**2
        # fx, fy, cl, cd, debug_info, chord, x_airfoil_center, z_airfoil_center = (
        #     compute_surface_forces(df, surface_x, surface_y, y_num, alpha)
        # )
        # print(
        #     f"\n[p = p * (0.5*rho*Uinf^2)] \n Fx: {fx:.4f}, Fy: {fy:.4f}, C_l: {cl:.4f}, C_d: {cd:.4f}"
        # )
        # df["pressure"] /= rho * U_inf**2

        # ## Spatial scaling
        # df["pressure"] *= spatial_scale
        # fx, fy, cl, cd, debug_info, chord, x_airfoil_center, z_airfoil_center = (
        #     compute_surface_forces(df, surface_x, surface_y, y_num, alpha)
        # )
        # print(
        #     f"\n[p = p * spatial_scale] \n Fx: {fx:.4f}, Fy: {fy:.4f}, C_l: {cl:.4f}, C_d: {cd:.4f}"
        # )
        # df["pressure"] /= spatial_scale

        # ## Spatial^2 scaling
        # df["pressure"] *= spatial_scale**2
        # fx, fy, cl, cd, debug_info, chord, x_airfoil_center, z_airfoil_center = (
        #     compute_surface_forces(df, surface_x, surface_y, y_num, alpha)
        # )
        # print(
        #     f"\n[p = p * spatial_scale^2] \n Fx: {fx:.4f}, Fy: {fy:.4f}, C_l: {cl:.4f}, C_d: {cd:.4f}"
        # )
        # df["pressure"] /= spatial_scale**2

        # ## Spatial^2 / Velocity^2
        # df["pressure"] /= (spatial_scale**2) / (U_inf**2)
        # fx, fy, cl, cd, debug_info, chord, x_airfoil_center, z_airfoil_center = (
        #     compute_surface_forces(df, surface_x, surface_y, y_num, alpha)
        # )
        # print(
        #     f"\n[p = p * spatial_scale^2 / U_inf^2] \n Fx: {fx:.4f}, Fy: {fy:.4f}, C_l: {cl:.4f}, C_d: {cd:.4f}"
        # )
        # df["pressure"] *= (spatial_scale**2) / (U_inf**2)

        # ## Lebesque I scaling
        # print(
        #     f'part 1: {np.sum(df["pressure"] / (0.5 * rho * (U_inf**2)))} part 2: {np.sum((df["V"] ** 2) / (U_inf**2))}'
        # )
        # df["pressure"] = df["pressure"] / (0.5 * rho * (U_inf**2)) + (
        #     (df["V"] ** 2) / (U_inf**2)
        # )
        # fx, fy, cl, cd, debug_info, chord, x_airfoil_center, z_airfoil_center = (
        #     compute_surface_forces(df, surface_x, surface_y, y_num, alpha)
        # )
        # print(
        #     f"\n[Cp,t = (p/(0.5*rho*Uinf^2)) + ((V/Uinf)^2)] \n C_l: {fy:.4f}, C_d: {fx:.4f}"
        # )

        # ## Lebesque I * (1/rho) scaling
        # df3["pressure"] = (1 / 1.2) * (
        #     df3["pressure"] / (0.5 * rho * (U_inf**2)) + ((df["V"] ** 2) / (U_inf**2))
        # )
        # fx, fy, cl, cd, debug_info, chord, x_airfoil_center, z_airfoil_center = (
        #     compute_surface_forces(df3, surface_x, surface_y, y_num, alpha)
        # )
        # # cl, cd = fy, fx
        # # fx, fy = cd * (0.5 * rho * (U_inf**2) * chord), cl * (
        # #     0.5 * rho * (U_inf**2) * chord
        # # )
        # print(f"x_airfoil_center: {x_airfoil_center:.4f}, {z_airfoil_center:.4f}")
        # print(
        #     f"\n[Cp,t = (1/rho_windtunnel)*(p/(0.5*rho*Uinf^2)) + ((V/Uinf)^2)] \n C_l: {fy:.4f}, C_d: {fx:.4f}"
        # )

        # ## Lebesque II scaling
        # df2["pressure"] = 2 * (df2["pressure"] + 0.5 * ((df2["V"] / U_inf) ** 2))
        # fx, fy, cl, cd, debug_info, chord, x_airfoil_center, z_airfoil_center = (
        #     compute_surface_forces(df2, surface_x, surface_y, y_num, alpha)
        # )
        # print(f"\n[Cp,t = 2*(p+0.5*(V/Uinf)^2)] \n C_l: {fy:.4f}, C_d: {fx:.4f}")


# Example usage
if __name__ == "__main__":
    main()
