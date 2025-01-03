import pandas as pd
import numpy as np
import os
import scipy.interpolate as interpolate
from scipy.interpolate import griddata
from pathlib import Path
from utils import project_dir, interp2d_batch, csv_reader
import matplotlib.pyplot as plt


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
    x_surface,
    y_surface,
    density,
    p_ref,
    mu,
    is_plot=False,
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
    print(f"Fx_p: {Fx_p:.2f}, Fy_p: {Fy_p:.2f}, Fx_v: {Fx_v:.4f}, Fy_v: {Fy_v:.4f}")

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
            label="Airfoil Surface (used for force calculation)",
        )
        plt.pcolormesh(
            np.unique(df["x"]),
            np.unique(df["y"]),
            df.pivot(index="y", columns="x", values="pressure").values,
            shading="auto",
            cmap="jet",
        )

        # plt.scatter(surface_x_tau_w, surface_y_tau_w, c="b", label="Airfoil Surface")
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
    rho = 1  # 1.2 #TODO: change back to 1.2
    U_inf = 15
    Re = 1e6
    p_ref = 0
    is_with_plot = False
    is_with_NOCA = False

    alpha = 6
    for y_num in [1, 2, 3, 4, 5]:
        print(f"\nProcessing alpha: {alpha}, y_num: {y_num}")
        df = csv_reader(is_CFD=True, alpha=alpha, y_num=y_num, alpha_d_rod=7)
        # Reading in the airfoil centers
        x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
            alpha, y_num, is_with_chord=True
        )
        if y_num == 1:
            ref_chord = chord
        # mu only influences the deviatoric stress tensor
        mu = (rho * U_inf * 1) / Re

        print(f"rho: {rho}, U_inf: {U_inf}, Re: {Re} ---> mu: {mu}, ")
        ## Use actual airfoils as surface points
        surface_x, surface_y = plotting.plot_airfoil(
            None, {"y_num": y_num, "alpha": alpha}, is_return_surface_points=True
        )
        # Flip point order
        surface_x = surface_x[::-1]
        surface_y = surface_y[::-1]
        if is_with_NOCA:
            ## NOCA
            Fx, Fy, C_l, C_d = running_NOCA(df, alpha, y_num)
            print(
                f"NOCA          --- Fx: {Fx:.2f}, Fy: {Fy:.2f}, C_l: {C_l:.2f}, C_d: {C_d:.2f}"
            )

        ### Compute surface forces
        Fx_total, Fy_total = compute_surface_forces(
            df,
            x_surface=surface_x,
            y_surface=surface_y,
            density=rho,
            p_ref=p_ref,
            mu=mu,
            is_plot=is_with_plot,
        )
        # q_infc influences the pressure force, through non-dimensionalisation
        q_infc = 0.5 * rho * U_inf**2 * ref_chord
        print(f"rho: {rho}, U_inf: {U_inf}, chord: {ref_chord} ---> q_infc: {q_infc}")
        print(
            f"P-integration --- Fx: {Fx_total:.2f}, Fy: {Fy_total:.2f}, C_l: {Fy_total/q_infc:.2f}, C_d: {Fx_total/q_infc:.2f}"
        )
    alpha = 16
    for y_num in [1]:
        print(f"\nProcessing alpha: {alpha}, y_num: {y_num}")
        df = csv_reader(is_CFD=True, alpha=alpha, y_num=y_num, alpha_d_rod=7)
        # Reading in the airfoil centers
        x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
            alpha, y_num, is_with_chord=True
        )
        mu = (rho * U_inf * 1) / Re
        q_infc = 0.5 * rho * U_inf**2 * chord
        print(f"rho: {rho}, U_inf: {U_inf}, mu: {mu}, Re: {Re}, q_infc: {q_infc}")
        ## Use actual airfoils as surface points
        surface_x, surface_y = plotting.plot_airfoil(
            None, {"y_num": y_num, "alpha": alpha}, is_return_surface_points=True
        )
        # Flip point order
        surface_x = surface_x[::-1]
        surface_y = surface_y[::-1]
        if is_with_NOCA:
            ## NOCA
            Fx, Fy, C_l, C_d = running_NOCA(df, alpha, y_num)
            print(
                f"NOCA          --- Fx: {Fx:.2f}, Fy: {Fy:.2f}, C_l: {C_l:.2f}, C_d: {C_d:.2f}"
            )

        ### Compute surface forces
        Fx_total, Fy_total = compute_surface_forces(
            df,
            x_surface=surface_x,
            y_surface=surface_y,
            density=rho,
            p_ref=p_ref,
            mu=mu,
            is_plot=False,
        )
        print(
            f"P-integration --- Fx: {Fx_total:.2f}, Fy: {Fy_total:.2f}, C_l: {Fy_total/q_infc:.2f}, C_d: {Fx_total/q_infc:.2f}"
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
