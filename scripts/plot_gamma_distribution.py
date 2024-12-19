import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from utils import project_dir
from calculating_circulation import calculate_circulation
from utils import reading_optimal_bound_placement
import calculating_airfoil_centre
from defining_bound_volume import boundary_ellipse, boundary_rectangle
from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from plot_styling import set_plot_style, plot_on_ax
import force_from_noca
from plotting import (
    load_data,
    find_areas_needing_interpolation,
    interpolate_missing_data,
)


def get_VSM_gamma_distribution():
    # Defining discretisation
    n_panels = 54
    spanwise_panel_distribution = "split_provided"

    ### rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
    vsm_input_path = Path(project_dir) / "data" / "vsm_input"
    csv_file_path = (
        Path(vsm_input_path)
        / "TUDELFT_V3_LEI_KITE_rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.csv"
    )
    (
        LE_x_array,
        LE_y_array,
        LE_z_array,
        TE_x_array,
        TE_y_array,
        TE_z_array,
        d_tube_array,
        camber_array,
    ) = np.loadtxt(csv_file_path, delimiter=",", skiprows=1, unpack=True)
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs = []

    for i in range(len(LE_x_array)):
        LE = np.array([LE_x_array[i], LE_y_array[i], LE_z_array[i]])
        TE = np.array([TE_x_array[i], TE_y_array[i], TE_z_array[i]])
        rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.append(
            [LE, TE, ["lei_airfoil_breukels", [d_tube_array[i], camber_array[i]]]]
        )
    CAD_wing = Wing(n_panels, spanwise_panel_distribution)

    for i, CAD_rib_i in enumerate(
        rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
    ):
        # Scaling down to WindTunnelModel Size
        CAD_rib_i[0] = CAD_rib_i[0] / 6.5
        CAD_rib_i[1] = CAD_rib_i[1] / 6.5

        ### using breukels
        CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])

    wing_aero_CAD_19ribs = WingAerodynamics([CAD_wing])

    # Solvers

    VSM_solver = Solver(
        aerodynamic_model_type="VSM",
        is_with_artificial_damping=True,
    )

    # setting va
    def setting_va(
        wing_aero: object,
        Umag: float = 3.15,
        angle_of_attack: float = 10,
        side_slip: float = 0,
        yaw_rate: float = 0,
    ):
        aoa_rad = np.deg2rad(angle_of_attack)
        side_slip = np.deg2rad(side_slip)
        vel_app = (
            np.array(
                [
                    np.cos(aoa_rad) * np.cos(side_slip),
                    np.sin(side_slip),
                    np.sin(aoa_rad),
                ]
            )
            * Umag
        )
        wing_aero.va = (vel_app, yaw_rate)
        return wing_aero

    Umag = 15
    angle_of_attack = 6
    side_slip = 0
    yaw_rate = 0

    wing_aero_CAD_19ribs = setting_va(
        wing_aero_CAD_19ribs, Umag, angle_of_attack, side_slip, yaw_rate
    )
    ### plotting distributions
    results = VSM_solver.solve(wing_aero_CAD_19ribs)
    CAD_y_coordinates = [
        panels.aerodynamic_center[1] for panels in wing_aero_CAD_19ribs.panels
    ]
    VSM_gamma_distribution = results["gamma_distribution"]

    # Converting the

    return VSM_gamma_distribution, CAD_y_coordinates


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


def computing_circulation(df, plot_params, is_ellipse=True, n_points=10):

    alpha = plot_params["alpha"]
    y_num = plot_params["y_num"]
    U_inf = 15

    # Calculate airfoil center
    x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
        alpha, y_num, is_with_chord=True
    )
    d1centre = (x_airfoil, y_airfoil)
    drot = 0

    # Get initial bound placement
    dLx, dLy, iP = reading_optimal_bound_placement(
        alpha, y_num, is_with_N_datapoints=True
    )

    # Initialize list to store circulation values
    circulation_values = []
    F_x_list = []
    F_y_list = []
    C_l_list = []
    C_d_list = []

    # Create ranges for dLx and dLy: ±5% with 10 datapoints
    dLx_range = np.linspace(dLx * 0.95, dLx * 1.05, n_points)
    dLy_range = np.linspace(dLy * 0.95, dLy * 1.05, n_points)

    # Nested loop to go through 10x10 options
    for current_dLx in dLx_range:
        for current_dLy in dLy_range:
            # Create a copy of the dataframe to avoid modifying the original
            current_df = df.copy()

            # Interpolation check (assuming these functions and parameters exist)
            max_perc_interpolated_zones = 1.2
            perc_of_interpolated_points = 0
            n_datapoints = 104304

            # Check if not CFD data and interpolation is needed
            if not plot_params.get("is_CFD", False):
                # Find areas needing interpolation
                plot_params["interpolation_zones"] = find_areas_needing_interpolation(
                    current_df,
                    alpha,
                    y_num,
                    plot_params.get("rectangle_size"),
                    dLx=current_dLx,
                    dLy=current_dLy,
                )

                # Determine number of poitns that are interpolated
                n_points_nan = 0
                for interpolation_zone_i in plot_params["interpolation_zones"]:
                    x_min, x_max, y_min, y_max = interpolation_zone_i["bounds"]
                    subset = df[
                        (df["x"] >= x_min)
                        & (df["x"] <= x_max)
                        & (df["y"] >= y_min)
                        & (df["y"] <= y_max)
                    ]
                    dropped_by_nan = subset[subset["u"].isna()]
                    n_points_nan += len(dropped_by_nan)

                # Calculate percentage of interpolated points
                perc_of_interpolated_points = 100 * (n_points_nan / n_datapoints)

                # Interpolate if within acceptable range
                if perc_of_interpolated_points < max_perc_interpolated_zones:
                    for interpolation_zone in plot_params["interpolation_zones"]:
                        current_df, _ = interpolate_missing_data(
                            current_df,
                            interpolation_zone,
                        )
                else:
                    # Skip this iteration if too many points need interpolation
                    print(
                        f"---> SKIPPED perc_of_interpolated_points: {perc_of_interpolated_points:.2f}%, "
                        f"dLx: {current_dLx:.2f}, dLy: {current_dLy:.2f}"
                    )
                    continue

            # Determine boundary based on shape (ellipse or rectangle)
            if is_ellipse:
                d2curve = boundary_ellipse(
                    d1centre,
                    drot,
                    current_dLx,
                    current_dLy,
                    iP,
                )
            else:
                d2curve = boundary_rectangle(
                    d1centre,
                    drot,
                    current_dLx,
                    current_dLy,
                    iP,
                )

            # Calculate circulation for current parameters
            circulation = calculate_circulation(current_df, d2curve)
            circulation_values.append(circulation)

            # Computing NOCA
            mu = 1.7894e-5
            is_with_maximim_vorticity_location_correction = True
            if alpha == 6:
                rho = 1.20
            else:
                rho = 1.18

            F_x, F_y, C_l, C_d = force_from_noca.main(
                df,
                d2curve,
                mu=mu,
                is_with_maximim_vorticity_location_correction=is_with_maximim_vorticity_location_correction,
                rho=rho,
                U_inf=U_inf,
                c=chord,
            )
            F_x_list.append(F_x)
            F_y_list.append(F_y)
            C_l_list.append(C_l)
            C_d_list.append(C_d)

    # Return average circulation
    return (
        np.mean(circulation_values),
        np.mean(F_x),
        np.mean(F_y),
        np.mean(C_l),
        np.mean(C_d),
    )


def get_PIV_and_CFD_gamma_distribution(alpha: int = 6):

    cfd_ellipse_list = []
    cfd_rectangle_list = []
    piv_ellipse_list = []
    piv_rectangle_list = []
    for y_num in range(1, 8):
        print(f"\n-----> y_num: {y_num}")

        ## CFD
        print(f"CFD")
        plot_params_cfd = {
            "is_CFD": True,
            "y_num": y_num,
            "alpha": alpha,
            "d_alpha_rod": 7.25,
            "project_dir": project_dir,
            "plot_type": ".pdf",
            "title": None,
            "spanwise_CFD": False,
            "rectangle_size": 0.05,
        }
        df, x_mesh, y_mesh, plot_params_cfd = load_data(plot_params_cfd)
        (
            cfd_gamma_ellipse,
            cfd_fx_ellipse,
            cfd_fy_ellipse,
            cfd_cl_ellipse,
            cfd_cd_ellipse,
        ) = computing_circulation(df, plot_params_cfd, is_ellipse=True)
        (
            cfd_gamma_rectangle,
            cfd_fx_rectangle,
            cfd_fy_rectangle,
            cfd_cl_rectangle,
            cfd_cd_rectangle,
        ) = computing_circulation(df, plot_params_cfd, is_ellipse=False)
        cfd_ellipse_list.append(
            [
                cfd_gamma_ellipse,
                cfd_fx_ellipse,
                cfd_fy_ellipse,
                cfd_cl_ellipse,
                cfd_cd_ellipse,
            ]
        )
        cfd_rectangle_list.append(
            [
                cfd_gamma_rectangle,
                cfd_fx_rectangle,
                cfd_fy_rectangle,
                cfd_cl_rectangle,
                cfd_cd_rectangle,
            ]
        )

        ## PIV

        if y_num >= 7:
            piv_ellipse_list.append(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            piv_rectangle_list.append(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
        else:
            print(f"PIV")
            plot_params_piv = {
                "is_CFD": False,
                "y_num": y_num,
                "alpha": alpha,
                "d_alpha_rod": 7.25,
                "project_dir": project_dir,
                "plot_type": ".pdf",
                "title": None,
                "spanwise_CFD": False,
                "rectangle_size": 0.05,
            }
            df, x_mesh, y_mesh, plot_params_piv = load_data(plot_params_piv)
            (
                piv_gamma_ellipse,
                piv_fx_ellipse,
                piv_fy_ellipse,
                piv_cl_ellipse,
                piv_cd_ellipse,
            ) = computing_circulation(df, plot_params_piv, is_ellipse=True)
            (
                piv_gamma_rectangle,
                piv_fx_rectangle,
                piv_fy_rectangle,
                piv_cl_rectangle,
                piv_cd_rectangle,
            ) = computing_circulation(df, plot_params_piv, is_ellipse=False)

            piv_ellipse_list.append(
                [
                    piv_gamma_ellipse,
                    piv_fx_ellipse,
                    piv_fy_ellipse,
                    piv_cl_ellipse,
                    piv_cd_ellipse,
                ]
            )
            piv_rectangle_list.append(
                [
                    piv_gamma_rectangle,
                    piv_fx_rectangle,
                    piv_fy_rectangle,
                    piv_cl_rectangle,
                    piv_cd_rectangle,
                ]
            )

    return (
        cfd_ellipse_list,
        cfd_rectangle_list,
        piv_ellipse_list,
        piv_rectangle_list,
    )


def plot_gamma_distribution(save_path):

    csv_path = (
        Path(project_dir) / "processed_data" / "quantitative_chordwise_analysis.csv"
    )

    ## acquiring data
    cfd_ellipse, cfd_rectangle, piv_ellipse, piv_rectangle = (
        get_PIV_and_CFD_gamma_distribution()
    )
    df_y_locations = pd.read_csv(
        Path(project_dir) / "processed_data" / "gamma_distribution" / "y_locations.csv",
        index_col=False,
    )
    y_numbers = df_y_locations["PIV_mm"] / 1000

    df = pd.DataFrame(
        {
            "y_numbers": y_numbers,
            "cfd_gamma_ellipse": [x[0] for x in cfd_ellipse],
            "cfd_fx_ellipse": [x[1] for x in cfd_ellipse],
            "cfd_fy_ellipse": [x[2] for x in cfd_ellipse],
            "cfd_cl_ellipse": [x[3] for x in cfd_ellipse],
            "cfd_cd_ellipse": [x[4] for x in cfd_ellipse],
            "cfd_gamma_rectangle": [x[0] for x in cfd_rectangle],
            "cfd_fx_rectangle": [x[1] for x in cfd_rectangle],
            "cfd_fy_rectangle": [x[2] for x in cfd_rectangle],
            "cfd_cl_rectangle": [x[3] for x in cfd_rectangle],
            "cfd_cd_rectangle": [x[4] for x in cfd_rectangle],
            "piv_gamma_ellipse": [x[0] for x in piv_ellipse],
            "piv_fx_ellipse": [x[1] for x in piv_ellipse],
            "piv_fy_ellipse": [x[2] for x in piv_ellipse],
            "piv_cl_ellipse": [x[3] for x in piv_ellipse],
            "piv_cd_ellipse": [x[4] for x in piv_ellipse],
            "piv_gamma_rectangle": [x[0] for x in piv_rectangle],
            "piv_fx_rectangle": [x[1] for x in piv_rectangle],
            "piv_fy_rectangle": [x[2] for x in piv_rectangle],
            "piv_cl_rectangle": [x[3] for x in piv_rectangle],
            "piv_cd_rectangle": [x[4] for x in piv_rectangle],
        }
    )
    df.to_csv(csv_path, index=False)

    # loading data
    df = pd.read_csv(csv_path)
    cfd_gamma_ellipse = df["cfd_gamma_ellipse"]
    cfd_gamma_rectangle = df["cfd_gamma_rectangle"]
    piv_gamma_ellipse = df["piv_gamma_ellipse"]
    piv_gamma_rectangle = df["piv_gamma_rectangle"]

    ## plotting
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    VSM_gamma_distribution, CAD_y_coordinates = get_VSM_gamma_distribution()
    plot_on_ax(
        ax,
        x=CAD_y_coordinates,
        y=VSM_gamma_distribution,
        label="VSM",
        x_label=r"y [m]",
        y_label=r"$\Gamma$ [m$^2$/s]",  # r"$\Gamma [$m^2$/s]",
        # title="VSM gamma distribution",
    )
    plot_on_ax(
        ax,
        y_numbers,
        cfd_gamma_ellipse,
        label="CFD Ellipse",
        color="blue",
        marker="o",
        linestyle="-",
    )
    plot_on_ax(
        ax,
        y_numbers,
        cfd_gamma_rectangle,
        label="CFD Rectangle",
        color="blue",
        marker="s",
        linestyle="--",
    )
    plot_on_ax(
        ax,
        y_numbers[:6],
        piv_gamma_ellipse[:6],
        label="PIV Ellipse",
        color="red",
        marker="p",
        linestyle="-",
    )
    plot_on_ax(
        ax,
        y_numbers[:6],
        piv_gamma_rectangle[:6],
        label="PIV Rectangle",
        color="red",
        marker="*",
        linestyle="--",
    )

    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(r"y [m]")
    ax.set_ylabel(r"$\Gamma$ [m$^2$/s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

    # ## plotting CL, CD
    # # fig, ax = plt.subplots((1,2), figsize=(8, 5))
    # fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    # y_numbers =
    # plot_on_ax(
    #     axes[0],
    #     y_numbers,
    #     df["cfd_cl_ellipse"],
    #     label="CFD Ellipse",
    #     color="blue",
    #     marker="o",
    #     linestyle="-",
    # )
    # plot_on_ax(
    #     axes[0],
    #     y_numbers,
    #     df["cfd_cl_rectangle"],
    #     label="CFD Rectangle",
    #     color="blue",
    #     marker="s",
    #     linestyle="--",
    # )
    # plot_on_ax(
    #     axes[0],
    #     y_numbers[:6],
    #     df["piv_cl_ellipse"][:6],
    #     label="PIV Ellipse",
    #     color="red",
    #     marker="p",
    #     linestyle="-",
    # )
    # plot_on_ax(
    #     axes[0],
    #     y_numbers[:6],
    #     df["piv_cl_rectangle"][:6],
    #     label="PIV Rectangle",
    #     color="red",
    #     marker="*",
    #     linestyle="--",
    # )
    # plot_on_ax(
    #     axes[1],
    #     y_numbers,
    #     df["cfd_cd_ellipse"],
    #     label="CFD Ellipse",
    #     color="blue",
    #     marker="o",
    #     linestyle="-",
    # )
    # plot_on_ax(
    #     axes[1],
    #     y_numbers,
    #     df["cfd_cd_rectangle"],
    #     label="CFD Rectangle",
    #     color="blue",
    #     marker="s",
    #     linestyle="--",
    # )
    # plot_on_ax(
    #     axes[1],
    #     y_numbers[:6],
    #     df["piv_cd_ellipse"][:6],
    #     label="PIV Ellipse",
    #     color="red",
    #     marker="p",
    #     linestyle="-",
    # )
    # plot_on_ax(
    #     axes[1],
    #     y_numbers[:6],
    #     df["piv_cd_rectangle"][:6],
    #     label="PIV Rectangle",
    #     color="red",
    #     marker="*",
    #     linestyle="--",
    # )
    # axes[0].set_xlabel(r"y [m]")
    # axes[0].set_ylabel(r"$C_{\mathrm{l}}$")
    # axes[1].set_xlabel(r"y [m]")
    # axes[1].set_ylabel(r"$C_{\mathrm{d}}$")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(save_path.parent / "quantitative_cl_cd.pdf")


def main():
    save_path = Path(project_dir) / "results" / "paper_plots" / "gamma_distribution.pdf"
    plot_gamma_distribution(save_path)


if __name__ == "__main__":
    main()
