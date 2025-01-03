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


def plot_gamma_distribution(save_path):
    ## loading data
    df_y_locations = pd.read_csv(
        Path(project_dir) / "processed_data" / "gamma_distribution" / "y_locations.csv",
        index_col=False,
    )
    y_numbers = df_y_locations["PIV_mm"] / 1000
    csv_path = (
        Path(project_dir)
        / "processed_data"
        / "quantitative_chordwise_analysis_alpha_6.csv"
    )
    df = pd.read_csv(csv_path)
    cfd_gamma_ellipse = df["ellipse_cfd_gamma"]
    cfd_gamma_rectangle = df["rectangle_cfd_gamma"]
    piv_gamma_ellipse = df["ellipse_piv_gamma"]
    piv_gamma_rectangle = df["rectangle_piv_gamma"]

    print(f"running VSM")
    VSM_gamma_distribution, CAD_y_coordinates = get_VSM_gamma_distribution()

    ## plotting
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

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
