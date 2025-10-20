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
from VSM.core.WingGeometry import Wing
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from plot_styling import set_plot_style, plot_on_ax
import force_from_noca
from plotting import (
    load_data,
    find_areas_needing_interpolation,
    interpolate_missing_data,
)


def computing_gamma_and_noca_fx_fy(
    df,
    plot_params,
    is_ellipse=True,
    n_points=10,
    d_perc=5,
    U_inf=15,
    mu=1.7894e-5,
    rho=1.2,
    drot=0,
    ref_chord=0.39834712,
    max_perc_interpolated_zones=1.2,
    n_datapoints=104304,
):

    alpha = plot_params["alpha"]
    y_num = plot_params["y_num"]

    # Calculate airfoil center
    x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
        alpha, y_num, is_with_chord=True
    )
    d1centre = (x_airfoil, y_airfoil)

    # Get initial bound placement
    dLx, dLy, iP = reading_optimal_bound_placement(
        alpha, y_num, is_with_N_datapoints=True
    )

    # Initialize list to store circulation values
    gamma_list = []
    F_x_list = []
    F_y_list = []

    # Create ranges for dLx and dLy: ±5% with 10 datapoints
    dmin = 1 - (d_perc / 100)
    dmax = 1 + (d_perc / 100)
    dLx_range = np.linspace(dLx * dmin, dLx * dmax, n_points)
    dLy_range = np.linspace(dLy * dmin, dLy * dmax, n_points)

    # Nested loop to go through 10x10 options
    for current_dLx in dLx_range:
        for current_dLy in dLy_range:
            # Create a copy of the dataframe to avoid modifying the original
            current_df = df.copy()

            # Interpolation check (assuming these functions and parameters exist)
            perc_of_interpolated_points = 0

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

                ## setting the smoothing on
                is_with_smoothing = True

            else:  # if CFD data
                is_with_smoothing = False

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
            circulation = calculate_circulation(current_df, d2curve, is_with_smoothing)
            gamma_list.append(circulation)

            # Computing NOCA
            # if alpha == 6:
            #     rho = 1.20
            # else:
            #     rho = 1.18
            F_x, F_y, C_l, C_d = force_from_noca.main(
                df,
                d2curve,
                mu=mu,
                is_with_maximim_vorticity_location_correction=True,
                rho=rho,
                U_inf=U_inf,
                ref_chord=ref_chord,
            )
            F_x_list.append(F_x)
            F_y_list.append(F_y)
            print(
                f"C_kutta:{(circulation*rho*U_inf)/((0.5 * rho * U_inf**2 * ref_chord)):.2f}, NOCA: C_l:{C_l:.2f}, C_d:{C_d:.2f}, % interpolated: {perc_of_interpolated_points:.2f}%"
            )

    # Return average circulation
    return (
        np.mean(gamma_list),
        np.mean(F_x),
        np.mean(F_y),
        np.std(gamma_list),
        np.std(F_x),
        np.std(F_y),
    )


def get_PIV_and_CFD_gamma_distribution_for_single_alpha(
    alpha: int = 6, y_num_list: list = [1]
):

    Re_cfd = 1e6
    Re_piv = 4.2e5
    rho = 1.20
    ref_chord = 0.39834712
    U_inf = 15
    mu_cfd = (rho * ref_chord * U_inf) / Re_cfd
    mu_piv = (rho * ref_chord * U_inf) / Re_piv
    print(f"mu_cfd: {mu_cfd}, mu_piv: {mu_piv}")
    # mu_piv = 1.7894e-5

    cfd_ellipse_list = []
    cfd_rectangle_list = []
    piv_ellipse_list = []
    piv_rectangle_list = []
    for y_num in y_num_list:
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
            cfd_gamma_ellipse_std,
            cfd_fx_ellipse_std,
            cfd_fy_ellipse_std,
        ) = computing_gamma_and_noca_fx_fy(
            df, plot_params_cfd, is_ellipse=True, mu=mu_cfd
        )
        (
            cfd_gamma_rectangle,
            cfd_fx_rectangle,
            cfd_fy_rectangle,
            cfd_gamma_rectangle_std,
            cfd_fx_rectangle_std,
            cfd_fy_rectangle_std,
        ) = computing_gamma_and_noca_fx_fy(
            df, plot_params_cfd, is_ellipse=False, mu=mu_cfd
        )
        cfd_ellipse_list.append(
            [
                cfd_gamma_ellipse,
                cfd_fx_ellipse,
                cfd_fy_ellipse,
                cfd_gamma_ellipse_std,
                cfd_fx_ellipse_std,
                cfd_fy_ellipse_std,
            ]
        )
        cfd_rectangle_list.append(
            [
                cfd_gamma_rectangle,
                cfd_fx_rectangle,
                cfd_fy_rectangle,
                cfd_gamma_rectangle_std,
                cfd_fx_rectangle_std,
                cfd_fy_rectangle_std,
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
                    0,
                    0,
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
                    0,
                    0,
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
            csv_path_std = plot_params_piv["csv_file_path_std"]
            df_std = pd.read_csv(csv_path_std)
            ## Normal
            (
                piv_gamma_ellipse,
                piv_fx_ellipse,
                piv_fy_ellipse,
                piv_gamma_ellipse_std,
                piv_fx_ellipse_std,
                piv_fy_ellipse_std,
            ) = computing_gamma_and_noca_fx_fy(
                df,
                plot_params_piv,
                is_ellipse=True,
                mu=mu_piv,
            )
            (
                piv_gamma_rectangle,
                piv_fx_rectangle,
                piv_fy_rectangle,
                piv_gamma_rectangle_std,
                piv_fx_rectangle_std,
                piv_fy_rectangle_std,
            ) = computing_gamma_and_noca_fx_fy(
                df,
                plot_params_piv,
                is_ellipse=False,
                mu=mu_piv,
            )
            ## mean - 1.96*std
            df_lower_bound = df.copy()
            df_lower_bound["u"] = df["u"] - 1.96 * df_std["u"]
            df_lower_bound["v"] = df["v"] - 1.96 * df_std["v"]
            df_lower_bound["w"] = df["w"] - 1.96 * df_std["w"]
            df_lower_bound["V"] = df["V"] - 1.96 * df_std["V"]
            df_lower_bound["dudx"] = df["dudx"] - 1.96 * df_std["dudx"]
            df_lower_bound["dudy"] = df["dudy"] - 1.96 * df_std["dudy"]
            df_lower_bound["dvdx"] = df["dvdx"] - 1.96 * df_std["dvdx"]
            df_lower_bound["dvdy"] = df["dvdy"] - 1.96 * df_std["dvdy"]
            df_lower_bound["dwdx"] = df["dwdx"] - 1.96 * df_std["dwdx"]
            df_lower_bound["dwdy"] = df["dwdy"] - 1.96 * df_std["dwdy"]
            df_lower_bound["vort_z"] = df["vort_z"] - 1.96 * df_std["vort_z"]

            (
                lower_piv_gamma_ellipse,
                lower_piv_fx_ellipse,
                lower_piv_fy_ellipse,
                lower_piv_gamma_ellipse_std,
                lower_piv_fx_ellipse_std,
                lower_piv_fy_ellipse_std,
            ) = computing_gamma_and_noca_fx_fy(
                df_lower_bound,
                plot_params_piv,
                is_ellipse=True,
                mu=mu_piv,
            )
            (
                lower_piv_gamma_rectangle,
                lower_piv_fx_rectangle,
                lower_piv_fy_rectangle,
                lower_piv_gamma_rectangle_std,
                lower_piv_fx_rectangle_std,
                lower_piv_fx_rectangle_std,
            ) = computing_gamma_and_noca_fx_fy(
                df_lower_bound,
                plot_params_piv,
                is_ellipse=False,
                mu=mu_piv,
            )

            piv_ellipse_list.append(
                [
                    piv_gamma_ellipse,
                    piv_fx_ellipse,
                    piv_fy_ellipse,
                    piv_gamma_ellipse_std,
                    piv_fx_ellipse_std,
                    piv_fy_ellipse_std,
                    lower_piv_gamma_ellipse,
                    lower_piv_fx_ellipse,
                    lower_piv_fy_ellipse,
                    lower_piv_gamma_ellipse_std,
                    lower_piv_fx_ellipse_std,
                    lower_piv_fy_ellipse_std,
                ]
            )
            piv_rectangle_list.append(
                [
                    piv_gamma_rectangle,
                    piv_fx_rectangle,
                    piv_fy_rectangle,
                    piv_gamma_rectangle_std,
                    piv_fx_rectangle_std,
                    piv_fy_rectangle_std,
                    lower_piv_gamma_rectangle,
                    lower_piv_fx_rectangle,
                    lower_piv_fy_rectangle,
                    lower_piv_gamma_rectangle_std,
                    lower_piv_fx_rectangle_std,
                    lower_piv_fx_rectangle_std,
                ]
            )

    return (
        cfd_ellipse_list,
        cfd_rectangle_list,
        piv_ellipse_list,
        piv_rectangle_list,
    )


def save_results_single_alpha(alpha, y_num_list):
    ## acquiring data
    cfd_ellipse, cfd_rectangle, piv_ellipse, piv_rectangle = (
        get_PIV_and_CFD_gamma_distribution_for_single_alpha(alpha, y_num_list)
    )
    df = pd.DataFrame(
        {
            "y_num": y_num_list,
            "ellipse_cfd_gamma": [x[0] for x in cfd_ellipse],
            "ellipse_cfd_fx": [x[1] for x in cfd_ellipse],
            "ellipse_cfd_fy": [x[2] for x in cfd_ellipse],
            "ellipse_cfd_gamma_std": [x[3] for x in cfd_ellipse],
            "ellipse_cfd_fx_std": [x[4] for x in cfd_ellipse],
            "ellipse_cfd_fy_std": [x[5] for x in cfd_ellipse],
            "rectangle_cfd_gamma": [x[0] for x in cfd_rectangle],
            "rectangle_cfd_fx": [x[1] for x in cfd_rectangle],
            "rectangle_cfd_fy": [x[2] for x in cfd_rectangle],
            "rectangle_cfd_gamma_std": [x[3] for x in cfd_rectangle],
            "rectangle_cfd_fx_std": [x[4] for x in cfd_rectangle],
            "rectangle_cfd_fy_std": [x[5] for x in cfd_rectangle],
            "ellipse_piv_gamma": [x[0] for x in piv_ellipse],
            "ellipse_piv_fx": [x[1] for x in piv_ellipse],
            "ellipse_piv_fy": [x[2] for x in piv_ellipse],
            "ellipse_piv_gamma_std": [x[3] for x in piv_ellipse],
            "ellipse_piv_fx_std": [x[4] for x in piv_ellipse],
            "ellipse_piv_fy_std": [x[5] for x in piv_ellipse],
            "ellipse_piv_gamma_lower_bound": [x[6] for x in piv_ellipse],
            "ellipse_piv_fx_lower_bound": [x[7] for x in piv_ellipse],
            "ellipse_piv_fy_lower_bound": [x[8] for x in piv_ellipse],
            "ellipse_piv_gamma_lower_bound_std": [x[9] for x in piv_ellipse],
            "ellipse_piv_fx_lower_bound_std": [x[10] for x in piv_ellipse],
            "ellipse_piv_fy_lower_bound_std": [x[11] for x in piv_ellipse],
            "rectangle_piv_gamma": [x[0] for x in piv_rectangle],
            "rectangle_piv_fx": [x[1] for x in piv_rectangle],
            "rectangle_piv_fy": [x[2] for x in piv_rectangle],
            "rectangle_piv_gamma_std": [x[3] for x in piv_rectangle],
            "rectangle_piv_fx_std": [x[4] for x in piv_rectangle],
            "rectangle_piv_fy_std": [x[5] for x in piv_rectangle],
            "rectangle_piv_gamma_lower_bound": [x[6] for x in piv_rectangle],
            "rectangle_piv_fx_lower_bound": [x[7] for x in piv_rectangle],
            "rectangle_piv_fy_lower_bound": [x[8] for x in piv_rectangle],
            "rectangle_piv_gamma_lower_bound_std": [x[9] for x in piv_rectangle],
            "rectangle_piv_fx_lower_bound_std": [x[10] for x in piv_rectangle],
            "rectangle_piv_fy_lower_bound_std": [x[11] for x in piv_rectangle],
        }
    )
    csv_path = (
        Path(project_dir)
        / "processed_data"
        / f"quantitative_chordwise_analysis_alpha_{alpha}_with_std.csv"
    )
    df.to_csv(csv_path, index=False)
    return df


def saving_results():
    alpha = 6
    y_num_list = [1, 2, 3, 4, 5, 6, 7]
    save_results_single_alpha(alpha, y_num_list)

    alpha = 16
    y_num_list = [1]
    save_results_single_alpha(alpha, y_num_list)


def main():

    # takes 30min or so
    saving_results()

    ## reading results
    alpha = 6
    df_alpha_6 = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / f"quantitative_chordwise_analysis_alpha_{alpha}.csv"
    )
    alpha = 16
    df_alpha_16 = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / f"quantitative_chordwise_analysis_alpha_{alpha}.csv"
    )

    ## printing results
    rho = 1.20
    U_inf = 15
    ref_chord = 0.39834712
    q_infc = 0.5 * rho * (U_inf**2) * ref_chord
    print(f"rho: {rho}, U_inf: {U_inf}, ref_chord: {ref_chord} --> q_infc: {q_infc}")
    print(f"\nalpha: 6")
    for y_num in [1, 2, 3, 4, 5]:
        print(f"\n-----> y_num: {y_num}")
        for name_prefix in [
            "ellipse_cfd",
            "ellipse_piv",
            "rectangle_cfd",
            "rectangle_piv",
        ]:
            fx = df_alpha_6[df_alpha_6["y_num"] == y_num][f"{name_prefix}_fx"].values[0]
            fy = df_alpha_6[df_alpha_6["y_num"] == y_num][f"{name_prefix}_fy"].values[0]
            gamma = df_alpha_6[df_alpha_6["y_num"] == y_num][
                f"{name_prefix}_gamma"
            ].values[0]
            print(
                f"{name_prefix} -- cl: {fy/q_infc:.2f}, cd: {fx/q_infc:.2f}, c_kutta: {gamma*rho*U_inf/q_infc:.2f}"
            )

    print(f"\nalpha: 16")
    for y_num in [1]:
        print(f"\n-----> y_num: {y_num}")
        for name_prefix in [
            "ellipse_cfd",
            "ellipse_piv",
            "rectangle_cfd",
            "rectangle_piv",
        ]:
            fx = df_alpha_16[df_alpha_16["y_num"] == y_num][f"{name_prefix}_fx"].values[
                0
            ]
            fy = df_alpha_16[df_alpha_16["y_num"] == y_num][f"{name_prefix}_fy"].values[
                0
            ]
            gamma = df_alpha_16[df_alpha_16["y_num"] == y_num][
                f"{name_prefix}_gamma"
            ].values[0]
            print(
                f"{name_prefix} -- cl: {fy/q_infc:.2f}, cd: {fx/q_infc:.2f}, c_kutta: {gamma*rho*U_inf/q_infc:.2f}"
            )


if __name__ == "__main__":
    # main()

    # alpha = 6
    # y_num_list = [1, 2, 3, 4, 5, 6, 7]
    # save_results_single_alpha(alpha, y_num_list)

    Re_cfd = 1e6
    Re_piv = 3.8e5
    rho = 1.20
    ref_chord = 0.39834712
    U_inf = 15
    mu_cfd = (rho * ref_chord * U_inf) / Re_cfd
    mu_piv = (rho * ref_chord * U_inf) / Re_piv
    print(f"mu_cfd: {mu_cfd}, mu_piv: {mu_piv}")
    breakpoint()
