from kite_piv_analysis.plotting import *
from kite_piv_analysis.plot_styling import set_plot_style
from kite_piv_analysis.masking import apply_snr_masking, apply_w_masking


def normal_masked_interpolated(plot_params: dict) -> None:
    """Create a 4x3 comparison of CFD and PIV data, with PIV masked/unmasked."""
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12, 8),
        gridspec_kw={
            "hspace": 0.00,  # A bit more vertical space for labels
            "wspace": 0.07,
        },
    )  # Minimal horizontal space since colorbars are outside)
    # fig.suptitle(
    #     rf'Y{plot_params["y_num"]} | α = {plot_params["alpha"]}° | V_inf = {plot_params["u_inf"]}m/s'
    # )

    is_with_circulation_analysis = plot_params["is_with_circulation_analysis"]
    is_with_bound = plot_params["is_with_bound"]
    is_with_interpolation = plot_params["is_with_interpolation"]
    data_columns = ["u", "v", "w"]
    data_labels = [r"$u_{{x}}$", r"$u_{{y}}$", r"$u_{{z}}$"]

    for i, (column, label) in enumerate(zip(data_columns, data_labels)):

        if i == 2:
            is_with_xlabel = True
        else:
            is_with_xlabel = False

        # Update color data label
        plot_params["color_data_col_name"] = column

        # ### CFD
        # plot_params["is_with_interpolation"] = False
        # if is_with_bound:
        #     plot_params["is_with_bound"] = True
        # if is_with_circulation_analysis:
        #     plot_params["is_with_circulation_analysis"] = True
        # df_cfd, x_mesh_cfd, y_mesh_cfd, plot_params = load_data(
        #     plot_params | {"is_CFD": True}
        # )
        # plot_params = plotting_on_ax(
        #     fig, axes[i, 3], df_cfd, x_mesh_cfd, y_mesh_cfd, plot_params
        # )
        # axes[i, 3].set_title(f"CFD")
        # if plot_params["is_with_cbar"]:
        #     add_colorbar(fig, axes[i, 3], plot_params)

        ### PIV raw
        plot_params["is_with_interpolation"] = False
        plot_params["is_with_bound"] = False
        plot_params["is_with_circulation_analysis"] = False
        plot_params["is_with_mask"] = False
        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
        plot_params = plotting_on_ax(
            fig,
            axes[i, 0],
            df_piv,
            x_mesh_piv,
            y_mesh_piv,
            plot_params,
            is_with_xlabel=is_with_xlabel,
            is_with_ylabel=True,
        )
        if plot_params["is_with_cbar"]:
            add_vertical_colorbar_for_row(fig, axes[i, :], plot_params)

        ### PIV Mask
        plot_params["is_with_interpolation"] = False
        plot_params["is_with_bound"] = False
        plot_params["is_with_circulation_analysis"] = False
        plot_params["is_with_mask"] = True
        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
        if plot_params.get("is_with_mask", False):
            df_piv = apply_w_masking(df_piv, plot_params)
            df_piv = apply_snr_masking(df_piv, plot_params)
        plot_params["is_with_mask"] = False
        plot_params = plotting_on_ax(
            fig,
            axes[i, 1],
            df_piv,
            x_mesh_piv,
            y_mesh_piv,
            plot_params,
            is_with_xlabel=is_with_xlabel,
            is_with_ylabel=False,
        )
        ### PIV Mask Reinterpolated
        if is_with_bound:
            plot_params["is_with_bound"] = True
        if is_with_circulation_analysis:
            plot_params["is_with_circulation_analysis"] = True
        if is_with_interpolation:
            plot_params["is_with_interpolation"] = True
            print(f'is with interpolation: {plot_params["is_with_interpolation"]}')

        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
        if plot_params.get("is_with_mask", False):
            df_piv = apply_w_masking(df_piv, plot_params)
            df_piv = apply_snr_masking(df_piv, plot_params)
        plot_params["is_with_mask"] = False
        plot_params = plotting_on_ax(
            fig,
            axes[i, 2],
            df_piv,
            x_mesh_piv,
            y_mesh_piv,
            plot_params,
            is_with_xlabel=is_with_xlabel,
            is_with_ylabel=False,
        )

        if i == 0:
            axes[i, 0].set_title(f"PIV Raw")
            axes[i, 1].set_title(
                # f'PIV Masked for {plot_params["column_to_mask"]} in bounds {plot_params["mask_lower_bound"]} to {plot_params["mask_upper_bound"]}'
                f"PIV Masked"
            )
            axes[i, 2].set_title(r"PIV Masked \& Interpolated")
            # if plot_params["is_with_cbar"]:
        #     add_colorbar(fig, axes[i, 2], plot_params)

        ### Reset things
        plot_params["min_cbar_value"] = None
        plot_params["max_cbar_value"] = None

        ## Axis settings
        axes[i, 0].grid(False)
        axes[i, 1].grid(False)
        axes[i, 2].grid(False)
        axes[i, 0].tick_params(labelbottom=False, labelleft=True)
        axes[i, 1].tick_params(labelbottom=False, labelleft=False)
        axes[i, 2].tick_params(labelbottom=False, labelleft=False)
        if i == 2:
            axes[i, 0].tick_params(labelbottom=True)
            axes[i, 1].tick_params(labelbottom=True)
            axes[i, 2].tick_params(labelbottom=True)

    # Adjust layout and save the plot
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle space
    # plt.tight_layout()
    save_plot(fig, plot_params)


def normal_masked_interpolated_3by2(plot_params: dict) -> None:
    """Create a 3x3 comparison: raw, |u_y|-only masked, and |w|+SNR masked."""
    is_pcolormesh = None
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12, 8),
        gridspec_kw={"hspace": -0.1, "wspace": 0.07},
    )  # Minimal spacing for compact layout

    data_columns = ["u", "v", "w"]
    data_labels = [r"$u_{{x}}$", r"$u_{{y}}$", r"$u_{{z}}$"]

    plot_params.update(
        {
            "is_with_interpolation": False,
            "is_with_bound": False,
            "is_with_circulation_analysis": False,
        }
    )

    for i, (column, label) in enumerate(zip(data_columns, data_labels)):
        # Determine x-axis label settings
        is_with_xlabel = i == 2  # Only for the last row

        # Update color data label
        plot_params["color_data_col_name"] = column

        ### PIV raw (left column)
        plot_params["is_with_mask"] = False
        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
        plot_params = plotting_on_ax(
            fig,
            axes[i, 0],
            df_piv,
            x_mesh_piv,
            y_mesh_piv,
            plot_params,
            is_with_xlabel=is_with_xlabel,
            is_with_ylabel=False,
            is_pcolormesh=is_pcolormesh,
        )

        ### PIV |w|-only mask (middle column)
        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
        df_piv = apply_w_masking(df_piv, plot_params)
        plot_params = plotting_on_ax(
            fig,
            axes[i, 1],
            df_piv,
            x_mesh_piv,
            y_mesh_piv,
            plot_params,
            is_with_xlabel=is_with_xlabel,
            is_with_ylabel=False,
            is_pcolormesh=is_pcolormesh,
        )

        ### PIV |w| + SNR mask (right column)
        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
        df_piv = apply_w_masking(df_piv, plot_params)
        df_piv = apply_snr_masking(df_piv, plot_params)
        plot_params = plotting_on_ax(
            fig,
            axes[i, 2],
            df_piv,
            x_mesh_piv,
            y_mesh_piv,
            plot_params,
            is_with_xlabel=is_with_xlabel,
            is_with_ylabel=False,
            is_label_left=False,
            is_pcolormesh=is_pcolormesh,
        )

        # setting titles
        if i == 0:
            axes[i, 0].set_title("Raw")
            axes[i, 1].set_title(r"Only $|u_y|$ masked")
            axes[i, 2].set_title(r"$|u_y|$ and ICV masked")

        # adding cbar
        add_vertical_colorbar_for_row(
            fig, axes[i, :], plot_params, label=label, labelpad=21
        )

        ### Reset things
        plot_params["min_cbar_value"] = None
        plot_params["max_cbar_value"] = None

    plt.tight_layout(pad=0.02)
    # Save the plot
    # save_path = (
    #     Path(project_dir)
    #     / "results"
    #     / "paper_plots_21_10_2025"
    #     / f"fig10_PIV_normal_masked_Y{plot_params['y_num']}.pdf"
    # )
    save_path = Path(project_dir) / "results" / "paper_plots_24_03_2026" / f"figC1.pdf"
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def main():
    set_plot_style()

    plot_params: PlotParams = {
        # Basic configuration
        "is_CFD": False,
        "spanwise_CFD": False,
        "y_num": 1,
        "alpha": 6,
        "project_dir": project_dir,
        "plot_type": ".pdf",
        "title": None,
        "is_CFD_PIV_comparison": False,
        "color_data_col_name": "V",
        "is_CFD_PIV_comparison_multicomponent_masked": False,
        "run_for_all_planes": False,
        "normal_masked_interpolated": True,
        # Plot_settings
        "xlim": (-0.2, 0.8),
        "ylim": (-0.2, 0.4),
        # Color and contour settings
        "is_with_cbar": True,
        "cbar_value_factor_of_std": 2.0,
        "min_cbar_value": None,
        "max_cbar_value": None,
        "subsample_color": 1,
        "countour_levels": 100,
        "cmap": "coolwarm",
        # Quiver settings
        "is_with_quiver": False,
        "subsample_quiver": 5,
        "u_inf": 15.0,
        # PIV specific settings
        "d_alpha_rod": 7.25,
        # Overlay settings
        "is_with_overlay": False,
        "overlay_alpha": 0.4,
        # Airfoil settings
        "is_with_airfoil": True,
        "airfoil_transparency": 1.0,
        # Raw image settings
        "subsample_factor_raw_images": 1,
        "intensity_lower_bound": 10000,
        # Boundary settings
        "is_with_bound": True,
        "drot": 0.0,
        "iP": 65,
        # insert
        # "d1centre": np.array([0.27, 0.13]),
        # "drot": 0,
        # "dLx": 0.8,
        # "dLy": 0.4,
        # "iP": 35,
        ##
        "ellipse_color": "red",
        "rectangle_color": "green",
        "bound_linewidth": 2.0,
        "bound_alpha": 1.0,
        # Circulation analysis
        "is_with_circulation_analysis": False,
        "rho": 1.20,
        "mu": 1.7894e-5,
        "is_with_maximim_vorticity_location_correction": True,
        "chord": 0.396,
        # Mask settings
        "is_with_mask": False,
        "is_with_w_mask": True,
        "w_mask_lower_bound": -3,
        "w_mask_upper_bound": 3,
        "is_with_snr_mask": [True, True],
        "snr_use_proxy": [True, True],
        "snr_column_to_mask": ["snr", "snr"],
        "snr_mask_lower_bound": [2.0, 2.0],
        "snr_mask_upper_bound": [np.inf, np.inf],
        "proxy_snr_components": [["u", "v"], ["u", "v"]],
        "proxy_snr_reduction": ["median", "mean"],
        "proxy_snr_min_std": [1e-6, 1e-6],
        "snr_mask_nan_as_invalid": [True, True],
        "snr_mask_strict": [False, False],
        "snr_apply_only_below_rear_airfoil_line": [False, True],
        "snr_rear_airfoil_fraction": [0.5, 0.5],
        "normal_masked_interpolated": True,
        ## Interpolation settings
        "is_with_interpolation": True,
        "interpolation_method": "nearest",
        "rectangle_size": 0.05,
    }
    if plot_params["is_CFD_PIV_comparison"]:
        type_label = "CFD_PIV"
    else:
        type_label = "CFD" if plot_params["is_CFD"] else "PIV"

    # normal_masked_interpolated(plot_params)
    normal_masked_interpolated_3by2(plot_params)
    print(
        f'{type_label} plot with color = {plot_params["color_data_col_name"]} | Y{plot_params["y_num"]} | α = {plot_params["alpha"]}°'
    )


if __name__ == "__main__":
    main()
