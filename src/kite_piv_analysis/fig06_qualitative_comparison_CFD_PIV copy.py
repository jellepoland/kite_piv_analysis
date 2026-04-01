from kite_piv_analysis.plotting import *
from kite_piv_analysis.plot_styling import set_plot_style
from matplotlib.gridspec import GridSpec
from kite_piv_analysis.masking import apply_snr_masking, apply_w_masking


def plotting_qualitative_CFD_PIV(alphas, y_nums, file_name, plot_params: dict) -> None:

    set_plot_style()

    # Set up alpha and y_num values
    # alphas = [6, 6, 6, 16]
    # y_nums = [1, 3, 4, 1]
    is_with_xlabel = False

    n_rows = len(alphas)
    n_cols = 2

    """Create a comparison of CFD and PIV data for different Y positions and alphas."""
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10.5, 13.6),  # int(20 * (n_rows / 6))),
        gridspec_kw={
            "hspace": -0.01,  # A bit more vertical space for labels
            "wspace": 0.07,
            # "height_ratios": [1, 1, 1, 1, 1, 2],
            # "hspace": 0.005,
        },
    )

    for row, (alpha, y_num) in enumerate(zip(alphas, y_nums)):
        # Update plot_params for current alpha and y_num
        current_params = plot_params.copy()
        current_params["alpha"] = alpha
        current_params["y_num"] = y_num

        alpha += 1  # corrections
        print(f"Plotting α = {alpha}, Y = {y_num}")

        # Add main figure title for first and last rows
        if row == 0 or row == (n_rows - 1):
            axes[row, 0].set_title(
                rf"CFD for $\alpha$ = {alpha}" + r"$^{\circ}$",
                fontsize=14,
                fontweight="bold",
                pad=5,
            )
            axes[row, 1].set_title(
                rf"PIV for $\alpha$ = {alpha}" + r"$^{\circ}$",
                fontsize=14,
                fontweight="bold",
                pad=5,
            )
        if row == (n_rows - 1):
            is_with_xlabel = True

        # Load and plot CFD data for this row
        current_params_cfd = current_params | {"is_CFD": True}
        df_cfd, x_mesh_cfd, y_mesh_cfd, current_params_cfd = load_data(
            current_params_cfd
        )
        current_params_cfd = plotting_on_ax(
            fig,
            axes[row, 0],
            df_cfd,
            x_mesh_cfd,
            y_mesh_cfd,
            current_params_cfd,
            is_with_xlabel=is_with_xlabel,
            is_with_ylabel=False,
        )

        # Load and plot PIV data for this row
        current_params_piv = current_params | {"is_CFD": False}
        df_piv, x_mesh_piv, y_mesh_piv, current_params_piv = load_data(
            current_params_piv
        )
        if current_params_piv.get("is_with_mask", False):
            df_piv = apply_w_masking(df_piv, current_params_piv)
            df_piv = apply_snr_masking(df_piv, current_params_piv)
        current_params_piv = current_params_piv | {"is_with_mask": False}
        current_params_piv = plotting_on_ax(
            fig,
            axes[row, 1],
            df_piv,
            x_mesh_piv,
            y_mesh_piv,
            current_params_piv,
            is_with_xlabel=is_with_xlabel,
            is_label_left=False,
        )

        # add cbar
        add_vertical_colorbar_for_row(
            fig,
            axes[row, :],
            current_params_piv,
            label=f"$Y{y_num}$\n$u$\n" + r"(ms$^{-1}$)",
            labelpad=21,
            fontsize=13,
        )

    # Save the plot
    # save_path = (
    #     Path(project_dir) / "results" / "paper_plots_21_10_2025" / f"{file_name}.pdf"
    # )
    save_path = (
        Path(project_dir) / "results" / "paper_plots_24_03_2026" / f"{file_name}.pdf"
    )
    tight_layout_pad = float(plot_params.get("tight_layout_pad", 0.02))
    save_bbox_tight = bool(plot_params.get("save_bbox_tight", True))
    save_pad_inches = float(plot_params.get("save_pad_inches", 0.01))

    fig.tight_layout(pad=tight_layout_pad)
    save_kwargs = {}
    if save_bbox_tight:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = save_pad_inches
    fig.savefig(save_path, **save_kwargs)
    plt.close()


def main():
    plot_params: PlotParams = {
        # Basic configuration
        "is_CFD": False,
        "y_num": 3,
        "alpha": 6,
        "project_dir": project_dir,
        "plot_type": ".pdf",
        "title": None,
        "spanwise_CFD": False,
        "is_CFD_PIV_comparison": True,
        "color_data_col_name": "V",
        "is_CFD_PIV_comparison_multicomponent_masked": False,
        "run_for_all_planes": False,
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
        "is_with_bound": False,
        "drot": 0.0,
        "iP": 65,
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
        "is_with_mask": True,
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
        "normal_masked_interpolated": False,
        # Figure export whitespace control
        "tight_layout_pad": 0.02,
        "save_bbox_tight": True,
        "save_pad_inches": 0.01,
        ## Interpolation settings
        "is_with_interpolation": False,
        "interpolation_method": "nearest",
        "rectangle_size": 0.05,
    }
    alphas = [6, 6, 6, 16]
    y_nums = [1, 3, 4, 1]
    file_name = "fig06_qualitative_comparison_CFD_PIV"
    file_name = "fig06"
    plotting_qualitative_CFD_PIV(alphas, y_nums, file_name, plot_params)


if __name__ == "__main__":
    main()
