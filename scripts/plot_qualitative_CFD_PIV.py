from plotting import *
from plot_styling import set_plot_style
from matplotlib.gridspec import GridSpec


def plotting_qualitative_CFD_PIV(plot_params: dict) -> None:

    set_plot_style()

    # Set up alpha and y_num values
    alphas = [6, 6, 6, 16]
    y_nums = [1, 3, 4, 1]
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
    save_path = (
        Path(project_dir)
        / "results"
        / "paper_plots"
        / "qualitative_comparison_CFD_PIV.pdf"
    )
    plt.tight_layout()
    fig.savefig(save_path)
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
        "rho": 1.225,
        "mu": 1.7894e-5,
        "is_with_maximim_vorticity_location_correction": True,
        "chord": 0.37,
        # Mask settings
        "is_with_mask": True,
        "column_to_mask": "w",
        "mask_lower_bound": -3,
        "mask_upper_bound": 3,
        "normal_masked_interpolated": False,
        ## Interpolation settings
        "is_with_interpolation": False,
        "interpolation_method": "nearest",
        "rectangle_size": 0.05,
    }
    plotting_qualitative_CFD_PIV(plot_params)


if __name__ == "__main__":
    main()
