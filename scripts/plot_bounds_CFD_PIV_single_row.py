from plotting import *
from plot_styling import set_plot_style
from matplotlib.gridspec import GridSpec


def main() -> None:

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
        "is_with_bound": True,
        "drot": 0.0,
        "iP": 65,
        ##
        "ellipse_color": "black",
        "rectangle_color": "white",
        "bound_linewidth": 1.5,
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
        "is_with_interpolation": True,
        "interpolation_method": "nearest",
        "rectangle_size": 0.05,
        "n_lim": 100,
    }
    set_plot_style()

    # Set up single alpha and y_num value
    alpha = 6
    y_num = 3
    is_with_xlabel = True  # Enable x labels since we only have one row

    # Set up for a single row plot with 2 columns (CFD and PIV)
    n_rows = 1
    n_cols = 2

    """Create a comparison of CFD and PIV data for a single Y position and alpha."""
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10, 3.25),  # Adjusted height for a single row
        gridspec_kw={
            "wspace": 0.07,
        },
    )

    # For a single row, axes is a 1D array, so we need to handle differently
    # Update plot_params for current alpha and y_num
    current_params = plot_params.copy()
    current_params["alpha"] = alpha
    current_params["y_num"] = y_num
    print(f"Plotting α = {alpha}, Y = {y_num}")

    # Add figure titles
    axes[0].set_title(
        rf"CFD",
        fontsize=14,
        fontweight="bold",
        pad=5,
    )
    axes[1].set_title(
        rf"PIV",
        fontsize=14,
        fontweight="bold",
        pad=5,
    )

    # Load and plot CFD data
    current_params_cfd = current_params | {"is_CFD": True}
    df_cfd, x_mesh_cfd, y_mesh_cfd, current_params_cfd = load_data(current_params_cfd)
    current_params_cfd = plotting_on_ax(
        fig,
        axes[0],
        df_cfd,
        x_mesh_cfd,
        y_mesh_cfd,
        current_params_cfd,
        is_with_xlabel=is_with_xlabel,
        is_with_ylabel=False,
    )

    # Load and plot PIV data
    current_params_piv = current_params | {"is_CFD": False}
    df_piv, x_mesh_piv, y_mesh_piv, current_params_piv = load_data(current_params_piv)
    current_params_piv = plotting_on_ax(
        fig,
        axes[1],
        df_piv,
        x_mesh_piv,
        y_mesh_piv,
        current_params_piv,
        is_with_xlabel=is_with_xlabel,
        is_label_left=False,
    )

    # Add colorbar
    add_vertical_colorbar_for_row(
        fig,
        axes,
        current_params_piv,
        label=f"$Y{y_num}$\n$|V|$\n(ms$^{{-1}}$)",
        labelpad=21,
        fontsize=13,
    )

    # Save the plot
    save_path = (
        Path(project_dir) / "results" / "paper_plots" / "bounds_CFD_PIV_single_row.pdf"
    )
    fig.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    main()
