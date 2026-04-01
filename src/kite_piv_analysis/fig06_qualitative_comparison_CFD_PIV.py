from kite_piv_analysis.plotting import *
from kite_piv_analysis.plot_styling import set_plot_style
from matplotlib.gridspec import GridSpec
from kite_piv_analysis.masking import apply_snr_masking, apply_w_masking
from scipy.interpolate import griddata


def _add_horizontal_colorbar_top(
    fig,
    ax,
    cax_mappable,
    vmin,
    vmax,
    label,
    fontsize=13,
    y_offset=0.005,
    tick_step=1,
    ticks=None,
):
    """Add a horizontal colorbar above the given axis."""
    bbox = ax.get_position()
    cbar_height = 0.012
    cbar_ax = fig.add_axes([bbox.x0, bbox.y1 + y_offset, bbox.width, cbar_height])

    if ticks is None:
        ticks = np.arange(int(vmin), int(vmax) + 1, tick_step)

    cbar = plt.colorbar(
        ScalarMappable(norm=cax_mappable.norm, cmap=cax_mappable.cmap),
        cax=cbar_ax,
        ticks=ticks,
        orientation="horizontal",
    )

    cbar_ax.xaxis.set_ticks_position("top")
    cbar_ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(direction="out")
    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    cbar.set_label(label, fontsize=fontsize, labelpad=8)
    cbar.ax.grid(False)
    return cbar


def plotting_qualitative_CFD_PIV(alphas, y_nums, file_name, plot_params: dict) -> None:

    set_plot_style()

    is_with_xlabel = False

    n_rows = len(alphas)
    n_cols = 3

    """Create a comparison of CFD and PIV data for different Y positions and alphas."""
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 12.17),
        gridspec_kw={
            "hspace": 0.01,
            "wspace": 0.01,
        },
    )

    u_inf = plot_params.get("u_inf", 15.0)
    dev_vmin = 0
    dev_vmax = 10

    velocity_cax = None
    dev_cax = None

    for row, (alpha, y_num) in enumerate(zip(alphas, y_nums)):
        # Update plot_params for current alpha and y_num
        current_params = plot_params.copy()
        current_params["alpha"] = alpha
        current_params["y_num"] = y_num

        alpha += 1  # corrections
        print(f"Plotting α = {alpha}, Y = {y_num}")

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
            is_with_ylabel=False,
        )

        # Store velocity cax from first row
        if velocity_cax is None:
            velocity_cax = current_params_piv["cax"]
            velocity_vmin = current_params_piv["min_cbar_value"]
            velocity_vmax = current_params_piv["max_cbar_value"]

        # --- Deviation column (col 2) ---
        col_name = current_params.get("color_data_col_name", "V")

        # Interpolate CFD onto PIV grid points
        cfd_on_piv = griddata(
            (df_cfd["x"].values, df_cfd["y"].values),
            df_cfd[col_name].values,
            (df_piv["x"].values, df_piv["y"].values),
            method="linear",
        )

        # Absolute percentage deviation normalised by U_inf
        deviation = np.abs(cfd_on_piv - df_piv[col_name].values) / u_inf * 100

        # Build synthetic dataframe on the PIV grid with a unique column name
        # so plot_color_contour doesn't override cbar limits via hardcoded ranges
        dev_col = "dev"
        df_dev = df_piv[["x", "y"]].copy()
        df_dev[dev_col] = deviation

        # Plot deviation
        dev_params = current_params.copy()
        dev_params["is_CFD"] = False
        dev_params["min_cbar_value"] = dev_vmin
        dev_params["max_cbar_value"] = dev_vmax
        dev_params["cmap"] = "viridis"
        dev_params["color_data_col_name"] = dev_col
        dev_params["is_with_mask"] = False
        dev_params["is_with_bound"] = False
        dev_params["is_with_quiver"] = False

        dev_params = plotting_on_ax(
            fig,
            axes[row, 2],
            df_dev,
            x_mesh_piv,
            y_mesh_piv,
            dev_params,
            is_with_xlabel=is_with_xlabel,
            is_label_left=False,
        )

        # Store dev cax from first row
        if dev_cax is None:
            dev_cax = dev_params["cax"]

        # Row label on the left
        axes[row, 0].set_ylabel(
            f"$Y{y_num}$\n" + rf"$\alpha = {alpha}^\circ$",
            rotation=0,
            fontsize=13,
            labelpad=30,
            va="center",
            ha="center",
        )

        # Force axis ticks for all columns
        for col in range(n_cols):
            axes[row, col].set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
            axes[row, col].set_yticks([-0.2, 0, 0.2])

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

    fig.tight_layout(pad=tight_layout_pad, rect=[0, 0, 1, 0.95])

    # Add horizontal colorbars at the top
    _add_horizontal_colorbar_top(
        fig,
        axes[0, 0],
        velocity_cax,
        velocity_vmin,
        velocity_vmax,
        label=r"CFD: $u$ (ms$^{-1}$)",
        ticks=[13, 14, 15, 16, 17],
    )
    _add_horizontal_colorbar_top(
        fig,
        axes[0, 1],
        velocity_cax,
        velocity_vmin,
        velocity_vmax,
        label=r"PIV: $u$ (ms$^{-1}$)",
        ticks=[13, 14, 15, 16, 17],
    )
    _add_horizontal_colorbar_top(
        fig,
        axes[0, 2],
        dev_cax,
        dev_vmin,
        dev_vmax,
        label=r"$|\Delta u|/U_\infty$ (\%)",
        tick_step=2,
    )

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
    alphas = [6, 6, 6, 6, 16]
    y_nums = [1, 2, 3, 4, 1]
    file_name = "fig06_qualitative_comparison_CFD_PIV"
    file_name = "fig06"
    plotting_qualitative_CFD_PIV(alphas, y_nums, file_name, plot_params)


if __name__ == "__main__":
    main()
