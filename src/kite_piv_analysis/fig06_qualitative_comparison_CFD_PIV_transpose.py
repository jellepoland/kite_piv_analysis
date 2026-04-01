from kite_piv_analysis.plotting import *
from kite_piv_analysis.plot_styling import set_plot_style
from matplotlib.gridspec import GridSpec
from kite_piv_analysis.masking import apply_snr_masking, apply_w_masking


def _add_vertical_colorbar_right_side(
    fig, ax_right, plot_params, label=None, fontsize=13, labelpad=21, x_offset=0.025
):
    """Add a vertical colorbar on the right side of the given axis."""
    cax = plot_params["cax"]
    vmin = plot_params["min_cbar_value"]
    vmax = plot_params["max_cbar_value"]

    bbox = ax_right.get_position()
    cbar_ax = fig.add_axes([bbox.x1 + x_offset, bbox.y0, 0.02, bbox.height])

    cbar = plt.colorbar(
        ScalarMappable(norm=cax.norm, cmap=cax.cmap),
        cax=cbar_ax,
        ticks=np.linspace(int(vmin), int(vmax), 10),
        orientation="vertical",
    )

    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(direction="out")
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    if label is None:
        label = "$u$\n" + r"(ms$^{-1}$)"

    cbar.set_label(label, labelpad=labelpad, fontsize=fontsize, rotation=0)
    cbar.ax.grid(False)
    return cbar


def plotting_qualitative_CFD_PIV(
    left_planes, right_planes, file_name, plot_params: dict
) -> None:
    """Create a 4-column comparison: CFD|PIV (left group) | CFD|PIV (right group)."""

    set_plot_style()

    n_rows = max(len(left_planes), len(right_planes))
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(19, n_rows * 3.0),
        gridspec_kw={
            "hspace": 0.01,
            "wspace": 0.07,
        },
    )

    for row in range(n_rows):
        has_left = row < len(left_planes)
        has_right = row < len(right_planes)

        # --- Left group (cols 0-1) ---
        if has_left:
            alpha, y_num = left_planes[row]
            current_params = plot_params.copy()
            current_params["alpha"] = alpha
            current_params["y_num"] = y_num
            alpha_disp = alpha + 1  # correction
            is_bottom_left = row == len(left_planes) - 1

            print(f"Plotting left: α = {alpha_disp}, Y = {y_num}")

            title_cfd = (
                rf"CFD, $\alpha = {alpha_disp}$" + r"$^{\circ}$" + f", $Y{y_num}$"
            )
            title_piv = (
                rf"PIV, $\alpha = {alpha_disp}$" + r"$^{\circ}$" + f", $Y{y_num}$"
            )

            # CFD on col 0
            p_cfd = current_params | {"is_CFD": True}
            df_cfd, x_mesh_cfd, y_mesh_cfd, p_cfd = load_data(p_cfd)
            p_cfd = plotting_on_ax(
                fig,
                axes[row, 0],
                df_cfd,
                x_mesh_cfd,
                y_mesh_cfd,
                p_cfd,
                is_with_xlabel=is_bottom_left,
                is_with_ylabel=False,
                is_label_left=True,
            )
            axes[row, 0].set_title(title_cfd, fontsize=12, fontweight="bold", pad=5)

            # PIV on col 1
            p_piv = current_params | {"is_CFD": False}
            df_piv, x_mesh_piv, y_mesh_piv, p_piv = load_data(p_piv)
            if p_piv.get("is_with_mask", False):
                df_piv = apply_w_masking(df_piv, p_piv)
                df_piv = apply_snr_masking(df_piv, p_piv)
            p_piv = p_piv | {"is_with_mask": False}
            p_piv = plotting_on_ax(
                fig,
                axes[row, 1],
                df_piv,
                x_mesh_piv,
                y_mesh_piv,
                p_piv,
                is_with_xlabel=is_bottom_left,
                is_with_ylabel=False,
                is_label_left=False,
            )
            axes[row, 1].set_title(title_piv, fontsize=12, fontweight="bold", pad=5)

            # Colorbar to the left of the row
            add_vertical_colorbar_for_row(
                fig,
                axes[row, :2],
                p_piv,
                label="$u$\n" + r"(ms$^{-1}$)",
                labelpad=21,
                fontsize=13,
            )

        # --- Right group (cols 2-3) ---
        if has_right:
            alpha, y_num = right_planes[row]
            current_params = plot_params.copy()
            current_params["alpha"] = alpha
            current_params["y_num"] = y_num
            alpha_disp = alpha + 1
            is_bottom_right = row == len(right_planes) - 1

            print(f"Plotting right: α = {alpha_disp}, Y = {y_num}")

            title_cfd = (
                rf"CFD, $\alpha = {alpha_disp}$" + r"$^{\circ}$" + f", $Y{y_num}$"
            )
            title_piv = (
                rf"PIV, $\alpha = {alpha_disp}$" + r"$^{\circ}$" + f", $Y{y_num}$"
            )

            # CFD on col 2
            p_cfd = current_params | {"is_CFD": True}
            df_cfd, x_mesh_cfd, y_mesh_cfd, p_cfd = load_data(p_cfd)
            p_cfd = plotting_on_ax(
                fig,
                axes[row, 2],
                df_cfd,
                x_mesh_cfd,
                y_mesh_cfd,
                p_cfd,
                is_with_xlabel=is_bottom_right,
                is_with_ylabel=False,
                is_label_left=False,
            )
            axes[row, 2].set_title(title_cfd, fontsize=12, fontweight="bold", pad=5)

            # PIV on col 3
            p_piv = current_params | {"is_CFD": False}
            df_piv, x_mesh_piv, y_mesh_piv, p_piv = load_data(p_piv)
            if p_piv.get("is_with_mask", False):
                df_piv = apply_w_masking(df_piv, p_piv)
                df_piv = apply_snr_masking(df_piv, p_piv)
            p_piv = p_piv | {"is_with_mask": False}
            p_piv = plotting_on_ax(
                fig,
                axes[row, 3],
                df_piv,
                x_mesh_piv,
                y_mesh_piv,
                p_piv,
                is_with_xlabel=is_bottom_right,
                is_with_ylabel=True,
                is_label_left=False,
            )
            axes[row, 3].set_title(title_piv, fontsize=12, fontweight="bold", pad=5)
        else:
            # Hide unused right-group axes
            axes[row, 2].set_visible(False)
            axes[row, 3].set_visible(False)

    # Save the plot
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
    left_planes = [(6, 1), (6, 2), (6, 3)]
    right_planes = [(6, 4), (16, 1), (16, 2)]
    file_name = "fig06"
    plotting_qualitative_CFD_PIV(left_planes, right_planes, file_name, plot_params)


if __name__ == "__main__":
    main()
