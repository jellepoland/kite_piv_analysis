from kite_piv_analysis.plotting import *
from kite_piv_analysis.plot_styling import set_plot_style
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from kite_piv_analysis.masking import apply_snr_masking, apply_w_masking


def plotting_qualitative_CFD_PIV(alphas, y_nums, file_name, plot_params: dict) -> None:

    set_plot_style()

    # Set up alpha and y_num values
    # alphas = [6, 6, 6, 16]
    # y_nums = [1, 3, 4, 1]
    # Build column specs for the requested layout
    # Column 1: Y1-Y4 at α = 7° (stored as 6 in the data)
    # Column 2: Y5-Y7 at α = 7° (stored as 6 in the data)
    # Column 3: Y1-Y4 at α = 17° (stored as 16 in the data)
    provided_cases = list(zip(alphas, y_nums))

    column_specs = [
        {"alpha_values": {6, 7}, "y_order": [1, 2, 3, 4], "title_alpha": 7},
        {"alpha_values": {6, 7}, "y_order": [5, 6, 7], "title_alpha": 7},
        {"alpha_values": {16, 17}, "y_order": [1, 2, 3, 4], "title_alpha": 17},
    ]

    n_rows = max(len(col["y_order"]) for col in column_specs)
    n_cols = len(column_specs)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13, 10.5),
        gridspec_kw={
            "hspace": 0.07,  # -0.01,
            "wspace": 0.07,  # 0.08,
        },
    )

    # Convenience to always index axes as 2D
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    cbar_labelpad = plot_params.get("cbar_labelpad", 10)

    for col_idx, spec in enumerate(column_specs):
        # Title on the top subplot of each column
        axes[0, col_idx].set_title(
            rf"$\alpha$ = {spec['title_alpha']}" + r"$^{\circ}$",
            fontsize=14,
            fontweight="bold",
            pad=5,
        )

        y_sequence = spec["y_order"]
        alpha_candidates = spec["alpha_values"]

        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]

            # If this column has fewer rows, hide the extra axes to keep plots top-aligned
            if row_idx >= len(y_sequence):
                if col_idx == 1 and row_idx == n_rows - 1:
                    # Use this empty slot to show a legend for the dashed line
                    handle = Line2D(
                        [],
                        [],
                        color="black",
                        linestyle="--",
                        linewidth=2.5,
                        label="y = 0.25 m",
                    )
                    ax.legend(
                        handles=[handle],
                        loc="center",
                        frameon=True,
                        fontsize=12,
                    )
                    ax.set_axis_off()
                else:
                    ax.axis("off")
                continue

            y_num = y_sequence[row_idx]
            alpha = next(
                (a for a, y in provided_cases if y == y_num and a in alpha_candidates),
                None,
            )

            # If not provided, fall back to the first candidate alpha for this column
            if alpha is None:
                alpha = min(alpha_candidates)

            current_params = plot_params.copy()
            current_params["alpha"] = alpha
            current_params["y_num"] = y_num
            current_params["is_CFD"] = False

            display_alpha = alpha + 1 if alpha in {6, 16} else alpha
            print(f"Plotting α = {display_alpha}, Y = {y_num}")

            try:
                df_piv, x_mesh_piv, y_mesh_piv, current_params_piv = load_data(
                    current_params
                )
            except FileNotFoundError:
                ax.text(
                    0.5,
                    0.5,
                    f"Missing data: α={display_alpha}, Y{y_num}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )
                ax.axis("off")
                continue

            if current_params_piv.get("is_with_mask", False):
                df_piv = apply_w_masking(df_piv, current_params_piv)
                df_piv = apply_snr_masking(df_piv, current_params_piv)
            current_params_piv = current_params_piv | {"is_with_mask": False}

            # Only bottom-most subplot in each column gets an x-label
            is_with_xlabel = row_idx == (len(y_sequence) - 1)

            current_params_piv = plotting_on_ax(
                fig,
                ax,
                df_piv,
                x_mesh_piv,
                y_mesh_piv,
                current_params_piv,
                is_with_xlabel=is_with_xlabel,
                is_label_left=False,
                is_with_ylabel=(col_idx == 2),
            )

            # Y label in the top-left of each subplot
            ax.text(
                0.02,
                0.96,
                f"Y{y_num}",
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

            if col_idx == 0:
                add_vertical_colorbar_for_row(
                    fig,
                    [ax],
                    current_params_piv,
                    label="$u$\n" + r"(ms$^{-1}$)",
                    labelpad=cbar_labelpad,
                    fontsize=12,
                    x_offset=0.03,
                )

            # Add reference line
            ax.axhline(
                y=0.25,
                color="black",
                linestyle="--",
                linewidth=2,
                label=r"$z = 0.25$ m",
            )

    # Save the plot
    # save_path = (
    #     Path(project_dir) / "results" / "paper_plots_21_10_2025" / f"{file_name}.pdf"
    # )
    save_path = (
        Path(project_dir) / "results" / "paper_plots_24_03_2026" / f"{file_name}.pdf"
    )
    plt.tight_layout(pad=0.02)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
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
        "cbar_labelpad": 21,
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
        ## Interpolation settings
        "is_with_interpolation": False,
        "interpolation_method": "nearest",
        "rectangle_size": 0.05,
    }
    alphas = [6, 6, 6, 6, 6, 6, 16, 16, 16, 16]
    y_nums = [1, 3, 4, 5, 6, 7, 1, 2, 3, 4]
    # file_name = "fig15_line_interference_PIV"
    file_name = "figD1"
    plotting_qualitative_CFD_PIV(alphas, y_nums, file_name, plot_params)


if __name__ == "__main__":
    main()
