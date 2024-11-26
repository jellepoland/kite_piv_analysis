from plotting import *
from plot_styling import set_plot_style


def normal_masked_interpolated(plot_params: dict) -> None:
    """Create a 4x3 comparison of CFD and PIV data, with PIV masked/unmasked."""
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 10),
        gridspec_kw={
            "hspace": 0.01,  # A bit more vertical space for labels
            "wspace": 0.07,
        },
    )  # Minimal horizontal space since colorbars are outside)
    # fig.suptitle(
    #     rf'Y{plot_params["y_num"]} | α = {plot_params["alpha"]}° | V_inf = {plot_params["u_inf"]}m/s'
    # )

    is_with_circulation_analysis = plot_params["is_with_circulation_analysis"]
    is_with_bound = plot_params["is_with_bound"]
    is_with_interpolation = plot_params["is_with_interpolation"]
    data_labels = ["u", "v", "w"]

    for i, label in enumerate(data_labels):

        if i == 2:
            is_with_xlabel = True
        else:
            is_with_xlabel = False

        # Update color data label
        plot_params["color_data_col_name"] = label

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
        plot_params["column_to_mask"] = "w"
        plot_params["mask_lower_bound"] = -2.5
        plot_params["mask_upper_bound"] = 2.5
        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
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

            ##TODO: change plotparmaters for the interpolation

        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
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
            axes[i, 2].set_title(f"PIV Masked \& Interpolated")
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
    """Create a 3x2 comparison of CFD and PIV data, with PIV masked/unmasked."""
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10, 10),
        gridspec_kw={"hspace": 0.01, "wspace": 0.07},
    )  # Minimal spacing for compact layout

    data_labels = ["u", "v", "w"]
    plot_params.update(
        {
            "is_with_interpolation": False,
            "is_with_bound": False,
            "is_with_circulation_analysis": False,
        }
    )

    for i, label in enumerate(data_labels):
        # Determine x-axis label settings
        is_with_xlabel = i == 2  # Only for the last row

        # Update color data label
        plot_params["color_data_col_name"] = label

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
        )

        ### PIV Mask (right column)
        plot_params["is_with_mask"] = True
        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
        plot_params = plotting_on_ax(
            fig,
            axes[i, 1],
            df_piv,
            x_mesh_piv,
            y_mesh_piv,
            plot_params,
            is_with_xlabel=is_with_xlabel,
            is_with_ylabel=True,  # Only for the 2nd row
            is_label_left=False,
        )

        # setting titles
        if i == 0:
            axes[i, 0].set_title(f"PIV Raw")
            axes[i, 1].set_title(
                # f'PIV Masked for {plot_params["column_to_mask"]} in bounds {plot_params["mask_lower_bound"]} to {plot_params["mask_upper_bound"]}'
                f"PIV Masked"
            )

        # adding cbar
        add_vertical_colorbar_for_row(fig, axes[i, :], plot_params)

        ### Reset things
        plot_params["min_cbar_value"] = None
        plot_params["max_cbar_value"] = None

        # Save the plot
    save_path = (
        Path(project_dir)
        / "results"
        / "paper_plots"
        / f"PIV_normal_masked_Y{plot_params['y_num']}.pdf"
    )
    fig.savefig(save_path)
    plt.close()


if __name__ == "__main__":

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
        "rho": 1.225,
        "mu": 1.7894e-5,
        "is_with_maximim_vorticity_location_correction": True,
        "chord": 0.37,
        # Mask settings
        "is_with_mask": False,
        "column_to_mask": "w",
        "mask_lower_bound": -3,
        "mask_upper_bound": 3,
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
