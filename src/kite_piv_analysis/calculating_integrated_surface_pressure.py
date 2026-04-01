import pandas as pd
import numpy as np
import os
import argparse
import scipy.interpolate as interpolate
from scipy.interpolate import griddata
from pathlib import Path
from kite_piv_analysis.utils import project_dir, interp2d_batch, csv_reader
from kite_piv_analysis.plot_styling import set_plot_style, PALETTE
import matplotlib.pyplot as plt


def visualize_interpolated_pressures(surface_x, surface_y, interpolated_pressures):
    """
    Visualize interpolated pressures on the airfoil surface

    Parameters:
    -----------
    df : pandas.DataFrame
        Pressure field data with 'x', 'y', 'pressure' columns
    surface_x : numpy.ndarray
        X coordinates of airfoil surface points
    surface_y : numpy.ndarray
        Y coordinates of airfoil surface points

    Returns:
    --------
    numpy.ndarray
        Interpolated pressure values
    """
    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot airfoil surface with interpolated pressures
    scatter = plt.scatter(
        surface_x, surface_y, c=interpolated_pressures, cmap="coolwarm", s=50
    )
    plt.title("Interpolated Pressures on Airfoil Surface")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.colorbar(scatter, label="Interpolated Pressure")

    plt.tight_layout()
    plt.show()


def verify_surface_normals(surface_x, surface_y):
    surface_points = np.column_stack((surface_x, surface_y))
    normals, _ = compute_surface_normals(surface_points)

    plt.figure(figsize=(10, 6))
    plt.plot(surface_x, surface_y, "b-", label="Airfoil Surface")

    # Plot a subset of normals for clarity
    skip = max(1, len(normals) // int(len(normals)))
    for i in range(0, len(normals), skip):
        midpoint = (surface_points[i] + surface_points[i + 1]) / 2
        plt.quiver(
            midpoint[0],
            midpoint[1],
            normals[i][0],
            normals[i][1],
            color="r",
            scale=10,
            width=0.003,
        )

    plt.title("Airfoil Surface with Normal Vectors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.show()


def running_NOCA(
    df,
    alpha: int,
    y_num: int,
    mu: float = 1.7894e-5,
    is_with_maximim_vorticity_location_correction: bool = True,
    U_inf: float = 15,
):

    from kite_piv_analysis import calculating_airfoil_centre
    from kite_piv_analysis import force_from_noca
    from kite_piv_analysis.utils import reading_optimal_bound_placement
    from kite_piv_analysis.defining_bound_volume import (
        boundary_ellipse,
        boundary_rectangle,
    )

    x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
        alpha, y_num, is_with_chord=True
    )
    d1centre = (x_airfoil, y_airfoil)
    drot = 0
    iP = 360

    # Run NOCA analysis with unified density.
    rho = 1.20
    dLx, dLy, iP = reading_optimal_bound_placement(
        alpha, y_num, is_with_N_datapoints=True
    )
    d2curve = boundary_ellipse(
        d1centre,
        drot,
        dLx,
        dLy,
        iP,
    )
    F_x, F_y, C_l, C_d = force_from_noca.main(
        df,
        d2curve,
        mu=mu,
        is_with_maximim_vorticity_location_correction=is_with_maximim_vorticity_location_correction,
        rho=rho,
        U_inf=U_inf,
        ref_chord=chord,
    )
    return F_x, F_y, C_l, C_d


def compute_surface_normals(surface_points):
    """
    Compute surface normals with a more robust method.

    Parameters:
    -----------
    surface_points : numpy.ndarray
        Array of surface point coordinates.

    Returns:
    --------
    numpy.ndarray of surface normals and segment lengths.
    """
    normals = []
    segment_lengths = []
    for i in range(len(surface_points) - 1):
        # Compute tangent vector
        tangent = surface_points[i + 1] - surface_points[i]

        # Compute perpendicular vector (rotated 90 degrees)
        # Use right-hand rule to determine normal direction
        normal = np.array([-tangent[1], tangent[0]])

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Append normal and segment length
        normals.append(normal)
        segment_lengths.append(np.linalg.norm(tangent))

    return np.array(normals), np.array(segment_lengths)


def compute_surface_forces(
    df,
    x_surface,
    y_surface,
    density,
    p_ref,
    mu,
    is_plot=False,
):
    """
    Compute aerodynamic forces on a 2D airfoil surface from CFD data.
    """
    surface_points = np.column_stack((x_surface, y_surface))
    # `density` is kept in the signature for backward compatibility.
    # Pressure and wall-shear fields in processed CFD are already dimensional.
    _ = density

    # Precompute normals and segment lengths
    normals, segment_lengths = compute_surface_normals(surface_points)
    segment_midpoints = 0.5 * (surface_points[:-1] + surface_points[1:])

    # Interpolate CFD fields onto segment midpoints (consistent with segment-based integration).
    pressure_surface = griddata(
        (df["x"], df["y"]), df["pressure"], segment_midpoints, method="linear"
    )

    # Preferred viscous source: CFD wall shear traction already on the surface.
    has_tau_w = ("tau_w_x" in df.columns) and ("tau_w_y" in df.columns)
    tau_w_x_surface = None
    tau_w_y_surface = None
    if has_tau_w:
        tau_w_x_surface = griddata(
            (df["x"], df["y"]), df["tau_w_x"], segment_midpoints, method="linear"
        )
        tau_w_y_surface = griddata(
            (df["x"], df["y"]), df["tau_w_y"], segment_midpoints, method="linear"
        )

    # Fallback viscous source from gradients if wall shear is unavailable.
    has_gradients = all(c in df.columns for c in ["dudx", "dudy", "dvdx", "dvdy"])
    dudx_surface = dudy_surface = dvdx_surface = dvdy_surface = None
    if has_gradients:
        dudx_surface = griddata(
            (df["x"], df["y"]), df["dudx"], segment_midpoints, method="linear"
        )
        dudy_surface = griddata(
            (df["x"], df["y"]), df["dudy"], segment_midpoints, method="linear"
        )
        dvdx_surface = griddata(
            (df["x"], df["y"]), df["dvdx"], segment_midpoints, method="linear"
        )
        dvdy_surface = griddata(
            (df["x"], df["y"]), df["dvdy"], segment_midpoints, method="linear"
        )

    # Initialize force components
    Fx_p, Fy_p = 0, 0  # Pressure forces
    Fx_v, Fy_v = 0, 0  # Viscous forces

    # NaN guards and viscous-mode selection.
    valid_pressure = np.isfinite(pressure_surface)
    if np.count_nonzero(valid_pressure) < 3:
        raise ValueError(
            "Too few valid pressure interpolation points on airfoil surface "
            f"({np.count_nonzero(valid_pressure)} valid segments)."
        )

    viscous_mode = None
    if has_tau_w and tau_w_x_surface is not None and tau_w_y_surface is not None:
        valid_tau = np.isfinite(tau_w_x_surface) & np.isfinite(tau_w_y_surface)
        if np.count_nonzero(valid_pressure & valid_tau) >= 3:
            viscous_mode = "tau_w"
    if viscous_mode is None and has_gradients:
        valid_grad = (
            np.isfinite(dudx_surface)
            & np.isfinite(dudy_surface)
            & np.isfinite(dvdx_surface)
            & np.isfinite(dvdy_surface)
        )
        if np.count_nonzero(valid_pressure & valid_grad) >= 3:
            viscous_mode = "gradients"

    # Loop through segments
    for i in range(len(normals)):
        if not valid_pressure[i]:
            continue

        normal = normals[i]
        length = segment_lengths[i]

        # Pressure force contribution
        pressure_difference = -(pressure_surface[i] - p_ref)
        Fx_p += normal[0] * pressure_difference * length
        Fy_p += normal[1] * pressure_difference * length

        # Viscous force contribution
        if viscous_mode == "tau_w":
            if np.isfinite(tau_w_x_surface[i]) and np.isfinite(tau_w_y_surface[i]):
                Fx_v += tau_w_x_surface[i] * length
                Fy_v += tau_w_y_surface[i] * length
        elif viscous_mode == "gradients":
            if (
                np.isfinite(dudx_surface[i])
                and np.isfinite(dudy_surface[i])
                and np.isfinite(dvdx_surface[i])
                and np.isfinite(dvdy_surface[i])
            ):
                # Deviatoric stress tensor components (2D approximation).
                R_dev_x = 2 * mu * dudx_surface[i]
                R_dev_y = 2 * mu * dvdy_surface[i]
                shear_xy = mu * (dudy_surface[i] + dvdx_surface[i])
                Fx_v += (R_dev_x * normal[0] + shear_xy * normal[1]) * length
                Fy_v += (shear_xy * normal[0] + R_dev_y * normal[1]) * length

    # Combine contributions

    Fx_total = Fx_p + Fx_v
    Fy_total = Fy_p + Fy_v
    print(
        f"Fx_p: {Fx_p:.2f}, Fy_p: {Fy_p:.2f}, "
        f"Fx_v: {Fx_v:.4f}, Fy_v: {Fy_v:.4f}, viscous_mode: {viscous_mode}"
    )

    # Visualizing
    if is_plot:
        verify_surface_normals(x_surface, y_surface)
        visualize_interpolated_pressures(x_surface, y_surface, pressure_surface)

        surface_x_airfoil = x_surface
        surface_y_airfoil = y_surface

        # extract surface from a filter based on tau_w
        df["tau_w_mag"] = np.sqrt(df["tau_w_x"] ** 2 + df["tau_w_y"] ** 2)
        df_surface = df[df["tau_w_mag"] > 1e-3]
        surface_x_tau_w = df_surface["x"].values
        surface_y_tau_w = df_surface["y"].values
        plt.plot(
            surface_x_airfoil,
            surface_y_airfoil,
            "r",
            label="Airfoil Surface (used for force calculation)",
        )
        plt.pcolormesh(
            np.unique(df["x"]),
            np.unique(df["y"]),
            df.pivot(index="y", columns="x", values="pressure").values,
            shading="auto",
            cmap="jet",
        )

        # plt.scatter(surface_x_tau_w, surface_y_tau_w, c="b", label="Airfoil Surface")
        plt.axis("equal")
        plt.show()

    return Fx_total, Fy_total


TABLE_ALPHA_TO_Y_NUMS = {
    6: [1, 2, 3, 4, 5, 6, 7],
    16: [1],
}
DEFAULT_PRESSURE_CACHE_PATH = (
    Path(project_dir) / "processed_data" / "pressure_integration_coefficients.csv"
)


def _reference_chords(alpha_to_y_nums: dict[int, list[int]]) -> dict[int, float]:
    from kite_piv_analysis import calculating_airfoil_centre

    chord_ref: dict[int, float] = {}
    for alpha in sorted(alpha_to_y_nums):
        y_ref = alpha_to_y_nums[alpha][0]
        _, _, chord = calculating_airfoil_centre.main(alpha, y_ref, is_with_chord=True)
        chord_ref[alpha] = chord
    return chord_ref


def _q_infc(
    alpha: int,
    y_num: int,
    alpha_to_y_nums: dict[int, list[int]],
    rho: float,
    U_inf: float,
    chord_ref: dict[int, float],
) -> float:
    from kite_piv_analysis import calculating_airfoil_centre

    _, _, chord = calculating_airfoil_centre.main(alpha, y_num, is_with_chord=True)
    if alpha == 6 and 6 in alpha_to_y_nums:
        chord_for_scale = chord_ref.get(6, chord)
    else:
        chord_for_scale = chord
    return 0.5 * rho * U_inf**2 * chord_for_scale


def compute_pressure_integration_coefficients(
    alpha_to_y_nums=None, is_plot: bool = False
) -> pd.DataFrame:
    """
    Compute CFD pressure-integration coefficients used by quantitative tables.

    Returns columns:
    alpha, y_num, C_l_p, C_d_p, Fx_total, Fy_total, q_infc
    """
    from kite_piv_analysis import plotting

    if alpha_to_y_nums is None:
        alpha_to_y_nums = TABLE_ALPHA_TO_Y_NUMS

    rho = 1.2
    U_inf = 15.0
    Re = 1e6
    p_ref = 0.0
    mu = (rho * U_inf * 1.0) / Re

    chord_ref = _reference_chords(alpha_to_y_nums)
    rows = []
    for alpha in sorted(alpha_to_y_nums.keys()):
        for y_num in alpha_to_y_nums[alpha]:
            print(f"Processing pressure integration for alpha={alpha}, Y{y_num} ...")
            df = csv_reader(is_CFD=True, alpha=alpha, y_num=y_num, alpha_d_rod=7)
            q_infc = _q_infc(
                alpha, y_num, alpha_to_y_nums, rho=rho, U_inf=U_inf, chord_ref=chord_ref
            )

            surface_x, surface_y = plotting.plot_airfoil(
                None, {"y_num": y_num, "alpha": alpha}, is_return_surface_points=True
            )
            surface_x = surface_x[::-1]
            surface_y = surface_y[::-1]

            Fx_total, Fy_total = compute_surface_forces(
                df,
                x_surface=surface_x,
                y_surface=surface_y,
                density=rho,
                p_ref=p_ref,
                mu=mu,
                is_plot=is_plot,
            )
            rows.append(
                {
                    "alpha": int(alpha),
                    "y_num": int(y_num),
                    "C_l_p": float(Fy_total / q_infc),
                    "C_d_p": float(Fx_total / q_infc),
                    "Fx_total": float(Fx_total),
                    "Fy_total": float(Fy_total),
                    "q_infc": float(q_infc),
                }
            )
    return pd.DataFrame(rows)


def _compute_surface_pressure_diagnostics(
    *,
    alpha: int,
    y_num: int,
    rho: float,
    U_inf: float,
    p_ref: float,
    alpha_to_y_nums: dict[int, list[int]],
    chord_ref: dict[int, float],
) -> dict:
    """
    Compute pressure-only sectional diagnostics along the airfoil contour.

    Returned fields include:
    s_star, cp, dCl_p, dCd_p, cum_Cl_p, cum_Cd_p, Cl_p_total, Cd_p_total
    """
    from kite_piv_analysis import plotting

    df = csv_reader(is_CFD=True, alpha=alpha, y_num=y_num, alpha_d_rod=7)
    x_surface, y_surface = plotting.plot_airfoil(
        None, {"y_num": y_num, "alpha": alpha}, is_return_surface_points=True
    )
    # Keep contour orientation consistent with pressure-integration coefficients.
    x_surface = x_surface[::-1]
    y_surface = y_surface[::-1]

    surface_points = np.column_stack((x_surface, y_surface))
    normals, segment_lengths = compute_surface_normals(surface_points)
    segment_midpoints = 0.5 * (surface_points[:-1] + surface_points[1:])

    pressure_surface = griddata(
        (df["x"], df["y"]), df["pressure"], segment_midpoints, method="linear"
    )
    valid = np.isfinite(pressure_surface)
    if np.count_nonzero(valid) < 3:
        raise ValueError(
            f"Too few valid pressure interpolation points for alpha={alpha}, Y{y_num}."
        )

    segment_s = np.cumsum(segment_lengths) - 0.5 * segment_lengths
    segment_s = segment_s / np.sum(segment_lengths)

    dynamic_pressure = 0.5 * rho * U_inf**2
    cp = (pressure_surface - p_ref) / dynamic_pressure
    pressure_difference = -(pressure_surface - p_ref)

    dFx = np.zeros_like(segment_lengths, dtype=float)
    dFy = np.zeros_like(segment_lengths, dtype=float)
    dFx[valid] = normals[valid, 0] * pressure_difference[valid] * segment_lengths[valid]
    dFy[valid] = normals[valid, 1] * pressure_difference[valid] * segment_lengths[valid]

    q_infc = _q_infc(
        alpha,
        y_num,
        alpha_to_y_nums,
        rho=rho,
        U_inf=U_inf,
        chord_ref=chord_ref,
    )
    dCd_p = dFx / q_infc
    dCl_p = dFy / q_infc
    cum_Cd_p = np.cumsum(dCd_p)
    cum_Cl_p = np.cumsum(dCl_p)

    return {
        "s_star": segment_s,
        "cp": cp,
        "dCl_p": dCl_p,
        "dCd_p": dCd_p,
        "cum_Cl_p": cum_Cl_p,
        "cum_Cd_p": cum_Cd_p,
        "Cl_p_total": float(np.sum(dCl_p)),
        "Cd_p_total": float(np.sum(dCd_p)),
    }


def plot_pressure_integration_diagnostics(
    *,
    alpha: int = 6,
    y_nums: list[int] | None = None,
    focus_y_num: int | None = 4,
    save_path: Path | None = None,
) -> Path:
    """
    Diagnostic 1x3 plot for sectional pressure integration:
    C_p(s*), cumulative C_l,p(s*), cumulative C_d,p(s*).
    """
    set_plot_style()

    if y_nums is None:
        y_nums = [2, 3, 4, 5]
    y_nums = sorted({int(y) for y in y_nums})
    if len(y_nums) < 2:
        raise ValueError("At least two y-planes are required for diagnostics.")

    if focus_y_num is None or focus_y_num not in y_nums:
        focus_y_num = 4 if 4 in y_nums else y_nums[len(y_nums) // 2]

    rho = 1.2
    U_inf = 15.0
    p_ref = 0.0
    alpha_to_y_nums = {int(alpha): y_nums}
    chord_ref = _reference_chords(alpha_to_y_nums)

    diagnostics: dict[int, dict] = {}
    for y_num in y_nums:
        diagnostics[y_num] = _compute_surface_pressure_diagnostics(
            alpha=alpha,
            y_num=y_num,
            rho=rho,
            U_inf=U_inf,
            p_ref=p_ref,
            alpha_to_y_nums=alpha_to_y_nums,
            chord_ref=chord_ref,
        )

    if save_path is None:
        y_token = "_".join(str(y) for y in y_nums)
        save_path = (
            Path(project_dir)
            / "results"
            / "paper_plots_24_03_2026"
            / f"pressure_integration_diagnostic_alpha_{alpha}_Y{y_token}.pdf"
        )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.8),
        sharex=True,
        gridspec_kw={"wspace": 0.23},
    )
    ax_cp, ax_cl, ax_cd = axes

    color_cycle = [
        PALETTE["Black"],
        PALETTE["Orange"],
        PALETTE["Sky Blue"],
        PALETTE["Bluish Green"],
        PALETTE["Yellow"],
        PALETTE["Blue"],
        PALETTE["Vermillion"],
        PALETTE["Reddish Purple"],
    ]
    for idx, y_num in enumerate(y_nums):
        curr = diagnostics[y_num]
        is_focus = y_num == focus_y_num
        color = color_cycle[idx % len(color_cycle)]
        lw = 2.4 if is_focus else 1.5
        alpha_line = 1.0 if is_focus else 0.9
        label = f"$Y{y_num}$"
        ax_cp.plot(
            curr["s_star"],
            curr["cp"],
            color=color,
            linewidth=lw,
            alpha=alpha_line,
            label=label,
        )
        ax_cl.plot(
            curr["s_star"],
            curr["cum_Cl_p"],
            color=color,
            linewidth=lw,
            alpha=alpha_line,
            label=f"Y{y_num}",
        )
        ax_cd.plot(
            curr["s_star"],
            curr["cum_Cd_p"],
            color=color,
            linewidth=lw,
            alpha=alpha_line,
            label=f"Y{y_num}",
        )

    ax_cp.invert_yaxis()
    ax_cp.set_xlabel(r"Normalized contour coordinate $s^*$ [-]")
    ax_cp.set_ylabel(r"$C_p$ [-]")
    ax_cp.grid(alpha=0.25, linestyle="--")

    ax_cl.set_xlabel(r"Normalized contour coordinate $s^*$ [-]")
    ax_cl.set_ylabel(r"Cumulative $C_{l,p}$ [-]")
    ax_cl.grid(alpha=0.25, linestyle="--")

    ax_cd.set_xlabel(r"Normalized contour coordinate $s^*$ [-]")
    ax_cd.set_ylabel(r"Cumulative $C_{d,p}$ [-]")
    ax_cd.grid(alpha=0.25, linestyle="--")

    legend_handles, legend_labels = ax_cp.get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=max(1, len(y_nums)),
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return save_path


def save_pressure_integration_coefficients(
    df: pd.DataFrame, save_path: Path = DEFAULT_PRESSURE_CACHE_PATH
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute pressure-integration coefficients and diagnostics."
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Generate pressure-integration diagnostic plot instead of CSV table.",
    )
    parser.add_argument("--alpha", type=int, default=6, help="Angle of attack.")
    parser.add_argument(
        "--y-nums",
        type=str,
        default="2,3,4,5",
        help="Comma-separated y-plane numbers, e.g. '2,3,4,5'.",
    )
    parser.add_argument(
        "--focus-y",
        type=int,
        default=4,
        help="Focus y-plane used for delta Cp panel.",
    )
    args = parser.parse_args()

    if args.diagnostic:
        y_nums = [int(token) for token in args.y_nums.split(",") if token.strip()]
        save_path = plot_pressure_integration_diagnostics(
            alpha=int(args.alpha),
            y_nums=y_nums,
            focus_y_num=int(args.focus_y),
        )
        print(f"\nSaved diagnostic pressure plot to: {save_path}")
        return

    df = compute_pressure_integration_coefficients()
    save_path = save_pressure_integration_coefficients(df)
    print(f"\nSaved pressure-integration coefficients to: {save_path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
