import numpy as np
from pathlib import Path
import pandas as pd
from scipy.signal import convolve2d
from utils import (
    project_dir,
    reshape_remove_nans,
    interp2d_batch,
    reading_optimal_bound_placement,
)
from defining_bound_volume import boundary_ellipse, boundary_rectangle
import calculating_airfoil_centre
from transforming_paraview_output import process_csv


def calculate_force_guanqun(
    df_1D: pd.DataFrame,
    d2curve: np.ndarray,
    mu,
    rho,
):

    # Reshape data
    n_rows = len(np.unique(df_1D["y"].values))
    n_cols = len(np.unique(df_1D["x"].values))

    # Reshape all relevant fields
    d2x = reshape_remove_nans(df_1D["x"], n_rows, n_cols)
    d2y = reshape_remove_nans(df_1D["y"], n_rows, n_cols)
    d2u = reshape_remove_nans(df_1D["u"], n_rows, n_cols)
    d2v = reshape_remove_nans(df_1D["v"], n_rows, n_cols)
    d2dudx = reshape_remove_nans(df_1D["dudx"], n_rows, n_cols)
    d2dudy = reshape_remove_nans(df_1D["dudy"], n_rows, n_cols)
    d2dvdx = reshape_remove_nans(df_1D["dvdx"], n_rows, n_cols)
    d2dvdy = reshape_remove_nans(df_1D["dvdy"], n_rows, n_cols)

    # Get pressure
    d2pressure = reshape_remove_nans(df_1D["pressure"], n_rows, n_cols)

    # Ensure inputs are NumPy arrays
    d2curve = np.asarray(d2curve)
    # Normal vector calculation - MATLAB style
    n1 = np.gradient(d2curve[:, 1]) / np.sqrt(
        np.gradient(d2curve[:, 0]) ** 2 + np.gradient(d2curve[:, 1]) ** 2
    )
    n2 = -np.gradient(d2curve[:, 0]) / np.sqrt(
        np.gradient(d2curve[:, 0]) ** 2 + np.gradient(d2curve[:, 1]) ** 2
    )
    n3 = np.zeros_like(n1)
    u = interp2d_batch(d2x, d2y, d2u, d2curve)
    v = interp2d_batch(d2x, d2y, d2v, d2curve)
    w = np.zeros_like(u)
    V = np.sqrt(u**2 + v**2)
    # Pressure
    p = interp2d_batch(d2x, d2y, d2pressure, d2curve)
    ### Spatial gradients of first and second order
    ddx = d2x[1, 1] - d2x[0, 0]
    ddy = d2y[1, 1] - d2y[0, 0]
    d2dudx, d2dudy = np.gradient(d2u, ddx, ddy)
    d2dvdx, d2dvdy = np.gradient(d2v, ddx, ddy)
    dudx = interp2d_batch(d2x, d2y, d2dudx, d2curve)
    dudy = interp2d_batch(d2x, d2y, d2dudy, d2curve)
    dvdx = interp2d_batch(d2x, d2y, d2dvdx, d2curve)
    dvdy = interp2d_batch(d2x, d2y, d2dvdy, d2curve)
    # Tensors
    Txx = -mu * (2 * dudx)
    Txy = -mu * (dudy + dvdx)
    Tyy = -mu * (2 * dvdy)

    # Mean convection term (f1)
    # for var in [n1, n2, n3, u, v, w, p, Txx, Txy, Tyy, dudx, dudy, dvdx, dvdy]:
    # print(
    # f"mean: {np.mean(var):.2f}, min: {np.min(var):.2f}, max: {np.max(var):.2f}"
    # )

    f1 = np.array(
        [
            -n1 * (u**2) - n2 * u * v - n3 * u * w,
            -n1 * u * v - n2 * (v**2) - n3 * v * w,
            -n1 * u * w - n2 * v * w - n3 * (w**2),
        ]
    )

    # Pressure term (f2)
    f2 = np.array([-n1 * p, -n2 * p, -n3 * p])

    # Stress tensor term (f4) - 2D case
    f4 = np.array([-n1 * Txx - n2 * Txy, -n1 * Txy - n2 * Tyy, np.zeros_like(Txy)])

    # Viscous stress term (f5) - 2D case
    f5 = np.array(
        [
            n1 * (2 * dudx) + n2 * (dudy + dvdx),
            n1 * (dvdx + dudy) + n2 * (2 * dvdy),
            np.zeros_like(dvdy),
        ]
    )
    # Calculate total force
    force = (f1 + f4) * rho + f2 + f5 * mu

    f1_mean_convection = np.sum(f1, axis=1) * rho
    f2_pressure = np.sum(f2, axis=1)
    f3_unsteady = np.zeros_like(f1_mean_convection)
    f4_turbulent_moment_transfer = np.sum(f4, axis=1) * rho
    f5_viscous_stress_term = np.sum(f5, axis=1) * mu
    ftotal = (
        f1_mean_convection
        + f2_pressure
        + f3_unsteady
        + f4_turbulent_moment_transfer
        + f5_viscous_stress_term
    )
    # print(f"f1_mean_convection    : {f1_mean_convection}")
    # print(f"f2_pressure : {f2_pressure}")
    # print(f"f3_unsteady : {f3_unsteady}")
    # print(f"f4_turbulent_moment_transfer: {f4_turbulent_moment_transfer}")
    # print(f"f5_viscous_stress_term  : {f5_viscous_stress_term}")
    # print(f"Total force : {ftotal}")

    force_components = np.vstack(
        [
            f1_mean_convection,
            f2_pressure,
            f3_unsteady,
            f4_turbulent_moment_transfer,
            f5_viscous_stress_term,
        ]
    )

    return force, force_components


def extracting_fx_fy_cl_cd(
    alpha: int,
    y_num: int,
    is_ellipse: bool,
    Rey: float = 1e6,
    rho: float = 1.2,
    U_inf: float = 15,
    spatial_scale: float = 2.584,
    velocity_scale: float = 1,
    is_with_NOCA: bool = False,
):

    input_path = (
        Path(project_dir) / "data" / "CFD_slices" / f"alpha_{alpha}" / f"Y{y_num}_1.csv"
    )
    output_path = None
    interpolated_df = process_csv(
        input_path, output_path, spatial_scale, velocity_scale, y_num, alpha
    )
    df_1D = interpolated_df.copy()

    # Reading in the airfoil centers
    x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
        alpha, y_num, is_with_chord=True
    )
    drot = 0
    d1centre = (x_airfoil, y_airfoil)
    dLx, dLy, iP = reading_optimal_bound_placement(
        alpha, y_num, is_with_N_datapoints=True
    )
    iP = 800
    dLx = dLx * 1.5
    dLy = dLy * 1.5
    if is_ellipse:
        d2curve = boundary_ellipse(d1centre, drot, dLx, dLy, iP)
    else:
        d2curve = boundary_rectangle(d1centre, drot, dLx, dLy, iP)

    # computing mu
    mu = (rho * U_inf * chord) / Rey

    force, force_components = calculate_force_guanqun(
        df_1D,
        d2curve,
        mu=mu,
        rho=rho,
    )
    total_force = np.sum(force, axis=1)  # /100
    Fx = total_force[0]
    Fy = total_force[1]
    # TODO: Should check other calculation method, maybe clean up NOCA? Inside the gamma_circulation
    # To figure out how to scale this thing properly...
    q_infc = 0.5 * rho * U_inf**2 * chord  # * 1.2
    # q_infc = 0.5 * 1.2 * 15**2 * chord
    # print(f"q_infc: {q_infc}")
    Cl = Fy / q_infc
    Cd = Fx / q_infc

    if not is_with_NOCA:
        print(f"\n alpha: {alpha}, y_num: {y_num} (pressure bound integration)")
        print(f"Fx: {Fx:.3f} N")
        print(f"Fy: {Fy:.3f} N")
        print(f"Cl: {Cl:.3f} N")
        print(f"Cd: {Cd:.3f} N")
    else:
        # Continue with your existing NOCA calculation
        spatial_scale = 2.584
        velocity_scale = 15
        input_path = (
            Path(project_dir)
            / "data"
            / "CFD_slices"
            / f"alpha_{alpha}"
            / f"Y{y_num}_1.csv"
        )
        output_path = None
        interpolated_df = process_csv(
            input_path, output_path, spatial_scale, velocity_scale, y_num, alpha
        )
        df_1D = interpolated_df.copy()

        # Reading in the airfoil centers
        x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
            alpha, y_num, is_with_chord=True
        )
        drot = 0
        d1centre = (x_airfoil, y_airfoil)
        dLx, dLy, iP = reading_optimal_bound_placement(
            alpha, y_num, is_with_N_datapoints=True
        )
        if is_ellipse:
            d2curve = boundary_ellipse(d1centre, drot, dLx, dLy, iP)
        else:
            d2curve = boundary_rectangle(d1centre, drot, dLx, dLy, iP)

        # computing mu
        mu = (rho * U_inf * chord) / 1e6
        from force_from_noca import forceFromVelNoca2D_V3

        n_rows = len(np.unique(df_1D["y"].values))
        n_cols = len(np.unique(df_1D["x"].values))
        d2x = reshape_remove_nans(df_1D["x"], n_rows, n_cols)
        d2y = reshape_remove_nans(df_1D["y"], n_rows, n_cols)
        d2u = reshape_remove_nans(df_1D["u"], n_rows, n_cols)
        d2v = reshape_remove_nans(df_1D["v"], n_rows, n_cols)
        d2vort_z = reshape_remove_nans(df_1D["vort_z"], n_rows, n_cols)
        d1Fn, d1Ft = forceFromVelNoca2D_V3(
            d2x=d2x,
            d2y=d2y,
            d2u=d2u,
            d2v=d2v,
            d2vortZ=d2vort_z,
            d2dudt=np.zeros_like(d2x),  # zero for steady flow
            d2dvdt=np.zeros_like(d2x),  # zero for steady flow
            d2curve=d2curve,
            dmu=mu,
            bcorMaxVort=True,
        )

        print(f"\n alpha: {alpha}, y_num: {y_num} (NOCA)")
        print(f"Fx: {Fx:.3f} N --- NOCA  {d1Fn[0]:.3f}N      ")
        print(f"Fy: {Fy:.3f} N --- NOCA  {d1Ft[0]:.3f}N      ")
        print(f"Cl: {Cl:.3f} N --- NOCA  {d1Ft[0]/q_infc:.3f}")
        print(f"Cd: {Cd:.3f} N --- NOCA  {d1Fn[0]/q_infc:.3f}")

    return Fx, Fy, Cl, Cd


def main():
    # alpha = 6
    # for y_num in [1, 2, 3, 4, 5]:
    #     Fx, Fy, Cl, Cd = extracting_fx_fy_cl_cd(alpha, y_num, is_ellipse=True)

    # alpha = 16
    # for y_num in [1]:
    #     Fx, Fy, Cl, Cd = extracting_fx_fy_cl_cd(alpha, y_num, is_ellipse=True)

    Fx, Fy, Cl, Cd = extracting_fx_fy_cl_cd(6, 1, is_ellipse=True)
    Fx, Fy, Cl, Cd = extracting_fx_fy_cl_cd(6, 1, is_ellipse=False)

    # Fx, Fy, Cl, Cd = extracting_fx_fy_cl_cd(6, 3, is_ellipse=True)
    # Fx, Fy, Cl, Cd = extracting_fx_fy_cl_cd(6, 5, is_ellipse=True)


if __name__ == "__main__":
    main()
