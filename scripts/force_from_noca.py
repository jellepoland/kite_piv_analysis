import numpy as np
from pathlib import Path
import pandas as pd
from scipy.signal import convolve2d
from utils import project_dir, reshape_remove_nans, interp2d_batch, csv_reader
from defining_bound_volume import boundary_ellipse, boundary_rectangle


def matlab_values():
    d2curveE_matlab = np.array(
        [
            [0.6700, 0.1300],
            [0.6632, 0.1667],
            [0.6430, 0.2022],
            [0.6101, 0.2353],
            [0.5656, 0.2647],
            [0.5111, 0.2896],
            [0.4483, 0.3090],
            [0.3795, 0.3224],
            [0.3069, 0.3291],
            [0.2331, 0.3291],
            [0.1605, 0.3224],
            [0.0917, 0.3090],
            [0.0289, 0.2896],
            [-0.0256, 0.2647],
            [-0.0701, 0.2353],
            [-0.1030, 0.2022],
            [-0.1232, 0.1667],
            [-0.1300, 0.1300],
            [-0.1232, 0.0933],
            [-0.1030, 0.0578],
            [-0.0701, 0.0247],
            [-0.0256, -0.0047],
            [0.0289, -0.0296],
            [0.0917, -0.0490],
            [0.1605, -0.0624],
            [0.2331, -0.0691],
            [0.3069, -0.0691],
            [0.3795, -0.0624],
            [0.4483, -0.0490],
            [0.5111, -0.0296],
            [0.5656, -0.0047],
            [0.6101, 0.0247],
            [0.6430, 0.0578],
            [0.6632, 0.0933],
            [0.6700, 0.1300],
        ]
    )
    d1nx_matlab = [
        0.983256978092441,
        0.936678299016319,
        0.790490015194276,
        0.628259612424673,
        0.480891102992294,
        0.35324066911471,
        0.241595196911805,
        0.14084416368581,
        0.0462821742181829,
        -0.0462821742181829,
        -0.140844163685809,
        -0.241595196911805,
        -0.35324066911471,
        -0.480891102992294,
        -0.628259612424672,
        -0.790490015194276,
        -0.936678299016319,
        -1,
        -0.93667829901632,
        -0.790490015194276,
        -0.628259612424672,
        -0.480891102992294,
        -0.35324066911471,
        -0.241595196911805,
        -0.14084416368581,
        -0.0462821742181826,
        0.0462821742181829,
        0.14084416368581,
        0.241595196911805,
        0.35324066911471,
        0.480891102992294,
        0.628259612424672,
        0.790490015194276,
        0.936678299016319,
        0.983256978092441,
    ]

    d1ny_matlab = [
        0.182224353565929,
        0.350191039508287,
        0.612474926734273,
        0.77800376566955,
        0.876780329993126,
        0.935532484568757,
        0.970377122993502,
        0.990031778053435,
        0.998928406017988,
        0.998928406017988,
        0.990031778053434,
        0.970377122993502,
        0.935532484568757,
        0.876780329993126,
        0.778003765669551,
        0.612474926734273,
        0.350191039508286,
        0,
        -0.350191039508286,
        -0.612474926734273,
        -0.77800376566955,
        -0.876780329993126,
        -0.935532484568757,
        -0.970377122993502,
        -0.990031778053434,
        -0.998928406017988,
        -0.998928406017988,
        -0.990031778053434,
        -0.970377122993502,
        -0.935532484568757,
        -0.876780329993126,
        -0.77800376566955,
        -0.612474926734273,
        -0.350191039508287,
        -0.182224353565929,
    ]

    d1u_matlab = [
        14.664519390607,
        14.772026587833,
        14.8584725136609,
        14.9222811949487,
        15.0281428687206,
        15.1583393509451,
        15.3429284499789,
        15.5955033708681,
        15.9177954844929,
        16.2945171599361,
        16.6478968676699,
        16.7200655780401,
        16.2883528994485,
        15.5469068576864,
        14.9709264279821,
        14.5917739434444,
        14.3863158587085,
        14.2523277839718,
        14.1200709533098,
        13.9420202113551,
        13.8312380083358,
        13.7303840657451,
        13.7488884733618,
        13.7895515132918,
        13.7838650081498,
        13.7572654476715,
        13.8116869711231,
        13.9696884150194,
        14.1899781539936,
        14.3662254846481,
        14.4759352410923,
        13.4666879566002,
        11.0885273211623,
        13.9917358676915,
        14.664519390607,
    ]

    d1v_matlab = [
        -1.75062175829421,
        -1.60749343535846,
        -1.52227127968833,
        -1.48251332604511,
        -1.45681984789922,
        -1.46307384064901,
        -1.45634578633718,
        -1.42384237325073,
        -1.28317338157099,
        -0.989817786644767,
        -0.469638259601709,
        0.357660146963551,
        1.13253222704574,
        1.43500022438579,
        1.30817540421522,
        1.10967111731811,
        0.92915417756054,
        0.780466679599771,
        0.622878531078654,
        0.455412696665699,
        0.237478038605629,
        -0.0545490657742351,
        -0.306262987094012,
        -0.476403512329325,
        -0.635696569732414,
        -0.884176729324467,
        -1.19214534926634,
        -1.4692525523212,
        -1.6166652577829,
        -1.72096041659617,
        -1.80883294255057,
        -1.87739842783061,
        -1.58601521700169,
        -1.87785825369569,
        -1.75062175829421,
    ]

    d1vortZ_matlab = [
        -0.032684067056061,
        -0.46273756492169,
        0.0832167039830301,
        -0.0388644968567874,
        -0.00754415801268366,
        0.0597647328114458,
        -0.103879834721507,
        0.0874794899985143,
        -0.0768534754201737,
        0.0613164666779975,
        -0.0394017370252931,
        -0.6510222173146,
        -0.370715760244217,
        0.211976370114885,
        0.0344261721436373,
        -0.472355046079524,
        0.0028493456161155,
        0.398816964553262,
        0.126656315877562,
        -0.0297831488622776,
        0.175155162611452,
        -0.549910811908851,
        0.168548778360207,
        0.00299861596797332,
        0.0573928118851489,
        -0.0173627541323162,
        0.0975298675291106,
        0.132457882712776,
        -0.0700593385324978,
        -0.140850866041277,
        0.251963081980278,
        25.1485970516756,
        6.85364748521887,
        -22.7165983019306,
        -0.0326840670560575,
    ]

    d1s_matlab = [
        0,
        0.0373756854841858,
        0.0782191987166055,
        0.124846015866618,
        0.178196031508229,
        0.2381452213467,
        0.303842444798866,
        0.373951957728044,
        0.446826050977507,
        0.520640738548148,
        0.593514831797611,
        0.663624344726788,
        0.729321568178955,
        0.789270758017426,
        0.842620773659036,
        0.889247590809049,
        0.930091104041469,
        0.967466789525655,
        1.00484247500984,
        1.04568598824226,
        1.09231280539227,
        1.14566282103388,
        1.20561201087235,
        1.27130923432452,
        1.3414187472537,
        1.41429284050316,
        1.4881075280738,
        1.56098162132327,
        1.63109113425244,
        1.69678835770461,
        1.75673754754308,
        1.81008756318469,
        1.8567143803347,
        1.89755789356712,
        1.93493357905131,
    ]

    d2curve_matlab = [
        [0.0270111082382577, 0.0722483332374306],
        [0.0202003481118185, 0.108998236800745],
        [0, 0.0998501279656955],
        [-0.0241073894814963, 0.0565384404241504],
        [-0.0337265168017037, 0.0327879563352816],
        [-0.0331077797882652, 0.0180385282767437],
        [-0.0181857213695635, 0.0114430201128982],
        [0, 0.0169935618553003],
    ]
    return (
        d2curveE_matlab,
        d1nx_matlab,
        d1ny_matlab,
        d1u_matlab,
        d1v_matlab,
        d1vortZ_matlab,
        d1s_matlab,
        d2curve_matlab,
    )


def circshift(array, shift):
    """
    Circularly shift the elements of an array similar to MATLAB's circshift.

    Parameters:
    - array: np.ndarray
        The input array to be shifted.
    - shift: int
        The number of positions to shift. Positive values shift to the right,
        and negative values shift to the left.

    Returns:
    - np.ndarray: The circularly shifted array.
    """
    return np.roll(array, shift)


def conv2(x, y, mode="same"):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def smooth_data(data, ismooth):
    kernel = np.ones((ismooth, ismooth)) / (ismooth**2)
    return conv2(data, kernel, mode="same")


def forceFromVelNoca2D_V3(
    d2x, d2y, d2u, d2v, d2vortZ, d2dudt, d2dvdt, d2curve, dmu, bcorMaxVort
):
    # # extract matlab values
    # (
    #     d2curveE_matlab,
    #     d1nx_matlab,
    #     d1ny_matlab,
    #     d1u_matlab,
    #     d1v_matlab,
    #     d1vortZ_matlab,
    #     d1s_matlab,
    #     d2curve_matlab,
    # ) = matlab_values()
    # print(f"\n--- COMPARING MATLAB VALUES ---")
    # print(f"DELTA d2curve: {np.max(d2curve- d2curveE_matlab)}")

    # Initial value
    iN = 2

    # Data smoothing
    bsmooth = False
    ismooth = 9
    if bsmooth:
        d2u = conv2(d2u, np.ones((ismooth, ismooth)) / (ismooth**2), mode="same")
        d2v = conv2(d2v, np.ones((ismooth, ismooth)) / (ismooth**2), mode="same")
        d2vortZ = conv2(
            d2vortZ, np.ones((ismooth, ismooth)) / (ismooth**2), mode="same"
        )
        d2dudt = conv2(d2dudt, np.ones((ismooth, ismooth)) / (ismooth**2), mode="same")
        d2dvdt = conv2(d2dvdt, np.ones((ismooth, ismooth)) / (ismooth**2), mode="same")

    # Curve coordinate
    d1_s_concatenate = np.concatenate(
        ([0], np.sqrt(np.sum((d2curve[1:] - d2curve[:-1]) ** 2, axis=1)))
    )
    d1s = np.cumsum(d1_s_concatenate, axis=0)

    # Ensure inputs are NumPy arrays
    d2curve = np.asarray(d2curve)
    # Normal vector calculation - MATLAB style
    d1ny = -np.gradient(d2curve[:, 0]) / np.sqrt(
        np.gradient(d2curve[:, 0]) ** 2 + np.gradient(d2curve[:, 1]) ** 2
    )
    d1nx = np.gradient(d2curve[:, 1]) / np.sqrt(
        np.gradient(d2curve[:, 0]) ** 2 + np.gradient(d2curve[:, 1]) ** 2
    )
    # Spatial gradients of first and second order
    ddx = d2x[1, 1] - d2x[0, 0]
    ddy = d2y[1, 1] - d2y[0, 0]

    d2dudx, d2dudy = np.gradient(d2u, ddx, ddy)
    d2dvdx, d2dvdy = np.gradient(d2v, ddx, ddy)

    d2d2udx2, d2d2udydx = np.gradient(d2dudx, ddx, ddy)
    d2d2vdx2, d2d2vdydx = np.gradient(d2dvdx, ddx, ddy)
    d2d2udxdy, d2d2udy2 = np.gradient(d2dudy, ddx, ddy)
    d2d2vdxdy, d2d2vdy2 = np.gradient(d2dvdy, ddx, ddy)

    # # Vector fields interpolated along curve
    # print(f"DELTA d1nx: {np.max(d1nx- d1nx_matlab)}")
    # print(f"DELTA d1ny: {np.max(d1ny- d1ny_matlab)}")
    # Note: You'll need to implement interp2 equivalent function
    d1u = interp2d_batch(d2x, d2y, d2u, d2curve)
    d1v = interp2d_batch(d2x, d2y, d2v, d2curve)
    d1vortZ = interp2d_batch(d2x, d2y, d2vortZ, d2curve)

    d1dudt = interp2d_batch(d2x, d2y, d2dudt, d2curve)
    d1dvdt = interp2d_batch(d2x, d2y, d2dvdt, d2curve)

    d1dudx = interp2d_batch(d2x, d2y, d2dudx, d2curve)
    d1dudy = interp2d_batch(d2x, d2y, d2dudy, d2curve)
    d1dvdx = interp2d_batch(d2x, d2y, d2dvdx, d2curve)
    d1dvdy = interp2d_batch(d2x, d2y, d2dvdy, d2curve)

    d1d2udx2 = interp2d_batch(d2x, d2y, d2d2udx2, d2curve)
    # d1d2udydx = interp2d_batch(d2x, d2y, d2d2udydx, d2curve)
    d1d2vdx2 = interp2d_batch(d2x, d2y, d2d2vdx2, d2curve)
    # d1d2vdydx = interp2d_batch(d2x, d2y, d2d2vdydx, d2curve)
    d1d2udxdy = interp2d_batch(d2x, d2y, d2d2udxdy, d2curve)
    d1d2udy2 = interp2d_batch(d2x, d2y, d2d2udy2, d2curve)
    d1d2vdxdy = interp2d_batch(d2x, d2y, d2d2vdxdy, d2curve)
    d1d2vdy2 = interp2d_batch(d2x, d2y, d2d2vdy2, d2curve)

    # Test: Change coord frame to minimise impact of vorticity term
    if bcorMaxVort:
        # Combine current and shifted vorticity values
        vort_combined = np.abs(
            d1vortZ
            + circshift(d1vortZ, 1)
            + circshift(d1vortZ, 2)
            + circshift(d1vortZ, 3)
            + circshift(d1vortZ, 4)
        )
        imaxVortZ = np.argmax(vort_combined)
        d2curve = d2curve - d2curve[imaxVortZ, :]

    # print(f"\n--- COMPARING ---")
    # print(f"DELTA d1u: {np.max(d1u- d1u_matlab)}")
    # print(f"DELTA d1v: {np.max(d1v- d1v_matlab)}")
    # print(f"DELTA d1vortZ: {np.max(d1vortZ- d1vortZ_matlab)}")
    # print(f"DELTA d1s: {np.max(d1s- d1s_matlab)}")
    # print(f"DELTA d2curve: {np.max(d2curve- d2curve_matlab)}")
    ### NORMAL FORCE CALCULATION ###
    # Calculate the various normal force contributions
    d2Fn = np.zeros((len(d1s), 11))

    ## Inviscid terms
    d2Fn[:, 1] = 0.5 * d1nx * (d1u**2 + d1v**2)
    # Convective term
    d2Fn[:, 2] = -(d1nx * d1u**2 + d1ny * d1v * d1u)
    # Rotational correction term
    d2Fn[:, 3] = (
        -1
        / (iN - 1)
        * (d1nx * d1u * d2curve[:, 1] * d1vortZ + d1ny * d1v * d2curve[:, 1] * d1vortZ)
    )
    d2Fn[:, 4] = 0

    ## Time dependent terms
    d2Fn[:, 5] = (
        -1 / (iN - 1) * d1nx * (d2curve[:, 0] * d1dudt + d2curve[:, 1] * d1dvdt)
    )
    d2Fn[:, 6] = (
        1 / (iN - 1) * (d1nx * d2curve[:, 0] * d1dudt + d1ny * d2curve[:, 1] * d1dudt)
    )
    d2Fn[:, 7] = -(d1nx * d1dudt * d2curve[:, 0] + d1ny * d1dvdt * d2curve[:, 0])

    ## Viscous terms
    d1nablaTau1 = 2 * d1d2udx2 + d1d2vdxdy + d1d2udy2
    d1nablaTau2 = d1d2udxdy + d1d2vdx2 + 2 * d1d2vdy2

    d2Fn[:, 8] = (
        1
        / (iN - 1)
        * dmu
        * (d1nx * (d2curve[:, 0] * d1nablaTau1 + d2curve[:, 1] * d1nablaTau2))
    )
    d2Fn[:, 9] = (
        -1
        / (iN - 1)
        * dmu
        * (d1nx * d2curve[:, 0] * d1nablaTau1 + d1ny * d2curve[:, 1] * d1nablaTau1)
    )
    d2Fn[:, 10] = dmu * (d1nx * 2 * d1dudx + d1ny * (d1dvdx + d1dudy))

    # Ensure total force is calculated in the first column of d2Fn
    d2Fn[:, 0] = np.sum(d2Fn[:, 1:], axis=1)

    # Integrate each column across rows (axis=0), producing 11 values
    d1Fn = np.trapz(d2Fn, d1s, axis=0)

    ### TANGENTIAL FORCE CALCULATION ###

    # Calculate the various tangential force contributions
    d2Ft = np.zeros((len(d1s), 11))

    # Inviscid terms
    d2Ft[:, 1] = 0.5 * d1ny * (d1u**2 + d1v**2)
    d2Ft[:, 2] = -(d1nx * d1u * d1v + d1ny * d1v**2)
    d2Ft[:, 3] = (
        -1
        / (iN - 1)
        * (-d1nx * d1u * d2curve[:, 0] * d1vortZ - d1ny * d1v * d2curve[:, 0] * d1vortZ)
    )
    d2Ft[:, 4] = 0

    # Time dependent terms
    d2Ft[:, 5] = (
        -1 / (iN - 1) * d1ny * (d2curve[:, 0] * d1dudt + d2curve[:, 1] * d1dvdt)
    )
    d2Ft[:, 6] = (
        1 / (iN - 1) * (d1nx * d2curve[:, 0] * d1dvdt + d1ny * d2curve[:, 1] * d1dvdt)
    )
    d2Ft[:, 7] = -(d1nx * d1dudt * d2curve[:, 1] + d1ny * d1dvdt * d2curve[:, 1])

    # Viscous terms
    d2Ft[:, 8] = (
        1
        / (iN - 1)
        * dmu
        * (
            d1ny
            * (
                d2curve[:, 0] * (2 * d1d2udx2 + d1d2vdxdy + d1d2udy2)
                + d2curve[:, 1] * (2 * d1d2udxdy + d1d2vdx2 + d1d2vdy2)
            )
        )
    )
    d2Ft[:, 9] = (
        -1
        / (iN - 1)
        * dmu
        * (d1nx * d2curve[:, 0] * d1nablaTau2 + d1ny * d2curve[:, 1] * d1nablaTau2)
    )
    d2Ft[:, 10] = dmu * (d1nx * (d1dudy + d1dvdx) + d1ny * 2 * d1dvdy)

    # Ensure total force is calculated in the first column of d2Ft
    d2Ft[:, 0] = np.sum(d2Ft[:, 1:], axis=1)

    # Integrate each column across rows (axis=0), producing 11 values
    d1Ft = np.trapz(d2Ft, d1s, axis=0)

    # # printing out all the components, with labels
    # print(f"\nF_X Normal force components:")
    # print(f"Inviscid terms [1,2,3] {d1Fn[1], d1Fn[2], d1Fn[3]}")
    # print(f"Time dependent terms [5,6,7] {d1Fn[5], d1Fn[6], d1Fn[7]}")
    # print(f"Viscous terms [8,9,10] {d1Fn[8], d1Fn[9], d1Fn[10]}")
    # print(f"------------+ {d1Fn[0]}")
    # print(f"\nF_Y Tangential force components:")
    # print(f"Inviscid terms [1,2,3] {d1Ft[1], d1Ft[2], d1Ft[3]}")
    # print(f"Time dependent terms [5,6,7] {d1Ft[5], d1Ft[6], d1Ft[7]}")
    # print(f"Viscous terms [8,9,10] {d1Ft[8], d1Ft[9], d1Ft[10]}")
    # print(f"------------+ {d1Ft[0]}")

    return d1Fn, d1Ft


def main(
    df_1D: pd.DataFrame,
    d2curve: np.ndarray,
    mu: float = 1.7894e-5,
    is_with_maximim_vorticity_location_correction: bool = True,
    rho: float = 1.225,
    U_inf: float = 15,
    c: float = 0.37,
):
    # print(f"\nRunning NOCA calculating F_x, F_y, C_l, C_d")
    # reshape df
    d2x = df_1D["x"].values
    d2y = df_1D["y"].values
    n_rows = len(np.unique(d2y))
    n_cols = len(np.unique(d2x))

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
        d2dudt=np.zeros_like(d2x),
        d2dvdt=np.zeros_like(d2x),
        d2curve=d2curve,
        dmu=mu,
        bcorMaxVort=is_with_maximim_vorticity_location_correction,
    )
    F_x = d1Fn[0]
    F_y = d1Ft[0]
    C_l = F_y / (0.5 * rho * U_inf**2 * c)
    C_d = F_x / (0.5 * rho * U_inf**2 * c)
    # print(f"F_x = {d1Fn[0]:.3f}N (F_n Normal force)")
    # print(f"F_y = {d1Ft[0]:.3f}N (F_t Tangential force)")
    # print(f"C_l = {C_l:.3f} (C_l Lift coefficient)")
    # print(f"C_d = {C_d:.3f} (C_d Drag coefficient)")

    return F_x, F_y, C_l, C_d


if __name__ == "__main__":

    df_1D = csv_reader(is_CFD=True, alpha=6, y_num=1, alpha_d_rod=7.25)

    ### Running for Ellipse ###
    is_ellipse = True
    d1centre = np.array([0.27, 0.13])
    drot = 0
    dLx = 0.8
    dLy = 0.4
    iP = 35

    # create d2curve
    if is_ellipse:
        d2curve = boundary_ellipse(d1centre, drot, dLx, dLy, iP)
        # print(f"Running NOCA on Ellipse, will take a while...")
    else:
        d2curve = boundary_rectangle(d1centre, drot, dLx, dLy, iP)
        # print(f"Running NOCA on Rectangle, will take a while...")

    main(
        df_1D,
        d2curve,
        mu=1.7894e-5,
        is_with_maximim_vorticity_location_correction=True,
    )

    ### Running for Rectangle ###
    is_ellipse = False
    d1centre = np.array([0.27, 0.13])
    drot = 0
    dLx = 0.8
    dLy = 0.4
    iP = 35

    # create d2curve
    if is_ellipse:
        d2curve = boundary_ellipse(d1centre, drot, dLx, dLy, iP)
        # print(f"Running NOCA on Ellipse, will take a while...")
    else:
        d2curve = boundary_rectangle(d1centre, drot, dLx, dLy, iP)
        # print(f"Running NOCA on Rectangle, will take a while...")

    main(
        df_1D,
        d2curve,
        mu=1.7894e-5,
        is_with_maximim_vorticity_location_correction=True,
    )
