import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from pathlib import Path
import pandas as pd
import os
from typing import TypedDict, Optional
from utils import project_dir
from io import StringIO
from defining_bound_volume import boundary_ellipse, boundary_rectangle
import force_from_noca
from calculating_circulation import calculate_circulation
from typing import Tuple, List, Union
from plotting import *
from scipy.spatial import ConvexHull


def transform_raw_csv_to_processed_df(
    alpha=6, spatial_scale=2.584, cm_offset=25
) -> pd.DataFrame:

    file_path = (
        Path(project_dir)
        / "data"
        / "CFD_slices"
        / "spanwise_slices"
        / f"alpha_{alpha}_CFD_spanwise_slice_{cm_offset}cm_1.csv"
    )

    # Load the raw data
    df = pd.read_csv(file_path)

    ### new
    # Transform the headers
    header_mapping = {
        "Points:0": "z",
        "Points:1": "y",
        "Points:2": "x",
        "Time": "time",
        "ReThetat": "ReTheta",
        "U:0": "u",
        "U:1": "v",
        "U:2": "w",
        "gammaInt": "gamma_int",
        "grad(U):0": "dudx",
        "grad(U):1": "dudy",
        "grad(U):2": "dudz",
        "grad(U):3": "dvdx",
        "grad(U):4": "dvdy",
        "grad(U):5": "dvdz",
        "grad(U):6": "dwdx",
        "grad(U):7": "dwdy",
        "grad(U):8": "dwdz",
        "vorticity:2": "vort_z",
        "k": "tke",
        "nut": "nu_t",
        "omega": "omega",
        "p": "pressure",
        "vorticity:0": "vort_x",
        "vorticity:1": "vort_y",
        # "vorticity:2": "vorticity_z",
        "wallShearStress:0": "tau_w_x",
        "wallShearStress:1": "tau_w_y",
        "wallShearStress:2": "tau_w_z",
        "yPlus": "y_plus",
    }
    df = df.rename(columns=header_mapping)
    variable_list = [
        "x",
        "y",
        "u",
        "v",
        "w",
    ]
    df = df[variable_list]

    # Scale spatial dimensions
    # df["x"] /= spatial_scale
    # df["y"] /= spatial_scale

    # filter data
    # x_range = (0, 0.8)  # becomes y
    # y_range = (-0.6, 0.1)  # becomes z
    y_range = (-1.6, 0.5)
    x_range = (-0.2, 2.4)
    y_mask = (df["y"] >= y_range[0]) & (df["y"] <= y_range[1])
    x_mask = (df["x"] >= x_range[0]) & (df["x"] <= x_range[1])
    mask = x_mask & y_mask
    df = df[mask]

    # Store the x,y values where u, v, w are all zero
    vel_mask = (df["u"] == 0) & (df["v"] == 0) & (df["w"] == 0)
    zero_vel_df = df[vel_mask]

    points = np.array(zero_vel_df[["x", "y"]])

    # scale points
    points[:, 0] *= spatial_scale / 6.5
    points[:, 1] *= spatial_scale / 6.5

    return points


def sort_points_by_angle(
    points: Union[List[List[float]], np.ndarray],
    reference_point: Tuple[float, float] = (0, -0.42),
) -> np.ndarray:
    """
    Sort points based on their angle with respect to a reference point.

    Parameters:
    -----------
    points : array-like
        Array of points with shape (n, 2) where each point is [x, y]
    reference_point : Tuple[float, float]
        Reference point for angle calculation (default: (0, -0.42))

    Returns:
    --------
    np.ndarray
        Array of sorted points with shape (n, 4) [x, y, angle, distance]
    """
    # Convert inputs to numpy arrays
    points = np.array(points)

    # Check input dimensions

    # if points.shape[1] != 2:
    #     points = np.reshape(points, (len(points) * 2, 2))
    #     # raise ValueError("Points array must have shape (n, 2)")

    # Calculate angles and distances relative to the reference point
    dx = points[:, 0] - reference_point[0]
    dy = points[:, 1] - reference_point[1]
    angles = np.arctan2(dy, dx)
    distances = np.sqrt(dx**2 + dy**2)

    # Combine points, angles, and distances into a single array
    data = np.hstack((points, angles[:, None], distances[:, None]))

    # Sort by angles
    sorted_data = data[np.argsort(angles)]

    return sorted_data


def plot_sorted_points(
    ax: plt.Axes,
    sorted_points: np.ndarray,
    window_size: int = 100,  # Size of point window for each hull
    step_size: int = 2,  # How many points to slide forward
    is_small_plot: bool = False,
):
    """
    Plot the sorted points with sliding window convex hulls.

    Parameters:
    -----------
    sorted_points : np.ndarray
        Array of sorted points with shape (n, 4) [x, y, angle, distance]
    window_size : int
        Number of points to include in each convex hull
    step_size : int
        Number of points to slide forward for each new hull
    """

    ##Plot original points as a line
    ax.plot(
        sorted_points[:, 0],
        sorted_points[:, 1],
        color="black",
        linestyle="-",
        linewidth=0.8,
    )

    points_list = []
    for point in sorted_points:
        if point[0] > 0.6 and point[1] < -0.38:
            points_list.append(point)
    if len(points_list) > 2:
        points_list = np.array(points_list)
        ax.plot(points_list[:, 0], points_list[:, 1], color="black", linewidth=1.7)

    ax.set_aspect("equal")
    return ax


def distance_to_ref_point(point):
    return np.sqrt((point[0] - 0) ** 2 + (point[1] - -0.42) ** 2)


def plot_with_indices(points):
    # Unpack the points into x and y coordinates
    x, y = zip(*points)

    # Plot the points
    plt.plot(x, y, "o", label="Points")

    # Annotate each point with its index
    for i, (xi, yi) in enumerate(points):
        plt.text(xi, yi, str(i), fontsize=12, ha="right", va="bottom")


def dist(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def find_closest_2_points_bruteforce(point, set2):
    # Calculate distances from the given point to each point in set2
    distances = []
    for p in set2:
        dist = np.sqrt((point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2)
        distances.append((dist, p))

    # Sort the points by distance
    distances.sort(key=lambda x: x[0])

    # Return the two closest points
    return distances[0][1], distances[1][1]


def distance_from_point_to_line(P1, P2, P3):
    # Unpack the points
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = P3

    # Calculate the distance using the formula
    numerator = abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
    denominator = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Return the distance
    return numerator / denominator


def save_contour_points_to_csv(points, alpha, cm_offset):
    sorted_points = sort_points_by_angle(points)

    # loop through points
    lower_side = []
    upper_side = []

    # First loop: Populate lower_side and upper_side
    for i in np.linspace(
        0, len(sorted_points) - 2, int(len(sorted_points) / 2), endpoint=False
    ):
        i = int(i)
        if distance_to_ref_point(sorted_points[i]) < distance_to_ref_point(
            sorted_points[i + 1]
        ):
            lower_side.append(sorted_points[i])
            upper_side.append(sorted_points[i + 1])
        else:
            lower_side.append(sorted_points[i + 1])
            upper_side.append(sorted_points[i])

    plt.plot(
        np.array(upper_side)[:, 0],
        np.array(upper_side)[:, 1],
        color="black",
        linewidth=1.4,
        label="Upper Side",
    )

    # looping through the upper_side points
    unit_vector_previous = upper_side[1] - upper_side[0]
    unit_vector_previous = unit_vector_previous / np.linalg.norm(unit_vector_previous)
    indices_to_remove = []
    removed_points = []
    if alpha == 6:
        if cm_offset == 10:
            alpha_tol_upper = 5
            additional_upper_indices_to_remove = []
            distance_tol_lower = 0.001
            additional_lower_indices_to_remove = [2]
        elif cm_offset == 15:
            alpha_tol_upper = 5
            additional_upper_indices_to_remove = [81, 136, 190]
            distance_tol_lower = 0.001
            additional_lower_indices_to_remove = [211]
        elif cm_offset == 20:
            alpha_tol_upper = 5
            additional_upper_indices_to_remove = [24]
            distance_tol_lower = 0.001
            additional_lower_indices_to_remove = [4, 131, 132, 213, 215, 291]
        elif cm_offset == 25:
            alpha_tol_upper = 5
            additional_upper_indices_to_remove = []
            distance_tol_lower = 0.001
            additional_lower_indices_to_remove = [145, 146, 227, 228, 304, 316]
        elif cm_offset == 30:
            alpha_tol_upper = 5
            additional_upper_indices_to_remove = [90, 208]
            distance_tol_lower = 0.001
            additional_lower_indices_to_remove = [66, 149, 150, 228, 230, 309]

    elif alpha == 16 and cm_offset == 25:
        alpha_tol_upper = 10
        additional_upper_indices_to_remove = [31, 85, 144, 145, 207]
        distance_tol_lower = 0.001
        additional_lower_indices_to_remove = [80, 158, 159, 238, 239, 241, 324]
    else:
        alpha_tol_upper = 5
        additional_upper_indices_to_remove = []
        distance_tol_lower = 0.01
        additional_lower_indices_to_remove = []

    print(
        f"upperside, additional_indices_to_remove: {additional_upper_indices_to_remove}"
    )
    for i in range(2, len(upper_side)):
        unit_vector = upper_side[i] - upper_side[i - 1]
        unit_vector = unit_vector / np.linalg.norm(unit_vector)

        # compute angle of unit vector wrt to previous unit vector
        delta_angle = np.arccos(np.dot(unit_vector, unit_vector_previous))

        # print(f"delta_angle: {np.rad2deg(delta_angle)}")
        if np.abs(delta_angle) > np.deg2rad(alpha_tol_upper):
            indices_to_remove.append(i)
            removed_points.append(upper_side[i])

        unit_vector_previous = unit_vector

    for index in sorted(indices_to_remove, reverse=True):
        upper_side.pop(index)

    for index in sorted(additional_upper_indices_to_remove, reverse=True):
        upper_side.pop(index)

    plt.plot(
        np.array(upper_side)[:, 0],
        np.array(upper_side)[:, 1],
        color="red",
        linewidth=1.4,
        label="Upper Side",
    )

    ############
    plt.plot(
        np.array(lower_side)[:, 0],
        np.array(lower_side)[:, 1],
        color="black",
        linewidth=1.4,
    )

    upper_side = [point[:2] for point in upper_side]
    lower_side = [point[:2] for point in lower_side]
    # plot_with_indices(upper_side)

    # loop through lower points
    indices_to_remove = []

    for i in range(0, len(lower_side)):
        # print(f"lower_side[i]: {lower_side[i]}")
        p1, p2 = find_closest_2_points_bruteforce(lower_side[i], upper_side)
        # print(f"p1: {p1}, p2: {p2}")
        distance = distance_from_point_to_line(p1, p2, lower_side[i])
        print(f"distance: {distance:0.4f}")

        if distance < distance_tol_lower:
            indices_to_remove.append(i)

    for index in sorted(indices_to_remove, reverse=True):
        lower_side.pop(index)

    for index in sorted(additional_lower_indices_to_remove, reverse=True):
        lower_side.pop(index)

    plt.plot(
        np.array(lower_side)[:, 0],
        np.array(lower_side)[:, 1],
        color="red",
        linewidth=1.4,
    )
    # plot_with_indices(lower_side)
    plt.show()

    # reverse top points
    upper_side = upper_side[::-1]

    # add upper to lower
    both_sides = lower_side.copy()
    both_sides.extend(upper_side)
    both_sides.append(lower_side[0])

    # saving the points as csv
    df = pd.DataFrame(both_sides, columns=["x", "y"])
    csv_path = (
        Path(project_dir)
        / "processed_data"
        / "CFD"
        / "spanwise_slices"
        / f"alpha_{alpha}_CFD_{cm_offset}cm_outline_wing.csv"
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def main(ax, csv_path):

    # read the points
    df = pd.read_csv(csv_path)
    points = np.array(df[["x", "y"]])
    ax.fill(points[:, 0], points[:, 1], color="black", alpha=1.0)

    return ax


if __name__ == "__main__":
    # Load the raw data
    alpha = 6
    cm_offset = 30  # [10, 15, 20, 25, 30]
    points = transform_raw_csv_to_processed_df(alpha, cm_offset=cm_offset)
    csv_path = save_contour_points_to_csv(points, alpha, cm_offset)

    # fig, ax = plt.subplots()
    # main(ax, csv_path)
    # plt.show()
