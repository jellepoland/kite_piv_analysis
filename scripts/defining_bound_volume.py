import numpy as np


def iP_checker(iP):
    if (iP - 9) % 6 == 0:
        iP += 1
        print(
            f"---> ERROR iP should not be in sequence 9,15,21,27 etc. (9 + 6n) sequence."
        )
        print(f"---> FIXED by setting iP from {iP-1} to {iP}")
    return iP


def boundary_ellipse(
    d1centre: np.ndarray, drot: float, dLx: float, dLy: float, iP: int
) -> np.ndarray:
    """
    Creates an elliptical boundary curve with a specified rotation and center point.

    Parameters:
    - d1centre (np.ndarray): A vector representing the (x, y) center coordinates of the ellipse.
    - drot (float): Rotation angle in degrees; 0 corresponds to no rotation.
    - dLx (float): Length of the ellipse along the x-axis (major axis).
    - dLy (float): Length of the ellipse along the y-axis (minor axis).
    - iP (int): Number of points used to define the ellipse boundary; higher values increase resolution.

    Returns:
    - np.ndarray: A matrix of rotated (x, y) coordinates for points on the ellipse boundary, with shape (iP, 2).

    Notes:
    - The rotation is applied around the ellipse center `d1centre`.
    - For sensitivity analysis, vary `iP` to find an optimal resolution for a given application.
    """

    # Generate evenly spaced angles around the ellipse
    d1theta = np.linspace(0, 2 * np.pi, iP)

    # Calculate x, y coordinates of points before rotation (ellipse centered at origin)
    d1x = (dLx / 2) * np.cos(d1theta)  # Semi-major axis scaling for x
    d1y = (dLy / 2) * np.sin(d1theta)  # Semi-minor axis scaling for y

    # Initialize rotated curve matrix
    d2curve_rot = np.zeros((iP, 2))
    # Apply rotation and translate to center point
    d2curve_rot[:, 0] = (
        d1x * np.cos(np.radians(-drot)) + d1y * np.sin(np.radians(-drot)) + d1centre[0]
    )
    d2curve_rot[:, 1] = (
        d1y * np.cos(np.radians(-drot)) - d1x * np.sin(np.radians(-drot)) + d1centre[1]
    )

    return d2curve_rot


def boundary_rectangle(
    d1centre: np.ndarray, drot: float, dLx: float, dLy: float, iP: int
) -> np.ndarray:
    """
    Creates a rectangular boundary curve with evenly distributed points, a specified rotation, and a center point.

    Parameters:
    - d1centre (np.ndarray): A vector representing the (x, y) center coordinates of the rectangle.
    - drot (float): Rotation angle in degrees; 0 corresponds to no rotation.
    - dLx (float): Total length of the rectangle along the x-axis.
    - dLy (float): Total width of the rectangle along the y-axis.
    - iP (int): Total number of points to evenly distribute around the rectangle boundary.

    Returns:
    - np.ndarray: A matrix of rotated (x, y) coordinates for points on the rectangle boundary.

    Notes:
    - The rotation is applied around the rectangle center `d1centre`.
    - The `iP` parameter determines the density of points along each side of the rectangle.
    """
    iP = iP - 1
    iP = iP_checker(iP)

    # Defining total length
    total_length = 2 * (dLx + dLy)

    # Calculating the number of points per side
    n_points_lower_horizontal = int(np.round(iP * dLx / total_length))
    n_points_right_vertical = int(np.round(iP * dLy / total_length))
    n_points_upper_horizontal = int(np.round(iP * dLx / total_length))
    n_points_left_vertical = int(np.round(iP * dLy / total_length))
    print(f"---> n_points_lower_horizontal: {n_points_lower_horizontal}")
    print(f"---> n_points_right_vertical: {n_points_right_vertical}")
    print(f"---> n_points_upper_horizontal: {n_points_upper_horizontal}")
    print(f"---> n_points_left_vertical: {n_points_left_vertical}")

    # Initialize an array to hold the boundary points
    d2curve = np.zeros((iP, 2))

    # Define the corner points
    bottom_left = np.array([-0.5 * dLx, -0.5 * dLy]) + d1centre
    bottom_right = np.array([0.5 * dLx, -0.5 * dLy]) + d1centre
    top_right = np.array([0.5 * dLx, 0.5 * dLy]) + d1centre
    top_left = np.array([-0.5 * dLx, 0.5 * dLy]) + d1centre

    # Populate the boundary points in clockwise order
    index = 0

    # Lower horizontal edge
    for i in range(n_points_lower_horizontal):
        d2curve[index] = (
            bottom_left + i * (bottom_right - bottom_left) / n_points_lower_horizontal
        )
        index += 1

    # Right vertical edge
    for i in range(n_points_right_vertical):
        d2curve[index] = (
            bottom_right + i * (top_right - bottom_right) / n_points_right_vertical
        )
        index += 1

    # Upper horizontal edge
    for i in range(n_points_upper_horizontal):
        d2curve[index] = (
            top_right + i * (top_left - top_right) / n_points_upper_horizontal
        )
        index += 1

    # Left vertical edge
    for i in range(n_points_left_vertical):
        d2curve[index] = (
            top_left + i * (bottom_left - top_left) / n_points_left_vertical
        )
        index += 1

    # Append the bottom_left starting point to d2curve
    d2curve = np.concatenate((d2curve, [d2curve[0]]), axis=0)

    # Rotate points around the center
    cos_rot = np.cos(np.radians(drot))
    sin_rot = np.sin(np.radians(drot))

    # Apply rotation and translation to each point
    for i in range(iP):
        x_rotated = d2curve[i, 0] * cos_rot - d2curve[i, 1] * sin_rot
        y_rotated = d2curve[i, 0] * sin_rot + d2curve[i, 1] * cos_rot
        d2curve[i] = [x_rotated, y_rotated]

    return d2curve
