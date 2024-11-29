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


def iP_checker(iP: int) -> int:
    """
    Ensure the number of points is a multiple of 4.

    Parameters:
    - iP (int): Number of points to distribute

    Returns:
    - int: Adjusted number of points to be a multiple of 4
    """
    # Ensure iP is a multiple of 4, always rounding up
    return 4 * ((iP + 3) // 4)


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
    - iP (int): Total number of points to distribute around the rectangle boundary.

    Returns:
    - np.ndarray: A matrix of rotated (x, y) coordinates for points on the rectangle boundary.
    """
    # Adjust iP to be a multiple of 4
    iP = iP_checker(iP)

    # Define the corner points
    bottom_left = np.array([-0.5 * dLx, -0.5 * dLy]) + d1centre
    bottom_right = np.array([0.5 * dLx, -0.5 * dLy]) + d1centre
    top_right = np.array([0.5 * dLx, 0.5 * dLy]) + d1centre
    top_left = np.array([-0.5 * dLx, 0.5 * dLy]) + d1centre

    # Initialize an array to hold the boundary points
    d2curve = np.zeros((iP, 2))

    # Calculate points per side (ensuring equal distribution)
    points_per_side = iP // 4

    # Distribute points along each side
    # Bottom side (left to right)
    for i in range(points_per_side):
        t = i / (points_per_side - 1) if points_per_side > 1 else 0
        d2curve[i] = bottom_left + t * (bottom_right - bottom_left)

    # Right side (bottom to top)
    for i in range(points_per_side):
        t = i / (points_per_side - 1) if points_per_side > 1 else 0
        d2curve[points_per_side + i] = bottom_right + t * (top_right - bottom_right)

    # Top side (right to left)
    for i in range(points_per_side):
        t = i / (points_per_side - 1) if points_per_side > 1 else 0
        d2curve[2 * points_per_side + i] = top_right + t * (top_left - top_right)

    # Left side (top to bottom)
    for i in range(points_per_side):
        t = i / (points_per_side - 1) if points_per_side > 1 else 0
        d2curve[3 * points_per_side + i] = top_left + t * (bottom_left - top_left)

    # Rotate points around the center
    cos_rot = np.cos(np.radians(drot))
    sin_rot = np.sin(np.radians(drot))

    # Apply rotation and translation to each point
    for i in range(iP):
        x = d2curve[i, 0]
        y = d2curve[i, 1]
        x_rotated = x * cos_rot - y * sin_rot
        y_rotated = x * sin_rot + y * cos_rot
        d2curve[i] = [x_rotated, y_rotated]

    return d2curve
