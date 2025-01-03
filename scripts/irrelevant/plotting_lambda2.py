import numpy as np
from scipy import linalg
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from utils import project_dir


def compute_lambda2(df):
    """
    Compute lambda2 criterion for vortex detection from velocity gradient tensor.
    """
    # Extract velocity gradients
    grad_U = np.zeros((len(df), 3, 3))
    grad_U[:, 0, 0] = df["grad(U):0"]
    grad_U[:, 0, 1] = df["grad(U):1"]
    grad_U[:, 0, 2] = df["grad(U):2"]
    grad_U[:, 1, 0] = df["grad(U):3"]
    grad_U[:, 1, 1] = df["grad(U):4"]
    grad_U[:, 1, 2] = df["grad(U):5"]
    grad_U[:, 2, 0] = df["grad(U):6"]
    grad_U[:, 2, 1] = df["grad(U):7"]
    grad_U[:, 2, 2] = df["grad(U):8"]

    # Initialize arrays for results
    lambda2 = np.zeros(len(df))
    Q = np.zeros(len(df))

    # Compute for each point
    for i in range(len(df)):
        # Rate of strain tensor S
        S = 0.5 * (grad_U[i] + grad_U[i].T)

        # Rate of rotation tensor Ω
        Omega = 0.5 * (grad_U[i] - grad_U[i].T)

        # Compute S² + Ω²
        S2_plus_O2 = np.matmul(S, S) + np.matmul(Omega, Omega)

        # Get eigenvalues (sorted in descending order)
        eigenvals = np.sort(linalg.eigvals(S2_plus_O2))[::-1]

        # # Store lambda2 (middle eigenvalue)
        # lambda2[i] = eigenvals[1]
        # # Compute Q-criterion
        # Q[i] = -0.5 * np.sum(eigenvals)

        # Add np.real() to handle complex eigenvalues
        lambda2[i] = np.real(eigenvals[1])
        Q[i] = -0.5 * np.real(np.sum(eigenvals))

    return lambda2, Q


if __name__ == "__main__":
    # Load data
    df = pd.read_csv(
        Path(project_dir)
        / "data"
        / "CFD_slices"
        / "spanwise_slices"
        / "alpha_6_CFD_spanwise_slice_10cm_1.csv"
    )

    # Compute lambda2 and Q-criterion
    lambda2, Q = compute_lambda2(df)

    # Add results to dataframe
    df["lambda2"] = lambda2
    df["Q"] = Q
    df["z"] = df["Points:0"]
    df["y"] = df["Points:1"]
    from scipy.interpolate import griddata

    n_subsample = 10

    # Use all points, not unique values
    y_values = df["y"].values
    z_values = df["z"].values

    y_grid = np.linspace(
        y_values.min(),
        y_values.max(),
        int(len(y_values) / n_subsample),
    )
    z_grid = np.linspace(
        z_values.min(),
        z_values.max(),
        int(len(z_values) / n_subsample),
    )

    Y_grid, Z_grid = np.meshgrid(y_grid, z_grid)
    color_data = griddata(
        (y_values, z_values),
        df["lambda2"].values,
        (Y_grid, Z_grid),
        method="linear",
    )

    # Plot results using pcolormesh
    fig, ax = plt.subplots()
    ax.pcolormesh(
        Y_grid,
        Z_grid,
        color_data,
        cmap="jet",
        shading="auto",
        # vmin=-100,
        # vmax=100,
    )
    plt.show()
