import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path


def plot_slice_data(csv_file):
    # Load data
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)

    # Extracting coordinates and velocity components
    x, y = data[:, 0], data[:, 1]
    u, v = data[:, 3], data[:, 4]

    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(u**2 + v**2)

    # Avoid division by zero by setting a minimum velocity magnitude
    velocity_magnitude[velocity_magnitude == 0] = (
        np.nan
    )  # Replace zero magnitudes with NaN

    # Normalize velocity components, handling non-finite values
    u_norm = np.divide(u, velocity_magnitude, where=np.isfinite(velocity_magnitude))
    v_norm = np.divide(v, velocity_magnitude, where=np.isfinite(velocity_magnitude))

    # Masking non-finite values
    mask = np.isfinite(u_norm) & np.isfinite(v_norm) & np.isfinite(velocity_magnitude)
    x = x[mask]
    y = y[mask]
    u_norm = u_norm[mask]
    v_norm = v_norm[mask]
    velocity_magnitude = velocity_magnitude[mask]
    u = u[mask]
    v = v[mask]

    # Create plots
    fig = plt.figure(figsize=(15, 12))

    # Plot normalized u-component
    ax1 = fig.add_subplot(221)
    mask = (u_norm > 0.8) & (u_norm < 1.2)
    u_norm = u_norm[mask]
    u_contour = ax1.tricontourf(
        x, y, u_norm, levels=20, cmap="coolwarm", vmin=u_norm.min(), vmax=u_norm.max()
    )
    plt.colorbar(u_contour, ax=ax1, label="Normalized u-velocity")
    ax1.set_title("Normalized u-component of Velocity")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")

    # Plot normalized v-component
    ax2 = fig.add_subplot(222)
    v_contour = ax2.tricontourf(
        x, y, v_norm, levels=20, cmap="coolwarm", vmin=v_norm.min(), vmax=v_norm.max()
    )
    plt.colorbar(v_contour, ax=ax2, label="Normalized v-velocity")
    ax2.set_title("Normalized v-component of Velocity")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    # Plot velocity magnitude
    ax3 = fig.add_subplot(223)
    velocity_magnitude = velocity_magnitude / np.average(velocity_magnitude)
    mask = (velocity_magnitude > 0.9) & (velocity_magnitude < 1.1)
    magnitude_contour = ax3.tricontourf(
        x[mask],
        y[mask],
        velocity_magnitude[mask],
        levels=200,
        cmap="viridis",
        # vmin=0.9,  # velocity_magnitude.min(),
        # vmax=1.1,  # velocity_magnitude.max(),
    )
    plt.colorbar(magnitude_contour, ax=ax3, label="Velocity Magnitude")
    ax3.set_title("Velocity Magnitude")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_aspect("equal")

    # Data points distribution
    ax4 = fig.add_subplot(224)
    ax4.scatter(x, y, c="b", s=1, alpha=0.5)
    ax4.set_title("Data Points Distribution")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_aspect("equal")

    plt.tight_layout()

    # Save plot
    plot_path = csv_file.parent / "slice_visualization_refined.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, root_path)
    save_dir = Path(root_path) / "processed_data" / "CFD" / "filtered_slice_data.csv"
    plot_slice_data(save_dir)
