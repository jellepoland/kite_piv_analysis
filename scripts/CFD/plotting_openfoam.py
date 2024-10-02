"""
run the code below in terminal using cmd:

pvpython plotting_openfoam.py


"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from vtk.util import numpy_support
from paraview.simple import *

# Go back to root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

file_path = (
    "/home/jellepoland/ownCloud/phd/data/V3A/Lebesque_folder/results/1e6/6/foam.foam"
)

# Load OpenFOAM case
foam_case = OpenFOAMReader(FileName=file_path)

# Update the pipeline to read data
foam_case.UpdatePipeline()

# Define a plane for slicing
slice_plane = Slice(Input=foam_case)
slice_plane.SliceType = "Plane"

# Set the plane parameters (origin and normal vector)
slice_plane.SliceType.Origin = [0, 0, 0]  # Set your origin here
slice_plane.SliceType.Normal = [0, 0, 1]  # Set your normal here

# Update the pipeline with the slice
slice_plane.UpdatePipeline()

# Extract data from the slice
# The following extracts point data (like velocity, pressure, etc.)
slice_data = servermanager.Fetch(slice_plane)


def plot_velocity_data(foam_file, slice_data, grid_size=100):
    # Load OpenFOAM case
    foam_case = OpenFOAMReader(FileName=foam_file)

    # Update the pipeline to read data
    foam_case.UpdatePipeline()

    # Extract the mesh data
    mesh_data = servermanager.Fetch(foam_case)

    # Initialize lists to store points and velocity data
    all_positions = []
    all_velocity_data = []

    # Check if the mesh data is a multi-block dataset
    if mesh_data.IsA("vtkMultiBlockDataSet"):
        num_blocks = mesh_data.GetNumberOfBlocks()
        for i in range(num_blocks):
            block = mesh_data.GetBlock(i)
            if block and block.IsA("vtkDataSet"):
                points = block.GetPoints()
                if points:
                    n_points = points.GetNumberOfPoints()
                    # Extract the position of each point
                    positions = np.array([points.GetPoint(j) for j in range(n_points)])
                    all_positions.append(positions)

                    # Extract the velocity data for this block
                    velocity_array = slice_data.GetPointData().GetArray(
                        "U"
                    )  # Change 'U' to your field name
                    velocity_data = numpy_support.vtk_to_numpy(velocity_array)
                    all_velocity_data.append(velocity_data)

    # Combine all positions and velocity data from all blocks
    all_positions = np.vstack(all_positions)
    all_velocity_data = np.concatenate(all_velocity_data)

    # Interpolate velocity data onto a grid
    grid_x, grid_y = np.mgrid[
        all_positions[:, 0].min() : all_positions[:, 0].max() : grid_size * 1j,
        all_positions[:, 1].min() : all_positions[:, 1].max() : grid_size * 1j,
    ]

    # Interpolate the velocity data onto the grid
    grid_velocity = griddata(
        all_positions[:, :2],
        all_velocity_data[:, :2],
        (grid_x, grid_y),
        method="linear",
    )

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(
        grid_velocity[:, :, 0],
        extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
        origin="lower",
    )
    plt.colorbar(label="Velocity Magnitude")
    plt.title("Interpolated Velocity Field")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()


# Call the plotting function
plot_velocity_data(file_path, slice_data)
