"""
run the code below in terminal using cmd:

pvpython openfoam_to_csv.py


"""

import os
import sys
from pathlib import Path
from paraview.simple import *
from vtk.util import numpy_support
import numpy as np

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

# Fetch the slice data
slice_data = servermanager.Fetch(slice_plane)

# Extract data from the multi-block dataset
if slice_data.GetNumberOfBlocks() > 0:
    block = slice_data.GetBlock(0)  # Get the first block
    if block:
        # Extract the point data from the block
        point_data = block.GetPointData()

        # Extract the array of interest (e.g., velocity, pressure)
        velocity_array = point_data.GetArray(
            "U"
        )  # Replace 'U' with the name of your field variable
        velocity_data = numpy_support.vtk_to_numpy(velocity_array)

        # Save or process the extracted data
        file_path_save = (
            Path(root_path) / "processed_data" / "CFD" / "velocity_data.csv"
        )
        np.savetxt(file_path_save, velocity_data, delimiter=",")

        print("Data extracted and saved successfully!")
    else:
        print("No valid block found.")
else:
    print("No blocks available in the slice data.")
