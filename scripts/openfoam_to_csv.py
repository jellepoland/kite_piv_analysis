"""
run the code below in terminal using cmd:

pvpython openfoam_to_csv.py


"""

# import os
# import sys
# from pathlib import Path
# from paraview.simple import *
# from vtk.util import numpy_support
# import numpy as np


# def scale_data(points, scale_factor=6.5):
#     """Scale spatial dimensions by the given factor."""
#     scaled_points = points.copy()
#     scaled_points *= scale_factor
#     return scaled_points


# def filter_data(
#     points, velocity_data, pressure_data=None, x_range=(-0.5, 2), y_range=(-0.6, 0.6)
# ):
#     # Create mask for points within the specified ranges
#     x_mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
#     y_mask = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
#     mask = x_mask & y_mask

#     # Apply mask to all data
#     filtered_points = points[mask]
#     filtered_velocity = velocity_data[mask]
#     filtered_pressure = pressure_data[mask] if pressure_data is not None else None

#     return filtered_points, filtered_velocity, filtered_pressure


# def extract_data(file_path, save_dir):
#     # Load OpenFOAM case
#     foam_case = OpenFOAMReader(FileName=file_path)
#     foam_case.UpdatePipeline()

#     # Define a plane for slicing
#     slice_plane = Slice(Input=foam_case)
#     slice_plane.SliceType = "Plane"
#     slice_plane.SliceType.Origin = [0, 0, 1e-9]
#     slice_plane.SliceType.Normal = [0, 0, 1]
#     slice_plane.UpdatePipeline()

#     # Fetch the slice data
#     slice_data = servermanager.Fetch(slice_plane)

#     if slice_data.GetNumberOfBlocks() > 0:
#         block = slice_data.GetBlock(0)
#         if block:
#             # Extract point coordinates
#             points = numpy_support.vtk_to_numpy(block.GetPoints().GetData())

#             # Extract velocity data
#             point_data = block.GetPointData()
#             velocity_array = point_data.GetArray("U")
#             velocity_data = numpy_support.vtk_to_numpy(velocity_array)

#             # Extract pressure data if available
#             pressure_array = point_data.GetArray("p")
#             pressure_data = (
#                 numpy_support.vtk_to_numpy(pressure_array) if pressure_array else None
#             )

#             # Filter data using scaled points
#             filtered_points, filtered_velocity, filtered_pressure = filter_data(
#                 points, velocity_data, pressure_data
#             )

#             # Scale points AFTER filtering
#             scaled_filtered_points = scale_data(filtered_points)

#             # Save filtered data
#             save_path = Path(save_dir)
#             save_path.mkdir(parents=True, exist_ok=True)

#             # Save as CSV files
#             save_points = np.column_stack(
#                 (
#                     scaled_filtered_points,  # These are already scaled
#                     filtered_velocity,
#                     (
#                         filtered_pressure
#                         if filtered_pressure is not None
#                         else np.zeros(len(scaled_filtered_points))
#                     ),
#                 )
#             )

#             header = "x,y,z,u,v,w" + (",p" if filtered_pressure is not None else "")
#             np.savetxt(
#                 save_path / "filtered_slice_data.csv",
#                 save_points,
#                 delimiter=",",
#                 header=header,
#             )

#             # Save unfiltered unscaled data as well
#             save_points_unfiltered = np.column_stack(
#                 (
#                     points,
#                     velocity_data,
#                     (
#                         pressure_data
#                         if pressure_data is not None
#                         else np.zeros(len(points))
#                     ),
#                 )
#             )
#             np.savetxt(
#                 save_path / "unfiltered_scaled_slice_data.csv",
#                 save_points_unfiltered,
#                 delimiter=",",
#                 header=header,
#             )

#             print(f"Data extracted, scaled, and filtered successfully!")
#             print(f"Scaling factor applied: 6.5")
#             print(f"Number of points before filtering: {len(points)}")
#             print(f"Number of points after filtering: {len(filtered_points)}")
#             print(f"Filtered data saved to: {save_path / 'filtered_slice_data.csv'}")
#             print(
#                 f"Unfiltered scaled data saved to: {save_path / 'unfiltered_scaled_slice_data.csv'}"
#             )

#             return filtered_points, filtered_velocity, filtered_pressure

#     print("No valid data found in the slice.")
#     return None, None, None


# if __name__ == "__main__":
#     file_path = "/home/jellepoland/ownCloud/phd/data/V3A/Lebesque_folder/results/1e6/6/foam.foam"

#     # Go back to root folder
#     root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     sys.path.insert(0, root_path)
#     save_dir = Path(root_path) / "processed_data" / "CFD"

#     filtered_points, filtered_velocity, filtered_pressure = extract_data(
#         file_path, save_dir
#     )

import os
import sys
from pathlib import Path
from paraview.simple import *
from vtk.util import numpy_support
import numpy as np


def scale_data(points, scale_factor=6.5):
    """Scale spatial dimensions by the given factor."""
    scaled_points = points.copy()
    scaled_points *= scale_factor
    return scaled_points


def filter_data(points, data_dict, x_range=(-3, 10), y_range=(-3.5, 3.5)):
    # Create mask for points within the specified ranges
    x_mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
    y_mask = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    mask = x_mask & y_mask

    # Apply mask to all data
    filtered_points = points[mask]
    filtered_data_dict = {key: data[mask] for key, data in data_dict.items()}

    return filtered_points, filtered_data_dict


def extract_data(file_path, save_dir, z_plane_origin=1):
    # Load OpenFOAM case
    foam_case = OpenFOAMReader(FileName=file_path)
    foam_case.UpdatePipeline()

    # Define a plane for slicing
    slice_plane = Slice(Input=foam_case)
    slice_plane.SliceType = "Plane"
    slice_plane.SliceType.Origin = [0, 0, z_plane_origin]
    slice_plane.SliceType.Normal = [0, 0, 1]
    slice_plane.UpdatePipeline()

    # Fetch the slice data
    slice_data = servermanager.Fetch(slice_plane)

    if slice_data.GetNumberOfBlocks() > 0:
        block = slice_data.GetBlock(0)
        if block:
            # Extract point coordinates
            points = numpy_support.vtk_to_numpy(block.GetPoints().GetData())

            # Create a dictionary to hold all extracted variables
            data_dict = {}

            # Extract available arrays from the OpenFOAM case
            point_data = block.GetPointData()

            def extract_variable(var_name):
                """Extract a variable array if it exists."""
                array = point_data.GetArray(var_name)
                if array:
                    data_dict[var_name] = numpy_support.vtk_to_numpy(array)
                    print(f"{var_name} extracted successfully")
                else:
                    print(f"{var_name} not found")

            # List of variables to extract
            variable_names = [
                "U",
                "p",
                "gammaInt",
                "grad(U)",
                "k",
                "nut",
                "omega",
                "vorticity",
            ]

            for var_name in variable_names:
                extract_variable(var_name)

            # Scale points BEFORE filtering
            scale_points = scale_data(points)

            # Filter data using points
            scaled_filtered_points, filtered_data_dict = filter_data(
                scale_points, data_dict
            )

            # Save filtered data
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Create the header dynamically based on variable dimensions
            header = ["x", "y", "z"]
            data_columns = [scaled_filtered_points]

            for var_name, data in filtered_data_dict.items():
                if data.ndim == 1:  # Scalar field
                    header.append(var_name)
                    data_columns.append(data)
                elif data.ndim == 2:  # Vector or tensor field
                    for i in range(data.shape[1]):
                        header.append(f"{var_name}_component{i}")
                    data_columns.append(data)

            save_points = np.column_stack(data_columns)

            # Save as CSV files
            np.savetxt(
                save_path / "filtered_slice_data.csv",
                save_points,
                delimiter=",",
                header=",".join(header),
            )

            print(f"Data extracted, scaled, and filtered successfully!")
            print(f"Scaling factor applied: 6.5")
            print(f"Number of points before filtering: {len(points)}")
            print(f"Number of points after filtering: {len(scaled_filtered_points)}")
            print(f"Filtered data saved to: {save_path / 'filtered_slice_data.csv'}")

            return scaled_filtered_points, filtered_data_dict

    print("No valid data found in the slice.")
    return None, None


if __name__ == "__main__":
    file_path = "/home/jellepoland/ownCloud/phd/data/V3A/Lebesque_folder/results/1e6/6/foam.foam"

    # Go back to root folder
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_path)
    save_dir = Path(root_path) / "processed_data" / "CFD"

    filtered_points, filtered_data_dict = extract_data(file_path, save_dir)
