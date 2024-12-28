import pandas as pd
from pathlib import Path
from utils import reading_optimal_bound_placement
from utils import project_dir


import pandas as pd
from pathlib import Path


def read_PIV_parameter_sweep_results(
    alpha: float, y_num: int, is_ellipse: bool, data_type: str = "PIV"
) -> pd.DataFrame:
    """
    Read the parameter sweep results CSV for a specific configuration.

    Args:
        alpha: Angle of attack.
        y_num: Y value index.
        is_ellipse: Whether the results are for an ellipse (True) or rectangle (False).
        project_dir: Base directory where the results are stored.
        data_type: Type of data collected.

    Returns:
        DataFrame containing the parameter sweep results.
    """
    # Construct the shape string
    shape = "Ellipse" if is_ellipse else "Rectangle"

    # Construct the file path
    save_folder = (
        Path(project_dir) / "processed_data" / "convergence_study" / "PIV_sweep"
    )
    file_path = Path(save_folder) / f"alpha_{alpha}_Y{y_num}_{data_type}_{shape}.csv"
    return pd.read_csv(file_path)


def read_results(alpha, y_num, is_CFD, is_ellipse):
    """
    Read Cl and Cd values from saved files based on alpha, y_num, is_CFD, and is_ellipse.

    Args:
        alpha: Angle of attack.
        y_num: Y value index.
        is_CFD: Flag for CFD data.
        is_ellipse: Flag for ellipse shape.

    Returns:
        tuple: Cl and Cd values.
    """
    # Define the folder path and filename pattern
    folder_path = Path(project_dir) / "processed_data" / "convergence_study"
    shape = "Ellipse" if is_ellipse else "Rectangle"
    data_type = "CFD" if is_CFD else "PIV"

    # Initialize results dictionary
    results = {"C_l": None, "C_d": None}

    # Get optimal boundary placement parameters
    dLx, dLy, iP = reading_optimal_bound_placement(
        alpha, y_num, is_with_N_datapoints=True
    )
    # print(f"optimal: dLx, dLy, iP: {dLx}, {dLy}, {iP}")

    # # Read the CSV file
    # df = pd.read_csv(f"{file_path})
    # df["dLx"] = pd.to_numeric(df["dLx"], errors="coerce")
    # df["dLy"] = pd.to_numeric(df["dLy"], errors="coerce")
    # df["iP"] = pd.to_numeric(df["iP"], errors="coerce")

    df_iP = pd.read_csv(
        Path(folder_path) / f"alpha_{alpha}_Y{y_num}_{data_type}_{shape}_iP.csv"
    )
    df_dLx = pd.read_csv(
        Path(folder_path) / f"alpha_{alpha}_Y{y_num}_{data_type}_{shape}_dLx.csv"
    )
    df_dLy = pd.read_csv(
        Path(folder_path) / f"alpha_{alpha}_Y{y_num}_{data_type}_{shape}_dLy.csv"
    )
    df = pd.concat([df_iP, df_dLx, df_dLy], axis=0, ignore_index=True)

    # print(f"df-head: {df.head()}")
    # print(f'df.ip {df["iP"].values}')
    # print(f'df.dLx {df["dLx"].values}')
    # print(f'df.dLy {df["dLy"].values}')

    if is_CFD:
        # # For CFD data, exact match on iP, dLx, and dLy
        # df_iP = df[(df["iP"] == iP)]
        # print(f"df_iP:{df_iP}")
        # df_iP_dLx = df_iP[(df_iP["dLx"] == dLx)]
        # print(df_iP_dLx)
        # df_iP_dLy = df_iP_dLx[(df_iP_dLx["dLy"] == dLy)]
        # print(df_iP_dLy)
        # filtered_df = df_iP_dLy
        filtered_df = df[(df["iP"] == iP) & (df["dLx"] == dLx) & (df["dLy"] == dLy)]
    else:
        # # For PIV data, allow ±5% tolerance on dLx and dLy
        # filtered_df = df[
        #     (df["iP"] == iP)
        #     & (df["dLx"].between(dLx * 0.9, dLx * 1.1))
        #     & (df["dLy"].between(dLy * 0.9, dLy * 1.1))
        # ]
        filtered_df = read_PIV_parameter_sweep_results(alpha, y_num, is_ellipse)

        # For non-CFD data, take the mean of matching rows
        if not filtered_df.empty:
            filtered_df = filtered_df.mean(numeric_only=True).to_frame().T

    # Extract Cl and Cd from the filtered data
    if not filtered_df.empty:
        results["C_l"] = filtered_df["C_l"].values[0]
        results["C_d"] = filtered_df["C_d"].values[0]
        results["d1centre_x"] = filtered_df["d1centre_x"].values[0]
        results["d1centre_y"] = filtered_df["d1centre_y"].values[0]
        results["perc_of_interpolated_points"] = filtered_df[
            "perc_of_interpolated_points"
        ].values[0]
        results["Gamma"] = filtered_df["Gamma"].values[0]
        results["F_kutta"] = filtered_df["F_kutta"].values[0]

    else:
        raise ValueError(
            f"No matching data found for the given parameters. File: {file_path}"
        )

    return results


def main():
    # Settings
    parameter_names = ["iP", "dLx", "dLy"]

    list_Gamma = []

    for alpha in [6, 16]:
        if alpha == 6:
            y_num_range = [1, 2, 3, 4, 5]
        elif alpha == 16:
            print(f" ------------------- ")
            y_num_range = [1]
        for y_num in y_num_range:
            results = read_results(alpha, y_num, True, True)
            d1centre_x = results["d1centre_x"]
            d1centre_y = results["d1centre_y"]
            perc_of_interpolated_points = results["perc_of_interpolated_points"]
            print(
                f"\n alpha={alpha} Y{y_num} centre_x: {d1centre_x:0.2f}, centre_y:{d1centre_y:0.2f}, perc_of_interpolated_points: {perc_of_interpolated_points}"
            )
            for is_CFD in [True, False]:
                for is_ellipse in [True, False]:
                    results = read_results(alpha, y_num, is_CFD, is_ellipse)
                    C_l = results["C_l"]
                    C_d = results["C_d"]
                    Gamma = results["Gamma"]
                    F_kutta = results["F_kutta"]
                    if is_CFD:
                        name = "CFD"
                    else:
                        name = "PIV"
                    if is_ellipse:
                        shape = "Ellipse"
                    else:
                        shape = "Rectangle"

                    print(
                        f"{name}  {shape} C_l: {C_l:.2f}, C_d: {C_d:.2f}, F_kutta: {F_kutta:.2f}, Gamma: {Gamma:.2f}"
                    )
                    list_Gamma.append(
                        {
                            "alpha": alpha,
                            "y_num": y_num,
                            "data_type": name,
                            "shape": shape,
                            "gamma": Gamma,
                        }
                    )

    df_y_locations = pd.read_csv(
        Path(project_dir) / "processed_data" / "circulation_plot" / "y_locations.csv",
        index_col=False,
    )
    # Initialize a new column for Gamma
    df_y_locations["Gamma_CFD_Ellipse"] = None
    df_y_locations["Gamma_CFD_Rectangle"] = None
    df_y_locations["Gamma_PIV_Ellipse"] = None
    df_y_locations["Gamma_PIV_Rectangle"] = None

    # Loop through the list_Gamma and populate the DataFrame
    for entry in list_Gamma:
        # Match the 'Y' column with y_num (e.g., Y1, Y2, etc.)
        mask = df_y_locations["Y"] == f'Y{entry["y_num"]}'

        if entry["alpha"] == 16:
            continue

        # Apply the relevant gamma value based on data_type and shape
        if entry["data_type"] == "CFD" and entry["shape"] == "Ellipse":
            df_y_locations.loc[mask, "Gamma_CFD_Ellipse"] = entry["gamma"]
        elif entry["data_type"] == "CFD" and entry["shape"] == "Rectangle":
            df_y_locations.loc[mask, "Gamma_CFD_Rectangle"] = entry["gamma"]
        elif entry["data_type"] == "PIV" and entry["shape"] == "Ellipse":
            df_y_locations.loc[mask, "Gamma_PIV_Ellipse"] = entry["gamma"]
        elif entry["data_type"] == "PIV" and entry["shape"] == "Rectangle":
            df_y_locations.loc[mask, "Gamma_PIV_Rectangle"] = entry["gamma"]

    df_y_locations.to_csv(
        Path(project_dir)
        / "processed_data"
        / "circulation_plot"
        / "y_locations_with_gamma.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
