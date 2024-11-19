import os
import pandas as pd
from utils import project_dir
from pathlib import Path
import numpy as np

# Define the directories and CSV file patterns
folder_dir = Path(project_dir) / "processed_data" / "stichted_planes_erik"
dirs = [Path(folder_dir) / "aoa_13", Path(folder_dir) / "aoa_23"]
csv_pattern = ".csv"
std_pattern = "_std.csv"

# Determine the masks using the non-_std files
masks = {}
for dir_name in dirs:
    for file in os.listdir(dir_name):
        if file.endswith(csv_pattern) and not file.endswith(std_pattern):
            file_path = Path(dir_name) / file
            df = pd.read_csv(file_path)
            mask = (df["w"] > -3) & (df["w"] < 3)
            masks[file.replace(csv_pattern, std_pattern)] = mask

factor = 1.96 / np.sqrt(250)

# Process the _std files using the stored masks
for dir_name in dirs:
    for file in os.listdir(dir_name):
        if file.endswith(std_pattern):
            file_path = Path(dir_name) / file
            df = pd.read_csv(file_path)
            mask = masks[file]

            # u_max = df["u"].max() * factor
            # v_max = df["v"].max() * factor
            # w_max = df["w"].max() * factor
            # V_max = df["V"].max() * factor

            df = df[mask]
            u_max_mask = df["u"].max() * factor
            v_max_mask = df["v"].max() * factor
            w_max_mask = df["w"].max() * factor
            V_max_mask = df["V"].max() * factor

            u_mean_mask = df["u"].mean() * factor
            v_mean_mask = df["v"].mean() * factor
            w_mean_mask = df["w"].mean() * factor
            V_mean_mask = df["V"].mean() * factor

            print(f"File: {file}")
            # print(f"u maximum: {u_max:.2f}")
            # print(f"v maximum: {v_max:.2f}")
            # print(f"w maximum: {w_max:.2f}")
            # print(f"V maximum: {V_max:.2f}")
            print(f"std(u) * k/sqrt(N) masked: {u_max_mask:.2f})")
            print(f"std(v) * k/sqrt(N) masked: {v_max_mask:.2f})")
            print(f"std(w) * k/sqrt(N) masked: {w_max_mask:.2f})")
            print(f"std(V) * k/sqrt(N) masked: {V_max_mask:.2f})")
            print(f"mean(u) * k/sqrt(N) masked: {u_mean_mask:.2f})")
            print(f"mean(v) * k/sqrt(N) masked: {v_mean_mask:.2f})")
            print(f"mean(w) * k/sqrt(N) masked: {w_mean_mask:.2f})")
            print(f"mean(V) * k/sqrt(N) masked: {V_mean_mask:.2f})")

            print()
