import torch
import numpy as np
import zipfile
import pandas as pd


# Path to your zip file
def generate_df_dict(zip_path, verbose, recording_no=-1, zfile=None):
    """
    Extract the CSV files from the zip and return a dictionary of dataframes.
    Subject 4, 7, 14's "climbingdown", "climbingup" is a zip of zips with the name {recording_no}_csv.zip.
    """
    if verbose:
        print("Extracting zip at", zip_path)
    if zfile is None:
        zfile = zipfile.ZipFile(zip_path, "r")

    df_dict = {}
    # List all files in the zip
    for filename in zfile.namelist():
        # print(filename.endswith(f'{recording_no}_csv.zip'))
        if filename.endswith(f"{recording_no}_csv.zip"):
            # This is the normal case.
            if verbose:
                print("Extracting nested zip from", filename)
            # Read the nested zip file
            with zfile.open(filename) as nested_zip:
                with zipfile.ZipFile(nested_zip) as z_nested:
                    # Recursively call to process files in the nested zip
                    nested_df_dict = generate_df_dict(zip_path, verbose, zfile=z_nested)
                    return nested_df_dict
        elif filename.endswith(f"csv.zip") and recording_no == 1:
            # Sometimes, {recording_no} is not included in the filename (when recording_no = 1)
            if verbose:
                print("Extracting nested zip from", filename)
            # Read the nested zip file
            with zfile.open(filename) as nested_zip:
                with zipfile.ZipFile(nested_zip) as z_nested:
                    # Recursively call to process files in the nested zip
                    nested_df_dict = generate_df_dict(zip_path, verbose, zfile=z_nested)
                    return nested_df_dict

        # Check for CSV files
        elif filename.endswith(".csv"):
            fileinfo = filename.split(
                "_"
            )  # e.g. acc_walking_chest.csv, Gyroscope_walking_chest.csv
            activity_class = fileinfo[1]
            sensor_loc = fileinfo[-1].split(".")[0]
            if "acc" in fileinfo[0]:
                sensor_type = "Acc"
            elif "Gyro" in fileinfo[0]:
                sensor_type = "Gyro"
            elif "MagneticField" in fileinfo[0]:
                sensor_type = "Mag"
            else:
                raise "unexpected sensor type"

            # Read the CSV file directly from the ZIP
            with zfile.open(filename) as f:
                df = pd.read_csv(f)
                # df["attr_time"] = pd.to_datetime(df["attr_time"], )
                df.rename(
                    columns={
                        "attr_x": f"{sensor_loc}_{sensor_type}X",
                        "attr_y": f"{sensor_loc}_{sensor_type}Y",
                        "attr_z": f"{sensor_loc}_{sensor_type}Z",
                    },
                    inplace=True,
                )
                df_dict[filename] = df.set_index("attr_time")
                if verbose:
                    print(
                        f"\t{filename} - num of samples: {df_dict[filename].shape[0]}"
                    )
    return df_dict


def generate_default_placement_params(B_rp_dict):
    """
    Generate default placement parameters for the body based on the body parameters B.
    :param B_rp_dict:
    :return:
    """
    return {
        "rp": {
            ("NECK", "LHD"): np.array(
                [
                    B_rp_dict[("L_CLAVICLE", "L_SHOULDER")][0] / 2,
                    0.0,
                    B_rp_dict[("NECK", "HEAD")][2] - 0.07,
                ]
            ),
            ("PELVIS", "WST"): np.array(
                [0.0, 0.1, B_rp_dict[("PELVIS", "BELLY")][2] * 2 / 3]
            ),
            ("BELLY", "LCH"): np.array(
                [
                    B_rp_dict[("L_CLAVICLE", "L_SHOULDER")][0] / 2,
                    0.1,
                    B_rp_dict[("PELVIS", "BELLY")][2] * 2 / 3,
                ]
            ),
            ("L_SHOULDER", "LUA"): np.array(
                [B_rp_dict[("L_SHOULDER", "L_ELBOW")][0] + 0.07, 0, 0.04]
            ),
            ("L_ELBOW", "LLA"): np.array(
                [B_rp_dict[("L_ELBOW", "L_WRIST")][0] + 0.025, 0, 0.025]
            ),
            ("L_HIP", "LTH"): np.array(
                [-0.05, 0.05, B_rp_dict[("L_HIP", "L_KNEE")][2] / 4]
            ),
            ("L_KNEE", "LSH"): np.array(
                [-0.04, 0.0, B_rp_dict[("L_KNEE", "L_ANKLE")][2] + 0.07]
            ),
        },
        "ro": {
            ("NECK", "LHD"): np.deg2rad(
                np.array([90.0, -90.0, 0.0])
                # np.array([90.0, 90.0, 0.0]) facing head side
                # np.array([-90.0, -90.0, 0.0]) up-side down facing the left
                # np.array([-90.0, 90.0, 0.0]) up-side down facing the right (head)
            ),  # adjust tilt of by rotating Z
            ("PELVIS", "WST"): np.deg2rad(
                np.array([-90.0, 0.0, -90.0])
            ),  # [-90, 0, 90] for some subjects
            ("BELLY", "LCH"): np.deg2rad(np.array([90.0, 180.0, 0.0])),
            ("L_SHOULDER", "LUA"): np.deg2rad(np.array([0.0, 0.0, -90.0])),
            ("L_ELBOW", "LLA"): np.deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_HIP", "LTH"): np.deg2rad(
                np.array([90.0, 180.0 + 30.0, 0.0])
            ),  # rotate along the thigh
            ("L_KNEE", "LSH"): np.deg2rad(np.array([90.0, 90.0, 0])),
        },
    }


def generate_B_range(B_rp_dict, min_scale=0.8, max_scale=1.2, near_zero=[-0.01, 0.01]):
    B_range_dict = {}
    for k, v in B_rp_dict.items():
        B_range_dict[k] = np.array(
            [sorted([e * min_scale, e * max_scale]) if e != 0 else near_zero for e in v]
        )
    return B_range_dict
