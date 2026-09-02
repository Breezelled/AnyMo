import pandas as pd
import torch

from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
import numpy as np

import consts as d3_consts
from wimusim.utils import resolve_child_pose, simulate_imu


def calc_joint_params_from_BS_quats(
    df_bs: pd.DataFrame,
    apply_savgol_filter: bool = True,
    savgol_window_length: int = 50,
    savgol_polyorder: int = 3,
    savgol_mode: str = "nearest",
) -> list[np.ndarray]:
    """
    Calculate joint parameters from the bluesense quaternions.
    :param df_bs:
    :param apply_savgol_filter:
    :param savgol_window_length:
    :param savgol_polyorder:
    :return:
    """
    q_BS_TRS, q_BS_RUA, q_BS_RWR = (
        R.from_quat(df_bs[quat_cols])
        for quat_cols in [
            [f"{loc}_Quat{axis}" for axis in "XYZW"] for loc in ["TRS", "RUA", "RWR"]
        ]
    )

    # Align the coordinate system of BlueSense with the TRS of the robot.
    q_TRS = q_BS_TRS * R.from_euler(
        "XYZ", [np.pi / 2, 0, 0]
    )  # Rotate 90 deg along x-axis.

    p_pelvis = q_TRS.as_euler("XYZ")
    p_shoulder = (q_TRS.inv() * q_BS_RUA).as_euler("ZYX")
    p_elbow = (q_BS_RUA.inv() * q_BS_RWR).as_euler("ZYX")

    if apply_savgol_filter:
        return [
            savgol_filter(
                joint_param,
                savgol_window_length,
                savgol_polyorder,
                axis=0,
                mode=savgol_mode,
            )
            for joint_param in (p_pelvis, p_shoulder, p_elbow)
        ]

    return [p_pelvis, p_shoulder, p_elbow]


def measurement_func(E, B, D, P):
    """
    Simulates the Inertial Measurement Unit (IMU) readings for TRS, RUA, and RWR positions.
    This function is designed specifically for the digit drawing dataset.

    Parameters:

    Returns:
    dict: A dictionary containing simulated IMU data for the torso (TRS), right upper arm (RUA),
          and right wrist (RWR) in the format {'IMU_NAME': (position, orientation)}.
    """
    device = E["g"].device

    p_world = torch.tensor(d3_consts.p_WORLD, device=device)
    o_world = torch.tensor(d3_consts.o_WORLD, device=device)

    p_pelvis, q_pelvis = resolve_child_pose(
        p_world,
        o_world,
        B["rp"]["BASE2PELVIS"],
        D["pelvis"],
        parent_ori_type="euler",
        child_ori_type="euler",
    )
    p_shoulder, q_shoulder = resolve_child_pose(
        p_pelvis,
        q_pelvis,
        B["rp"]["PELVIS2R_SHOULDER"],
        D["shoulder"],
        child_ori_type="euler",
        c_ori_convert_order="ZYX",
    )
    p_elbow, q_elbow = resolve_child_pose(
        p_shoulder,
        q_shoulder,
        B["rp"]["R_SHOULDER2R_ELBOW"],
        D["elbow"],
        child_ori_type="euler",
        c_ori_convert_order="ZYX",
    )

    p_TRS, q_TRS = resolve_child_pose(
        p_pelvis,
        q_pelvis,
        P["rp"]["PELVIS2TRS"],
        P["ro"]["PELVIS2TRS"],
        child_ori_type="euler",
    )
    p_RUA, q_RUA = resolve_child_pose(
        p_shoulder,
        q_shoulder,
        P["rp"]["R_SHOULDER2RUA"],
        P["ro"]["R_SHOULDER2RUA"],
        child_ori_type="euler",
    )
    p_RWR, q_RWR = resolve_child_pose(
        p_elbow,
        q_elbow,
        P["rp"]["R_ELBOW2RWR"],
        P["ro"]["R_ELBOW2RWR"],
        child_ori_type="euler",
    )

    return {
        "TRS": simulate_imu(p_TRS, q_TRS, g=E["g"]),
        "RUA": simulate_imu(p_RUA, q_RUA, g=E["g"]),
        "RWR": simulate_imu(p_RWR, q_RWR, g=E["g"]),
    }
