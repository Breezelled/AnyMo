import numpy as np
import torch

import pytorch3d.transforms.rotation_conversions as rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .consts import (
    M36H_JOINT_PAIR_DICT,
    M36H_JOINT_PARENT_CHILD_PAIRS,
    M36H_JOINT_ID_DICT,
    M36H_JOINT_CHILD_PARENT_DICT,
)

from wimusim.utils import resolve_child_pose


def ydown2zup(motion):
    """
    This function converts the motion data to the z-up coordinate system for compatibility with WIMUSim
    Presumably, the original coordinate system is y-down (X-left, Y-down, Z-back).
    This function convert the original motion data to the z-up coordinate system (X-right, Y-front, Z-up)

    Args:
    - motion: np.ndarray, shape=(T, J, 3), T: number of frames, J: number of joints, 3: x, y, z
    """
    motion_zup = motion.copy()
    motion_zup = -motion_zup[:, :, [0, 2, 1]]
    return motion_zup


def calc_B_param(motion: np.ndarray, target_height: float = 1.7, return_scale: bool = False, verbose: bool = False):
    """
    Calculate the B parameters from the 3d motion data (of shape [T, 17, 3]) based on the target height.
    Currently, only Human3.6M's 17 joints model is supported.

    Args:
    - motion: np.ndarray, shape=(T, J, 3), T: number of frames, J: number of joints, 3: x, y, z
    - target_height: float, the target height of the skeleton in meters
    - return_scale: bool, whether to return the scale factor
    - verbose: bool, whether to print the intermediate results
    """
    # Calculate the average length of each limb from the motion data
    limb_len_dict = {}
    for parent2child, (parent_id, child_id) in M36H_JOINT_PAIR_DICT.items():
        limb_len_dict[parent2child] = np.linalg.norm(
            motion[:, parent_id, :] - motion[:, child_id, :], axis=1
        ).mean()

    # Calculate the height of the skeleton
    skeleton_height_upper = (
        limb_len_dict[("PELVIS", "BELLY")]
        + limb_len_dict[("BELLY", "NECK")]
        + limb_len_dict[("NECK", "HEAD")]
    )
    skeleton_height_lower_right = (
        limb_len_dict[("R_HIP", "R_KNEE")] + limb_len_dict[("R_KNEE", "R_ANKLE")]
    )
    # skeleton_height_lower_left = limb_len_dict["L_HIP2L_KNEE"] + limb_len_dict["L_KNEE2L_ANKLE"]

    # Currently, the length of the lower body is calculated based on the right leg only.
    skeleton_height = skeleton_height_upper + skeleton_height_lower_right
    scale = target_height / skeleton_height

    if verbose:
        print(f"Skeleton height: {skeleton_height}")
        print(f"Scale: {scale}")

    B = {
        "rp": {
            ("BASE", "PELVIS"): torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            ("PELVIS", "BELLY"): torch.tensor(
                [0.0, 0.0, limb_len_dict[("PELVIS", "BELLY")] * scale],
                dtype=torch.float32,
            ),
            ("BELLY", "NECK"): torch.tensor(
                [0.0, 0.0, limb_len_dict[("BELLY", "NECK")] * scale],
                dtype=torch.float32,
            ),
            ("NECK", "NOSE"): torch.tensor(
                [
                    0.0,
                    limb_len_dict[("NECK", "HEAD")] * scale * np.sqrt(3) / 2.0,
                    limb_len_dict[("NECK", "HEAD")] * scale / 2.0,
                ],
                dtype=torch.float32,
            ),
            ("NOSE", "HEAD"): torch.tensor(
                [
                    0.0,
                    -limb_len_dict[("NECK", "HEAD")] * scale * np.sqrt(3) / 2.0,
                    limb_len_dict[("NECK", "HEAD")] * scale / 2.0,
                ],
                dtype=torch.float32,
            ),
            ("NECK", "HEAD"): torch.tensor(
                [0.0, 0.0, limb_len_dict[("NECK", "HEAD")] * scale], dtype=torch.float32
            ),
            ("BELLY", "R_CLAVICLE"): torch.tensor(
                [0.0, 0.0, limb_len_dict[("BELLY", "R_CLAVICLE")] * scale],
                dtype=torch.float32,
            ),
            ("R_CLAVICLE", "R_SHOULDER"): torch.tensor(
                [limb_len_dict[("R_CLAVICLE", "R_SHOULDER")] * scale, 0.0, 0.0],
                dtype=torch.float32,
            ),
            ("R_SHOULDER", "R_ELBOW"): torch.tensor(
                [limb_len_dict[("R_SHOULDER", "R_ELBOW")] * scale, 0.0, 0.0],
                dtype=torch.float32,
            ),
            ("R_ELBOW", "R_WRIST"): torch.tensor(
                [limb_len_dict[("R_ELBOW", "R_WRIST")] * scale, 0.0, 0.0],
                dtype=torch.float32,
            ),
            ("BELLY", "L_CLAVICLE"): torch.tensor(
                [0.0, 0.0, limb_len_dict[("BELLY", "L_CLAVICLE")] * scale],
                dtype=torch.float32,
            ),
            ("L_CLAVICLE", "L_SHOULDER"): torch.tensor(
                [-limb_len_dict[("L_CLAVICLE", "L_SHOULDER")] * scale, 0, 0.0],
                dtype=torch.float32,
            ),
            ("L_SHOULDER", "L_ELBOW"): torch.tensor(
                [-limb_len_dict[("L_SHOULDER", "L_ELBOW")] * scale, 0.0, 0.0],
                dtype=torch.float32,
            ),
            ("L_ELBOW", "L_WRIST"): torch.tensor(
                [-limb_len_dict[("L_ELBOW", "L_WRIST")] * scale, 0.0, 0.0],
                dtype=torch.float32,
            ),
            ("BASE", "R_HIP"): torch.tensor(
                [limb_len_dict[("BASE", "R_HIP")] * scale, 0.0, 0.0],
                dtype=torch.float32,
            ),
            ("R_HIP", "R_KNEE"): torch.tensor(
                [0.0, 0.0, -limb_len_dict[("R_HIP", "R_KNEE")] * scale],
                dtype=torch.float32,
            ),
            ("R_KNEE", "R_ANKLE"): torch.tensor(
                [0.0, 0.0, -limb_len_dict[("R_KNEE", "R_ANKLE")] * scale],
                dtype=torch.float32,
            ),
            ("BASE", "L_HIP"): torch.tensor(
                [-limb_len_dict[("BASE", "L_HIP")] * scale, 0.0, 0.0],
                dtype=torch.float32,
            ),
            ("L_HIP", "L_KNEE"): torch.tensor(
                [0.0, 0.0, -limb_len_dict[("L_HIP", "L_KNEE")] * scale],
                dtype=torch.float32,
            ),
            ("L_KNEE", "L_ANKLE"): torch.tensor(
                [0.0, 0.0, -limb_len_dict[("L_KNEE", "L_ANKLE")] * scale],
                dtype=torch.float32,
            ),
        }
    }

    if return_scale:
        return B, scale

    return B


# Ensure v is normalized
def get_rot_from_u_to_v(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute the quaternion that rotates vector u to vector v.
    """

    if isinstance(u, np.ndarray):
        u = torch.tensor(u)
    if isinstance(v, np.ndarray):
        v = torch.tensor(v)

    # Ensure u and v are 2D
    if u.dim() == 1:
        u = u.unsqueeze(0)
    if v.dim() == 1:
        v = v.unsqueeze(0)

    u_norm = normalize_vec(u)

    v_norm = normalize_vec(v)

    # Broadcast u and v to the same shape if necessary
    if u_norm.shape[0] == 1 and v_norm.shape[0] > 1:
        u_norm = u_norm.expand(v_norm.shape)
    elif v_norm.shape[0] == 1 and u_norm.shape[0] > 1:
        v_norm = v_norm.expand(u_norm.shape)

    # Axis of rotation
    axis = normalize_vec(torch.cross(u_norm, v_norm, dim=-1))

    # Angle of rotation
    # angle = torch.acos(torch.dot(u_norm, v_norm)) # This only supports 1d tensor
    dot = torch.sum(u_norm * v_norm, dim=-1, keepdim=True)
    angle = torch.acos(torch.clamp(dot, -1.0, 1.0))

    # Compute the quaternion q where v = q * u * q^-1
    angle_axis = axis * angle
    orientation_quaternion = torch.nn.functional.normalize(
        rc.axis_angle_to_quaternion(angle_axis)
    )  # magnitude of axis is the angle turned anticlockwise in radians
    return orientation_quaternion


def normalize_vec(v: torch.Tensor) -> torch.Tensor:
    return v / torch.norm(v, dim=-1, keepdim=True)


# Batch implementation (test passed)
def calc_root_orientation(motion, calc_translation=True):
    """
    Calculate the orientation of the root joint based on the 3D pose of the human.

    Args:
        - motion: np.ndarray, shape=(T, J, 3), T: number of frames, J: number of joints, 3: x, y, z
    """
    root_idx, belly_idx, lhip_idx, rhip_idx = 0, 7, 4, 1
    # sequence of 3d vector (T, 3) from the root to the belly
    ROOT_BELLY = normalize_vec(torch.tensor(motion[:, belly_idx] - motion[:, root_idx]))

    # Define the xyz axes of the root joint
    root_x_axis = normalize_vec(
        torch.tensor(motion[:, rhip_idx] - motion[:, lhip_idx])
    )  # LHIP-RHIP vector
    root_y_axis = torch.cross(ROOT_BELLY, root_x_axis)
    root_z_axis = torch.cross(root_x_axis, root_y_axis)

    # Convert the axes to a quaternion
    BASE_q_w = rc.matrix_to_quaternion(
        torch.stack([root_x_axis, root_y_axis, root_z_axis], dim=-1)
    )

    if calc_translation:
        BASE_p_w = motion[:, root_idx] - motion[0, root_idx]
    else:
        BASE_p_w = torch.zeros((BASE_q_w.shape[0], 3))
    return BASE_p_w, BASE_q_w


def calc_D_param(motion, B_dict, enable_translation=False, scale=1.0):
    """
    Calculate the D parameters from the 3d motion data based on the B parameters.

    Args:
    - motion: np.ndarray, shape=(T, J, 3), T: number of frames, J: number of joints, 3: x, y, z
    - B_dict: dict, the B parameters
    - enable_translation: bool, whether to enable translation
    - scale: float, the scale factor for the translation
    """

    root_p, root_q = calc_root_orientation(motion, calc_translation=enable_translation)

    D_t = {
        joint: torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(motion.shape[0], 4)
        for joint in list(M36H_JOINT_ID_DICT.keys())
    }
    Q_t = D_t.copy()  # Dictionary for global orientation

    D_t["BASE"] = root_q
    Q_t["BASE"] = root_q

    for parent_name, child_name in M36H_JOINT_PARENT_CHILD_PAIRS:
        if parent_name == "BASE":
            # Base joint (root) is already calculated
            continue
        grand_parent = M36H_JOINT_CHILD_PARENT_DICT[parent_name]
        parent_q_def = Q_t[
            grand_parent
        ]  # By default, it must be same as the grandparent joint's global orientation
        child_idx = M36H_JOINT_ID_DICT[child_name]
        parent_idx = M36H_JOINT_ID_DICT[parent_name]
        parent2child_vec_global = torch.tensor(
            motion[:, child_idx] - motion[:, parent_idx]
        )  # 3D vector from parent to child in global frame
        parent2child_vec_local = rc.quaternion_apply(
            rc.quaternion_invert(parent_q_def), parent2child_vec_global
        )  # 3D vector from parent to child in local frame

        # Calculate the quaternion that rotates the default relative vector to the current relative vector
        D_t[parent_name] = get_rot_from_u_to_v(
            B_dict["rp"][(parent_name, child_name)], parent2child_vec_local
        )
        Q_t[parent_name] = rc.quaternion_multiply(Q_t[grand_parent], D_t[parent_name])

    D_dict = {
        "translation": {"XYZ": root_p * scale},
        "orientation": Q_t,
    }

    return D_dict


# Construct default 3D pose
def construct_3d_pose(B, D_t, p_origin=torch.tensor([0.0, 0.0, 0.0])):
    """
    Construct the 3D pose of the humanoid using the resolved joint orientations.
    """

    if D_t["BASE"].dim() != 1:
        p_origin = p_origin.expand(D_t["BASE"].shape[0], 3)

    resolved_joint_pose_dict = {
        "BASE": (p_origin, D_t["BASE"]),  # BASE at the origin.
    }

    # Calculate pose (position and orientation) of humanoid's joints
    for parent_name, child_name in M36H_JOINT_PARENT_CHILD_PAIRS:
        resolved_joint_pose_dict[child_name] = resolve_child_pose(
            resolved_joint_pose_dict[parent_name][0],  # parent's global position
            resolved_joint_pose_dict[parent_name][1],  # parent's global orientation
            B["rp"][
                (parent_name, child_name)
            ],  # child's relative position to the parent
            D_t[child_name],  # child's relative orientation to the parent
        )

    return resolved_joint_pose_dict


def visualize_motion3d(motion3d, frame, view_config={"elev": 12.0, "azim": 80}):
    # H36M, 0: 'root',
    #       1: 'rhip',
    #       2: 'rkne',
    #       3: 'rank',
    #       4: 'lhip',
    #       5: 'lkne',
    #       6: 'lank',
    #       7: 'belly',
    #       8: 'neck',
    #       9: 'nose',
    #       10: 'head',
    #       11: 'lsho',
    #       12: 'lelb',
    #       13: 'lwri',
    #       14: 'rsho',
    #       15: 'relb',
    #       16: 'rwri'

    joint_pairs = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [8, 11],
        [8, 14],
        [9, 10],
        # [8, 10], # neck to head
        [11, 12],
        [12, 13],
        [14, 15],
        [15, 16],
    ]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"

    fig = plt.figure(0, figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(**view_config)

    j3d = motion3d[frame, :, :]

    # This is to compare default posture and the current posture
    for i in range(len(joint_pairs)):
        limb = joint_pairs[i]
        xs, ys, zs = [
            np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)
        ]  # x, y, z coordinates of each joint
        if joint_pairs[i] in joint_pairs_left:
            ax.plot3D(
                xs,
                ys,
                zs,
                color=color_left,
                lw=3,
                marker="o",
                markerfacecolor="w",
                markersize=3,
                markeredgewidth=2,
            )  # axis transformation for visualization
        elif joint_pairs[i] in joint_pairs_right:
            ax.plot3D(
                xs,
                ys,
                zs,
                color=color_right,
                lw=3,
                marker="o",
                markerfacecolor="w",
                markersize=3,
                markeredgewidth=2,
            )  # axis transformation for visualization
        else:
            ax.plot3D(
                xs,
                ys,
                zs,
                color=color_mid,
                lw=3,
                marker="o",
                markerfacecolor="w",
                markersize=3,
                markeredgewidth=2,
            )  # axis transformation for visualization

    plt.show(block=False)
    return
