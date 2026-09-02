import warnings
from typing import Literal, List, Dict
from torch import Tensor
import numpy.typing as npt

import torch
import pybullet as p
import numpy as np
import pytorch3d.transforms.rotation_conversions as rc
from scipy.signal import butter, lfilter

from wimusim.consts import H_DEFAULT, POS_DEFAULT, ORI_DEFAULT


def check_wimusim_param_args_consistency(
    data_type: Literal["numpy", "tensor"],
    device: str,
    requires_grad: bool,
):
    """
    Check if the arguments are consistent with each other.
    :param data_type: "numpy" or "tensor"
    :param device: "cpu" or "cuda"
    :param requires_grad: True or False
    :return: None
    """
    if data_type == "numpy":
        if device is not None:
            warnings.warn(f"device={device} is ignored when data_type is {data_type}")
        if requires_grad is True:
            warnings.warn(
                f"requires_grad={requires_grad} is ignored when data_type is {data_type}"
            )
    elif data_type == "tensor":
        pass
    else:
        raise ValueError(f"invalid data_type: {data_type}")


def resolve_child_pose(
    p_parent_world: Tensor,
    q_parent_world: Tensor,
    rel_p_child: Tensor,
    rel_q_child: Tensor,
    parent_ori_type: Literal["quat", "euler"] = "quat",
    p_ori_convert_order: str = "XYZ",
    child_ori_type: Literal["quat", "euler"] = "quat",
    c_ori_convert_order: str = "XYZ",
) -> (Tensor, Tensor):
    """
    Compute the world frame pose of a child object based on its relative pose to a parent and the parent's world frame pose.

    Parameters:
    - p_parent_world (Tensor): The world position of the parent object.
    - q_parent_world (Tensor): The world orientation quaternion (4 elements) or euler angles (3 elements) of the parent object, based on the parent_ori_type.
    - rel_p_child (Tensor): The position of the child object relative to the parent object in the parent's local frame.
    - rel_q_child (Tensor): The orientation quaternion (4 elements) or euler angles (3 elements) of the child object relative to the parent object, based on the child_ori_type.
    - parent_ori_type (Literal["quat", "euler"]): The type of the parent's orientation representation; 'quat' for quaternion, 'euler' for euler angles.
    - p_ori_convert_order (str): The order of rotations for converting euler angles to a quaternion for the parent's orientation if euler angles are used. Should be one of 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'.
    - child_ori_type (Literal["quat", "euler"]): The type of the child's orientation representation; 'quat' for quaternion, 'euler' for euler angles.
    - c_ori_convert_order (str): The order of rotations for converting euler angles to a quaternion for the child's orientation if euler angles are used. Should follow the same convention as p_ori_convert_order.

    Returns:
    - (Tensor, Tensor): A tuple containing the world position (Tensor) and orientation quaternion (Tensor) of the child object.

    This function allows for flexible input of orientation representations and conversion orders to accommodate different use cases and coordinate systems conventions.
    """
    if parent_ori_type == "euler":
        q_parent_world = rc.matrix_to_quaternion(
            rc.euler_angles_to_matrix(q_parent_world, p_ori_convert_order)
        )
        q_parent_world = rc.standardize_quaternion(q_parent_world)
    if child_ori_type == "euler":
        rel_q_child = rc.matrix_to_quaternion(
            rc.euler_angles_to_matrix(rel_q_child, c_ori_convert_order)
        )
        rel_q_child = rc.standardize_quaternion(rel_q_child)

    # Compute the child's world frame position by applying the parent's orientation to the child's relative position
    # and then adding the parent's world position.
    p_child_world = rc.quaternion_apply(q_parent_world, rel_p_child) + p_parent_world

    # Compute the child's world frame orientation by multiplying the parent's orientation with the child's relative orientation.
    q_child_world = rc.quaternion_multiply(q_parent_world, rel_q_child)

    # Standardize and normalize the quaternion to ensure it is a unit quaternion
    q_child_world = rc.standardize_quaternion(q_child_world)
    # q_child_world /= torch.norm(q_child_world, dim=1, keepdim=True) # Not necessary

    return p_child_world, q_child_world


def simulate_imu(
    p: Tensor,
    q: Tensor,
    ba: Tensor,
    bg: Tensor,
    eta_a: Tensor,
    eta_g: Tensor,
    g: Tensor,
    ts: float = 0.01,
    use_matlab_H: bool = False,
    mat_eng=None,
    H_config: dict = H_DEFAULT,
) -> (Tensor, Tensor):
    """
    Simulate IMU readings for acceleration and gyroscopic data based on position and orientation data.

    When `enable_H` is set to True, the function uses an external Matlab engine to compute
    acceleration and gyroscopic data, which does not retain the computational graph
    (i.e., `grad_fn` will not be present in the generated `acc` and `gyro` tensors).

    Parameters:
        p (Tensor): A tensor of shape (T, 3) representing position data over time.
        q (Tensor): A tensor of shape (T, 4) representing quaternion orientation data over time.
        ba (Tensor): A tensor of shape (3,) representing the accelerometer bias.
        bg (Tensor): A tensor of shape (3,) representing the gyroscope bias.
        eta_a (Tensor): A tensor of shape (T, 3) representing the accelerometer noise.
        eta_g (Tensor): A tensor of shape (T, 3,) representing the gyroscope noise.
        g (Tensor): A tensor of shape (3,) representing the gravity vector in the global frame.
        ts (float): The sampling rate in seconds. Default is 0.01.
        use_matlab_H (bool): If set to True, use Matlab engine for computation, which disables gradient computation.
        mat_eng: An active Matlab engine session for computation, if None, a new engine will be started.
        H_config (dict): Configuration for the Matlab engine IMU simulation. If None, defaults will be used.

    Returns:
        (Tensor, Tensor): A tuple containing:
                          - Acceleration data in the local frame (including gravity component),
                            as a Tensor with shape (T, 3).
                          - Gyroscopic data representing angular velocity in the local frame,
                            as a Tensor with shape (T, 3).

    Note:
        The tensors returned when `enable_H` is True will not have `grad_fn`, and thus
        will not be suitable for gradient-based optimization or backpropagation.
    """

    # Ensure it is a unit quaternion
    # q = normalize_quaternion(q)

    # Invert the quaternion to apply for transforming from global to local frame
    q_inv = rc.quaternion_invert(q)

    # Compute global acceleration by taking the second derivative of position
    acc_global = compute_derivative(p, dt=ts, n=2)

    # Transform global acceleration to the local frame
    acc_local = rc.quaternion_apply(q_inv, acc_global)

    # Transform gravity to the local frame
    g_local = rc.quaternion_apply(q_inv, g)

    # Compute the change in quaternions between consecutive time steps
    if len(q.shape) == 3:
        l_dq = rc.quaternion_multiply(q[:, 1:2], q_inv[:, 0:1])
        c_dq = rc.quaternion_multiply(q[:, 2:], q_inv[:, :-2])
        r_dq = rc.quaternion_multiply(q[:, -1:], q_inv[:, -2:-1])
        # print(l_dq.shape, c_dq.shape, r_dq.shape)
        dq = torch.cat([l_dq, c_dq, r_dq], dim=1)

    elif len(q.shape) < 3:
        l_dq = rc.quaternion_multiply(q[1], q_inv[0]).unsqueeze(0)
        # print(q[2:].shape)
        # print(q_inv[:-2].shape)
        c_dq = rc.quaternion_multiply(q[2:], q_inv[:-2])
        r_dq = rc.quaternion_multiply(q[-1], q_inv[-2]).unsqueeze(0)
        dq = torch.cat([l_dq, c_dq, r_dq])
    else:
        raise ValueError(
            "Input tensor must be either 1D (T, )/ 2D (T, D)/ 3D (N, T, D)"
        )

    # Convert the quaternion changes to euler angle changes
    dq_euler = rc.matrix_to_euler_angles(rc.quaternion_to_matrix(dq), "XYZ")

    # Account for the time step between measurements for the central differences
    dq_euler[1:-1] *= 0.5

    # Convert change in euler angles to angular velocity
    ang_vel = dq_euler / ts

    # When enable_H is True, use MATLAB engine to simulate IMU readings
    if use_matlab_H:
        import matlab.engine

        # If no MATLAB engine is provided, start a new MATLAB engine session
        if mat_eng is None:
            mat_eng = matlab.engine.start_matlab()
        # Set up the IMU configuration parameters in the MATLAB workspace
        mat_eng.workspace["accelParams"] = mat_eng.accelparams(
            *[
                item
                for pair in {
                    key: val
                    for key, val in H_config["acc_config"].items()
                    if val is not None
                }.items()
                for item in pair
            ]
        )
        mat_eng.workspace["gyroParams"] = mat_eng.gyroparams(
            *[
                item
                for pair in {
                    key: val
                    for key, val in H_config["gyro_config"].items()
                    if val is not None
                }.items()
                for item in pair
            ]
        )

        # Create the imuSensor object in MATLAB with the given configurations
        mat_eng.workspace["IMU"] = mat_eng.imuSensor(
            "accel-gyro",
            "ReferenceFrame",
            "NED",
            "SampleRate",
            1.0 / ts,
            "Accelerometer",
            mat_eng.workspace["accelParams"],
            "Gyroscope",
            mat_eng.workspace["gyroParams"],
        )

        # Convert acceleration and angular velocity for compatibility
        # https://uk.mathworks.com/matlabcentral/answers/1444929-wrong-gravitational-acceleration-in-imu-model-imusemsor-in-sensor-fusion-toolbox
        mat_eng.workspace["acc"] = -acc_global.detach().cpu().numpy()
        mat_eng.workspace["angVel"] = ang_vel.detach().cpu().numpy()
        mat_eng.workspace["orientation"] = mat_eng.quaternion(q.detach().cpu().numpy())

        # Run the MATLAB imuSensor simulation and retrieve the results
        acc_mat, gyro_mat = mat_eng.eval("IMU(acc, angVel, orientation)", nargout=2)

        # Convert MATLAB output back to PyTorch tensors and ensure they are on the same device as input
        acc = torch.tensor(acc_mat, device=p.device)
        gyro = torch.tensor(gyro_mat, device=p.device)

    else:
        # If enable_H is False, compute IMU readings directly in PyTorch

        # Subtract gravity to get the acceleration in the local frame.
        acc = acc_local - g_local
        # Transform angular velocity to the local frame
        gyro = rc.quaternion_apply(q_inv, ang_vel)  # # Add bias and noise too

        # Add bias and noise to the acceleration and gyroscopic data
        acc += ba + eta_a
        gyro += bg + eta_g

    return acc, gyro


def calc_rmse(
    predictions: npt.NDArray[float], targets: npt.NDArray[float]
) -> npt.NDArray[float]:
    assert (
        predictions.shape == targets.shape
    ), f"array shapes do not much: predictions.shape: {predictions.shape}, targets.shape: {targets.shape}"
    return np.sqrt(np.mean(np.square(predictions - targets)))


def power_penalty(angle, min_val, max_val, power=3) -> float:
    """
    Compute the penalty for an angle being outside the specified range.
    :param angle:
    :param min_val:
    :param max_val:
    :param power:
    :return:
    """
    if angle < min_val:
        return (min_val - angle) ** power
    if angle > max_val:
        return (angle - max_val) ** power
    return 0.0


def compute_derivative(x, dt: float = 1.0, n: int = 1):
    """
    Compute the n-th derivative of a sequence of data points using finite differences.

    For the first derivative, it uses a central difference method for interior data points and
    one-sided differences for the edges. For higher-order derivatives, it iteratively applies
    the difference computation.

    Parameters:
    x (torch.Tensor): A tensor containing the data points along the 0th dimension.
    d (float): The spacing between the data points. `d` stands for delta
    n (int, optional): The order of the derivative to compute. Default is 1 for first derivative.

    Returns:
    torch.Tensor: A tensor containing the n-th derivative of the input tensor.
    """
    assert (
        type(n) is int and n > 0
    ), "The order of the derivative must be positive integer."

    # Handle input of shape (N, T, D) N: batch size
    if len(x.shape) == 3:
        for _ in range(n):
            l_dx = (x[:, 1] - x[:, 0]).unsqueeze(1)
            c_dx = (x[:, 2:] - x[:, :-2]) / 2
            r_dx = (x[:, -1] - x[:, -2]).unsqueeze(1)
            x = torch.cat([l_dx, c_dx, r_dx], dim=1) / dt
    # Handle input of shape (T, D)
    elif len(x.shape) < 3:
        for _ in range(n):
            l_dx = (x[1] - x[0]).unsqueeze(0)
            c_dx = (x[2:] - x[:-2]) / 2
            r_dx = (x[-1] - x[-2]).unsqueeze(0)
            x = torch.cat([l_dx, c_dx, r_dx], dim=0) / dt
    else:
        raise ValueError(
            "Input tensor must be either 1D (T, )/ 2D (T, D)/ 3D (N, T, D)", x.shape
        )

    return x


def normalize_quaternion(quat):
    """
    Normalize a quaternion to have unit length.

    :param quat: (N, 4) tensor of quaternions
    :return:
    """
    norm = torch.sqrt(torch.norm(quat, p=2, dim=1, keepdim=True))
    quat_normalized = quat / norm
    return quat_normalized


def quaternion_slerp(q1, q2, t):
    """
    Perform Spherical Linear Interpolation (Slerp) between two quaternions.

    Args:
        q1: Starting quaternion (tensor of shape [4]).
        q2: Ending quaternion (tensor of shape [4]).
        t: Interpolation factor (0 <= t <= 1).

    Returns:
        Interpolated quaternion (tensor of shape [4]).
    """
    # Normalize both quaternions to ensure they represent rotations
    q1 = q1 / q1.norm()
    q2 = q2 / q2.norm()

    # Compute the dot product between q1 and q2
    dot_product = torch.dot(q1, q2)

    # If the dot product is negative, reverse one quaternion to ensure the shortest path
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product

    # Clamp dot product to stay within the domain of acos (i.e., between -1 and 1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute the angle between q1 and q2
    theta_0 = torch.acos(dot_product)  # Angle between q1 and q2
    sin_theta_0 = torch.sin(theta_0)

    if sin_theta_0 == 0.0:
        # q1 and q2 are identical, so return q1 (or q2, both are the same)
        return q1

    # Compute the interpolation factor for each quaternion
    sin_theta_t = torch.sin(t * theta_0)
    sin_theta_1_t = torch.sin((1 - t) * theta_0)

    # Interpolate
    slerped_quat = (sin_theta_1_t / sin_theta_0) * q1 + (sin_theta_t / sin_theta_0) * q2
    return slerped_quat


def interpolate_quaternions_slerp(quaternions, mask):
    """
    Interpolate missing quaternion values using Slerp.

    Args:
        quaternions: Tensor of quaternions (shape: N, 4).
        mask: Boolean mask indicating which quaternion values to interpolate.

    Returns:
        Interpolated quaternion tensor with missing values replaced.
    """
    nan_idx = torch.nonzero(mask).squeeze()  # Indices of masked (NaN) values
    valid_idx = torch.nonzero(~mask).squeeze()  # Indices of valid quaternions

    if len(nan_idx) == 0:
        return quaternions  # No NaNs to interpolate

    # Perform Slerp interpolation for each NaN index
    interpolated_quats = torch.clone(quaternions)

    for idx in nan_idx:
        # Find nearest valid quaternions before and after the missing index
        valid_before = valid_idx[valid_idx < idx]
        valid_after = valid_idx[valid_idx > idx]

        if len(valid_before) == 0:
            # No valid quaternions before this index, use the nearest after
            interpolated_quats[idx] = quaternions[valid_after.min()]
        elif len(valid_after) == 0:
            # No valid quaternions after this index, use the nearest before
            interpolated_quats[idx] = quaternions[valid_before.max()]
        else:
            # Interpolate using Slerp between the nearest valid quaternions
            before = valid_before.max()
            after = valid_after.min()

            # Calculate the interpolation parameter 't'
            t = (idx - before).float() / (after - before).float()

            # Perform Slerp interpolation between the valid quaternions
            interpolated_quats[idx] = quaternion_slerp(
                quaternions[before], quaternions[after], t
            )

    return interpolated_quats


# To deal with NaN when calc_rom_loss
def detect_exceeding_indices(values, threshold=torch.pi / 2, window_size=15):
    """
    Detect indices where the values exceed the given threshold.
    Mask surrounding indices within the window_size (e.g., ±15).

    Args:
        values: Tensor of values (in degrees).
        threshold: Threshold (in degrees) to detect exceeding values.
        window_size: Number of indices around exceeding values to mask.

    Returns:
        mask: A boolean mask of the same size as `values`, where True indicates
              that the index should be masked.
    """
    exceed_mask = values > threshold  # Detect where values exceed the threshold
    mask = torch.zeros_like(values, dtype=torch.bool)  # Initialize mask

    # For each exceeding value, mask the surrounding indices
    for idx in torch.nonzero(exceed_mask):
        start_idx = max(0, idx - window_size)
        end_idx = min(values.size(0), idx + window_size)
        mask[start_idx:end_idx] = True

    return mask


def create_capsule_shape(
    radius: float, height: float, position=POS_DEFAULT, orientation=ORI_DEFAULT
):
    """
    Create a capsule collision shape in PyBullet.
    :param radius: radius of the capsule
    :param height: height of the capsule
    :param position: position of the collision shape
    :param orientation: orientation of the collision shape

    :return: ID of the collision shape
    """

    return p.createCollisionShape(
        p.GEOM_CAPSULE,
        radius=radius,
        height=height,
        collisionFrameOrientation=orientation,
        collisionFramePosition=position,
    )


def sliding_window(x, y, window, stride, scheme="max"):
    data, target = [], []
    start = 0
    while start + window < x.shape[0]:
        end = start + window
        x_segment = x[start:end]
        if scheme == "last":
            # last scheme: : last observed label in the window determines the segment annotation
            y_segment = y[start:end][-1]
        elif scheme == "max":
            # max scheme: most frequent label in the window determines the segment annotation
            y_segment = np.argmax(np.bincount(y[start:end]))
        data.append(x_segment)
        target.append(y_segment)
        start += stride

    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.int64)

    return data, target


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


def resample(data, target, factor=2, verbose=False):
    if factor < 2:
        print("No resampling performed.")
        return data, target
    fs = 50
    cutoff = fs / (factor * 2)

    init_shapes = (data.shape, target.shape)

    data = butter_lowpass_filter(data, cutoff, fs)
    data = data[::factor]
    target = target[::factor]

    if verbose:
        print(
            f"Downsampled data from {init_shapes[0]} samples @ {fs}Hz => {data.shape} samples @ {fs/factor:.2f}Hz"
        )
        print(
            f"Downsampled labels from {init_shapes[1]} labels @ {fs}Hz => {target.shape} samples @ {fs/factor:.2f}Hz"
        )
    return data, target


def standardize(data, mean=None, std=None, verbose=False):
    """
    Standardizes all sensor channels

    :param data: numpy integer matrix (N, channels)
        Sensor data
    :param mean: (optional) numpy integer array (channels, )
        Array containing mean values for each sensor channel. When given, the mean is subtracted from the data.
    :param std: (optional) numpy integer array (channels, )
        Array containing the standard deviation of each sensor channel. When given, the data is divided by the standard deviation.
    :param verbose: bool
        Whether to print debug information
    :return:
    """
    if verbose:
        # I want to display the mean (given or calculated) here as the verbose message
        if mean is None or std is None:
            print("mean and std not specified. Calculating from data...")
            print(f"data - mean: {torch.mean(data, axis=0)}")
            print(f"data - std: {torch.std(data, axis=0)}")

    if mean is None:
        mean = torch.mean(data, axis=0)
    if std is None:
        std = torch.std(data, axis=0)
        std[std == 0] = 1

    if verbose:
        print(f"mean used: {mean}")
        print(f"std used: {std}")

    return (data - mean) / std


# Util functions for optimization
def generate_B_rp_range(
    B_rp_dict, min_scale=0.8, max_scale=1.2, near_zero=[-0.01, 0.01]
) -> Dict[str, npt.NDArray[float]]:
    """
    Generate a default range dict of B_rp values for optimization.
    :param B_rp_dict:
    :param min_scale:
    :param max_scale:
    :param near_zero:
    :return: Dict[str, npt.NDArray[float]]:
    """
    B_rp_range_dict = {}
    for k, v in B_rp_dict.items():
        B_rp_range_dict[k] = np.array(
            [sorted([e * min_scale, e * max_scale]) if e != 0 else near_zero for e in v]
        )
    return B_rp_range_dict


def generate_P_rp_range(P_rp_dict, rp_offset_m_range=(-0.05, 0.05)) -> Dict[str, npt.NDArray[float]]:
    """
    Generate a default range dict of P_rp values for optimization.

    :param P_rp_dict:
    :param rp_offset_m_range:
    :return: Dict[str, npt.NDArray[float]]:
    """
    rp_min_offset_m, rp_max_offset_m = rp_offset_m_range

    p_rp_range_dict = {
        key: np.array([[e + rp_min_offset_m, e + rp_max_offset_m] for e in rp])
        for key, rp in P_rp_dict.items()
    }
    return p_rp_range_dict


def generate_P_ro_range(P_ro_dict, ro_offset_deg_range=(-20, 20)) -> Dict[str, npt.NDArray[float]]:
    """
    Generate a default range dict of P_ro values for optimization.
    :param P_ro_dict:
    :param ro_offset_deg_range:
    :return:
    """
    ro_min_offset_deg, ro_max_offset_deg = ro_offset_deg_range
    p_ro_range_dict = {
        key: np.array(
            [
                [
                    e + np.deg2rad(ro_min_offset_deg),
                    e + np.deg2rad(ro_max_offset_deg),
                ]
                for e in ro
            ]
        )
        for key, ro in P_ro_dict.items()
    }
    return p_ro_range_dict


def generate_default_H_configs(imu_names: List[str]):
    """
    Generate default H configurations for each IMU in imu.
    :param imu_names: List of IMU names
    :return:
    """
    return {
        "ba": {imu_name: np.array([0.0, 0.0, 0.0]) for imu_name in imu_names},
        "bg": {imu_name: np.array([0.0, 0.0, 0.0]) for imu_name in imu_names},
        "sa": {imu_name: np.array([0.05, 0.05, 0.05]) for imu_name in imu_names},
        "sg": {imu_name: np.array([0.05, 0.05, 0.05]) for imu_name in imu_names},
        "sa_range_dict": {
            imu_name: np.array([(0.0, 0.3), (0.0, 0.3), (0.0, 0.3)])
            for imu_name in imu_names
        },
        "sg_range_dict": {
            imu_name: np.array([(0.0, 0.3), (0.0, 0.3), (0.0, 0.3)])
            for imu_name in imu_names
        },
    }
