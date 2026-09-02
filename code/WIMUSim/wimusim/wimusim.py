import warnings
import time

import matplotlib.pyplot as plt
import pathlib
import numpy as np
from collections import OrderedDict
from typing import Literal, Dict, List, Tuple, Set
import pybullet as p
from scipy.spatial.transform import Rotation as R

import pytorch3d.transforms.rotation_conversions as rc
import torch

from wimusim import consts as c
from wimusim import utils
#
from dataset_configs.digit_drawing_dataset import consts as digit_consts
from dataset_configs.realdisp import consts as realdisp_consts
from dataset_configs.h36m import consts as h36m_consts
from dataset_configs.xsens import consts as xsens_consts


class WIMUSim:

    class Environment:
        def __init__(
            self,
            g=c.E_DEFAULT["g"],
            data_type: Literal["numpy", "tensor"] = "tensor",
            device=None,
            requires_grad=False,
        ):
            """
            :param g: (x, y, z) tuple of gravity vector
            :param data_type: Data type of g (numpy or torch)
            :param device: Device to be used for torch
            :param requires_grad: requires_grad to be used for torch
            """
            utils.check_wimusim_param_args_consistency(data_type, device, requires_grad)
            self.data_type: Literal["numpy", "tensor"] = data_type
            self.device = device
            self.requires_grad = requires_grad

            # Check the data type of g and convert it so that it matches the data_type.
            if self.data_type == "numpy" and isinstance(g, torch.Tensor):
                self.g: np.ndarray = g.detach().cpu().numpy()
            elif self.data_type == "tensor" and isinstance(g, np.ndarray):
                self.g: torch.Tensor = torch.tensor(
                    g,
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=self.requires_grad,
                )
            elif self.data_type == "tensor" and isinstance(g, torch.Tensor):
                self.g: np.ndarray = g.to(self.device).requires_grad_(
                    self.requires_grad
                )
            else:
                self.g = g

        def __repr__(self):
            return f"E(g={self.g})"

        def __str__(self):
            return self.__repr__()

        def as_tensor(
            self, device=None, requires_grad=False
        ) -> Dict[str, torch.Tensor]:
            if self.data_type == "tensor":
                return {"g": self.g.to(device).requires_grad_(requires_grad)}

            elif self.data_type == "numpy":
                if device is None:
                    device = self.device
                if requires_grad is None:
                    requires_grad = self.requires_grad

                return {
                    "g": torch.tensor(
                        self.g, device=device, requires_grad=requires_grad
                    )
                }

            else:
                raise ValueError("Invalid data_type", self.data_type)

        def as_numpy(self) -> Dict[str, np.ndarray]:
            return {"g": np.array(self.g)}

    class Body:
        def __init__(
            self,
            rp: Dict[Tuple[str, str], np.ndarray] = c.B_DEFAULT["rp"],
            rp_range_dict: Dict[Tuple[str, str], np.ndarray] = None,
            rom_dict: Dict[str, np.ndarray] = None,
            child_parent_pairs: List[Tuple[str, str]] = None,
            device=None,
            requires_grad=False,
        ):
            """
            :param rp: dictionary of relative position vectors for each link
            :param rp_range_dict: dictionary of range of segment length for each joint pair
            :param rom_dict: dictionary of range of motion for each joint
            :param device = device to be used for torch
            :param requires_grad = requires_grad to be used for torch
            """
            if type(list(rp.keys())[0]) == str:
                # Convert the keys to tuple
                warnings.warn(
                    "Please use tuple (parent, child) for keys. Converting...",
                    UserWarning,
                )
                rp = {tuple(k.split("2")): v for k, v in rp.items()}

            # Set child_parent_pairs
            if child_parent_pairs is None:
                self.child_parent_pairs = [
                    (child, parent) for parent, child in rp.keys()
                ]
            self.validate_child_parent_pairs()

            self.device = device
            self.requires_grad = requires_grad

            self.rp = {
                k: (
                    torch.tensor(
                        v, device=self.device, requires_grad=self.requires_grad
                    )
                    if isinstance(v, np.ndarray)
                    else v.to(self.device).requires_grad_(self.requires_grad)
                )
                for k, v in rp.items()
            }

            if rp_range_dict is not None:
                self.rp_range_dict = {
                    k: torch.tensor(v, device=device, dtype=torch.float32)
                    for k, v in rp_range_dict.items()
                }
            else:
                self.rp_range_dict = None

            if rom_dict is not None:
                self.rom_dict = {
                    k: torch.tensor(v, device=device, dtype=torch.float32)
                    for k, v in rom_dict.items()
                }

            self.symmetry_key_pairs = self.get_symmetry_key_pairs()

        def __repr__(self):
            return f"B(rp={self.rp})"

        def __str__(self):
            return self.__repr__()

        def validate_child_parent_pairs(self):
            """
            Validates the child_parent_pairs. `child` (expect for "BASE") must have been appeared before as `parent`.
            This is expected in WIMUSim.measurement_func()
            :return: bool
            """

            # TODO: Ideally, we would like to find possible cycles in the child_parent_pairs.
            # First, we convert child_parent_pairs to a graph.
            # Then, we set child_parent_pairs by adding pairs from the root to the leaf.
            appeared = ["BASE"]
            for child, parent in self.child_parent_pairs:
                if parent not in appeared:
                    raise ValueError(
                        f"Invalid child_parent_pairs. {parent} must have been appeared before."
                    )
                else:
                    appeared.append(child)

        def get_symmetry_key_pairs(self):
            """
            Set the symmetry key pairs for the body. If there is no symmetry, set it to None.

            :return:
            """
            right_side_key_pairs = [key for key in self.rp.keys() if "R_" in key[1]]
            if len(right_side_key_pairs) == 0:
                return None

            left_side_key_pairs = []
            for right_side_key_pair in right_side_key_pairs:
                left_side_key_pair = (
                    right_side_key_pair[0].replace("R_", "L_"),
                    right_side_key_pair[1].replace("R_", "L_"),
                )
                if left_side_key_pair in self.rp.keys():
                    left_side_key_pairs.append(left_side_key_pair)
                else:
                    warnings.warn(
                        f"Symmetry key pair {left_side_key_pair} does not exist."
                    )
                    # remove that key pair from the right_side_key_pairs
                    right_side_key_pairs.remove(right_side_key_pair)
            if len(left_side_key_pairs) == 0:
                return None

            return list(zip(right_side_key_pairs, left_side_key_pairs))

        def as_tensor(
            self, device=None, requires_grad=False
        ) -> Dict[str, Dict[str, torch.Tensor]]:
            if self.data_type == "tensor":
                return {
                    "rp": {
                        k: v.to(device).requires_grad_(requires_grad)
                        for k, v in self.rp.items()
                    },
                }

            elif self.data_type == "numpy":
                if device is None:
                    device = self.device
                if requires_grad is None:
                    requires_grad = self.requires_grad

                return {
                    "rp": {
                        k: torch.tensor(v, device=device, requires_grad=requires_grad)
                        for k, v in self.rp.items()
                    },
                }

            else:
                raise ValueError("Invalid data_type", self.data_type)

        def as_numpy(self) -> Dict[str, Dict[str, np.ndarray]]:
            if self.data_type == "tensor":
                return {
                    "rp": {k: v.detach().cpu().numpy() for k, v in self.rp.items()},
                    "ro": {k: v.detach().cpu().numpy() for k, v in self.ro.items()},
                }
            elif self.data_type == "numpy":
                return {"rp": self.rp, "ro": self.ro}
            else:
                raise ValueError("Invalid data_type", self.data_type)

    class Placement:
        """
        Placement of the IMUs on the body.

        """

        def __init__(
            self,
            rp: Dict[tuple, np.ndarray],
            ro: Dict[tuple, np.ndarray],
            rp_range_dict: Dict[tuple, np.ndarray] = None,
            ro_range_dict: Dict[tuple, np.ndarray] = None,
            device=None,
            requires_grad=False,
        ):
            """
            :param rp: dictionary of relative position vectors for each link
            :param ro: dictionary of relative orientation vectors for each link
            :param device = device to be used for torch
            :param requires_grad = requires_grad to be used for torch
            """

            if type(list(rp.keys())[0]) == str:
                # Convert the keys to tuple
                warnings.warn(
                    "Please use tuple (parent, child) for keys. Converting...",
                    UserWarning,
                )
                rp = {tuple(k.split("2")): v for k, v in rp.items()}

            if type(list(ro.keys())[0]) == str:
                # Convert the keys to tuple
                warnings.warn(
                    "Please use tuple (parent, child) for keys. Converting...",
                    UserWarning,
                )
                ro = {tuple(k.split("2")): v for k, v in ro.items()}

            self.device = device
            self.requires_grad = requires_grad

            self.imu_names = list([key[1] for key in rp.keys()])
            self.joint_imu_pairs = list(rp.keys())

            # Check the data type of rp and ro; and convert it so that it matches the data_type.

            self.rp = {
                k: (
                    torch.tensor(
                        v, device=self.device, requires_grad=self.requires_grad
                    )
                    if isinstance(v, np.ndarray)
                    else v.to(self.device).requires_grad_(self.requires_grad)
                )
                for k, v in rp.items()
            }
            self.ro = {
                k: (
                    torch.tensor(
                        v, device=self.device, requires_grad=self.requires_grad
                    )
                    if isinstance(v, np.ndarray)
                    else v.to(self.device).requires_grad_(self.requires_grad)
                )
                for k, v in ro.items()
            }

            self.set_rp_range_dict(rp_range_dict)
            self.set_ro_range_dict(ro_range_dict)

        def __repr__(self):
            return f"P(rp={self.rp}, ro={self.ro})"

        def __str__(self):
            return self.__repr__()

        def set_rp_range_dict(self, rp_range_dict: Dict[tuple, np.ndarray]):
            if rp_range_dict is not None:
                self.rp_range_dict = {
                    k: torch.tensor(v, device=self.device, dtype=torch.float32)
                    for k, v in rp_range_dict.items()
                }
            else:
                self.rp_range_dict = None

        def set_ro_range_dict(self, ro_range_dict: Dict[tuple, np.ndarray]):
            if ro_range_dict is not None:
                self.ro_range_dict = {
                    k: torch.tensor(v, device=self.device, dtype=torch.float32)
                    for k, v in ro_range_dict.items()
                }
            else:
                self.ro_range_dict = None

        def as_numpy(self) -> Dict[str, Dict[str, np.ndarray]]:
            return {
                "rp": {k: v.detach().cpu().numpy() for k, v in self.rp.items()},
                "ro": {k: v.detach().cpu().numpy() for k, v in self.ro.items()},
            }

    class Dynamics:
        def __init__(
            self,
            orientation: Dict[str, torch.Tensor],  # Required
            sample_rate: float,
            translation: Dict[str, torch.Tensor] = None,
            data_type: Literal["numpy", "tensor"] = "tensor",
            device=None,
            requires_grad=False,
        ):
            """
            :param orientation: dictionary of orientation for each link
            :param sample_rate: sample rate of the data
            :param translation: dictionary of translation the base link
            :param data_type: Data type of translation and orientation (numpy or torch)
            :param device = device to be used for torch
            :param requires_grad = requires_grad to be used for torch
            """
            utils.check_wimusim_param_args_consistency(data_type, device, requires_grad)

            self.translation = {}
            self.orientation = {}
            self.data_type = data_type
            self.device = device
            self.requires_grad = requires_grad
            self.sample_rate = sample_rate

            # Determine the shape of the data
            if len(next(iter(orientation.values())).shape) == 2:
                # When the shape of D.translation["XYZ"]  is (T, 3), batch_size = 1
                self.batch_size: int = 1
            elif len(next(iter(orientation.values())).shape) == 3:
                # When the shape of D.translation["XYZ"] is (N, T, 3), then batch_size = N
                self.batch_size: int = next(iter(orientation.values())).shape[0]
            self.n_samples: int = next(iter(orientation.values())).shape[-2]

            # Check the data type of translation and orientation and convert them so that they match the data_type.
            if data_type == "tensor":
                for k, v in orientation.items():
                    if isinstance(v, np.ndarray):
                        self.orientation[k] = torch.tensor(
                            v,
                            device=self.device,
                            requires_grad=self.requires_grad,
                        )
                    elif isinstance(v, torch.Tensor):
                        # self.orientation[k] = v.to(self.device)
                        self.orientation[k] = v.to(self.device).requires_grad_(
                            self.requires_grad
                        )
                    else:
                        raise ValueError("Invalid data type for translation", type(v))

                if translation is not None:
                    for k, v in translation.items():
                        if isinstance(v, np.ndarray):
                            self.translation[k] = torch.tensor(
                                v,
                                device=self.device,
                                requires_grad=self.requires_grad,
                            )
                        elif isinstance(v, torch.Tensor):
                            self.translation[k] = v.to(
                                self.device,
                            ).requires_grad_(self.requires_grad)
                        else:
                            raise ValueError(
                                "Invalid data type for translation", type(v)
                            )
                else:
                    # When translation is not given, initialize it with zeros
                    # If batch_size is 1, then the shape is (T, 3)
                    # If batch_size is N (>1), then the shape is (N, T, 3)
                    if self.batch_size == 1:
                        self.translation = {
                            "XYZ": torch.zeros(
                                (self.n_samples, 3),
                                device=self.device,
                                requires_grad=self.requires_grad,
                            )
                        }
                    else:
                        self.translation = {
                            "XYZ": torch.zeros(
                                (self.batch_size, self.n_samples, 3),
                                device=self.device,
                                requires_grad=self.requires_grad,
                            )
                        }

            elif data_type == "numpy":
                warnings.warn("data_type='numpy' will be deprecated.", UserWarning)
                for k, v in translation.items():
                    if isinstance(v, np.ndarray):
                        self.translation[k] = v
                    elif isinstance(v, torch.Tensor):
                        self.translation[k] = v.detach().cpu().numpy()
                    else:
                        raise ValueError("Invalid data type for translation", type(v))

                for k, v in orientation.items():
                    if isinstance(v, np.ndarray):
                        self.orientation[k] = v
                    elif isinstance(v, torch.Tensor):
                        self.orientation[k] = v.detach().cpu().numpy()
                    else:
                        raise ValueError("Invalid data type for translation", type(v))
            else:
                raise ValueError("Invalid data_type", data_type)

        def __repr__(self):
            return f"D(translation={self.translation}, orientation={self.orientation})"

        def __str__(self):
            return self.__repr__()

        def as_tensor(
            self, device=None, requires_grad=False
        ) -> Dict[str, Dict[str, torch.Tensor]]:
            if device is None:
                device = self.device
            if requires_grad is None:
                requires_grad = self.requires_grad

            if self.data_type == "tensor":
                return {
                    "translation": {
                        k: v.to(device).requires_grad_(requires_grad)
                        for k, v in self.translation.items()
                    },
                    "orientation": {
                        k: v.to(device).requires_grad_(requires_grad)
                        for k, v in self.orientation.items()
                    },
                }
            elif self.data_type == "numpy":
                return {
                    "translation": {
                        k: v.to(device).requires_grad_(requires_grad)
                        for k, v in self.translation.items()
                    },
                    "orientation": {
                        k: v.to(device).requires_grad_(requires_grad)
                        for k, v in self.orientation.items()
                    },
                }

        def as_numpy(self) -> Dict[str, Dict[str, np.ndarray]]:
            if self.data_type == "tensor":
                return {
                    "translation": {
                        k: v.detach().cpu().numpy() for k, v in self.translation.items()
                    },
                    "orientation": {
                        k: v.detach().cpu().numpy() for k, v in self.orientation.items()
                    },
                }
            elif self.data_type == "numpy":
                return {
                    "translation": self.translation,
                    "orientation": self.orientation,
                }

    class Hardware:
        """
        Hardware parameters for the IMU sensors. Currently, constant bias (3D vector) and noise (N(0, \sigma)) are supported.
        Each 6-axis IMU has 6 parameters: 3 for accelerometer and 3 for gyroscope.

        Unless specified, the default values (bias=[0., 0., 0.] and sigma=0) are used.

        """

        def __init__(
            self,
            ba: dict = None,
            bg: dict = None,
            sa: dict = None,
            sg: dict = None,
            sa_range_dict: dict = None,
            sg_range_dict: dict = None,
            device: str = "cpu",
            requires_grad=True,
        ):
            """

            :param ba: dictionary of accelerometer bias (imu_name: torch.Tensor (3,))
            :param bg: dictionary of gyroscope bias (imu_name: torch.Tensor (3,))
            :param sa: dictionary of accelerometer noise (imu_name: torch.Tensor (3,))
            :param ga: dictionary of gyroscope noise (imu_name: torch.Tensor (3,))
            :param device: device to be used for torch
            """

            # Check if ba, sa, bg, sg all have the same keys
            if not (ba.keys() == sa.keys() == bg.keys() == sg.keys()):
                raise ValueError(
                    "ba, sa, bg, sg must have the same keys. Check the keys of the dictionaries."
                )

            self.ba = {
                imu_name: torch.tensor(
                    acc_bias,
                    device=device,
                    requires_grad=requires_grad,
                    dtype=torch.float32,
                )
                for imu_name, acc_bias in ba.items()
            }  # imu_name: torch.Tensor (3,)
            self.sa = {
                imu_name: torch.tensor(
                    acc_noise_std,
                    device=device,
                    requires_grad=requires_grad,
                    dtype=torch.float32,
                )
                for imu_name, acc_noise_std in sa.items()
            }  # imu_name: torch.Tensor (3,)
            # acc_noise_dict = dict() # The length of the data should be determined before initializing it

            self.bg = {
                imu_name: torch.tensor(
                    gyro_bias,
                    device=device,
                    requires_grad=requires_grad,
                    dtype=torch.float32,
                )
                for imu_name, gyro_bias in bg.items()
            }  # imu_name: torch.Tensor (3,)
            self.sg = {
                imu_name: torch.tensor(
                    gyro_noise_std,
                    device=device,
                    requires_grad=requires_grad,
                    dtype=torch.float32,
                )
                for imu_name, gyro_noise_std in sg.items()
            }  # imu_name: torch.Tensor (3,)

            if sa_range_dict is not None:
                self.sa_range_dict = {
                    imu_name: torch.tensor(
                        sa_range,
                        device=device,
                        dtype=torch.float32,
                    )
                    for imu_name, sa_range in sa_range_dict.items()
                }
            else:
                self.sa_range_dict = None

            if sg_range_dict is not None:
                self.sg_range_dict = {
                    imu_name: torch.tensor(
                        sg_range,
                        device=device,
                        dtype=torch.float32,
                    )
                    for imu_name, sg_range in sg_range_dict.items()
                }
            else:
                self.sg_range_dict = None

            self.device = device

            # Will be initialized in init_noise_dict() after the shape of D is determined
            self.eta_a = None
            self.eta_g = None

        def init_noise_dict(self, shape: tuple):
            """
            Initialize the noise dictionary with the given number of samples.
            :param shape:
            :return:
            """
            self.eta_a = {
                imu_name: self.gen_3d_noise(shape, acc_noise_std.data)
                for imu_name, acc_noise_std in self.sa.items()
            }
            self.eta_g = {
                imu_name: self.gen_3d_noise(shape, gyro_noise_std.data)
                for imu_name, gyro_noise_std in self.sg.items()
            }

        def gen_3d_noise(
            self, shape, std: torch.tensor, required_grad=True
        ) -> torch.Tensor:
            """
            Generate 3-channel noise.
            :param shape: shape of the noise (N, T, 3) or (T, 3)
            :param std: standard deviation of the noise for each axis (torch.Tensor (3,))
            :param required_grad: requires_grad to be used for torch
            :return:
            """
            return (
                torch.randn(
                    size=shape,
                    device=self.device,
                    dtype=torch.float32,
                )
                * std
            ).requires_grad_(required_grad)

        def __repr__(self):
            return f"H(ba={self.ba}, sa={self.sa}, bg={self.bg}, sg={self.sg})"

        def __str__(self):
            return self.__repr__()

        def to_dict(self) -> dict:
            return vars(self)

    class HardwareMatlab:
        """
        Hardware parameters for the IMU sensors. This cannot be used when parameterizing real wearable IMU data.
        """

        def __init__(
            self,
            acc_config: dict = c.H_DEFAULT["acc_config"],
            gyro_config: dict = c.H_DEFAULT["gyro_config"],
        ):
            """
            :param acc_config: dictionary of accelerometer parameters
            :param gyro_config: dictionary of gyroscope parameters
            """
            self.acc_config = acc_config
            self.gyro_config = gyro_config

        def __repr__(self):
            return f"H(accelerometer={self.acc_config}, gyroscope={self.gyro_config})"

        def __str__(self):
            return self.__repr__()

        def to_dict(self) -> dict:
            return vars(self)

    def __init__(
        self,
        B,
        D,
        P,
        H,
        E=Environment(),
        device=None,
        dataset_name: Literal[
            "H3.6M", "REALDISP", "DigitDrawingDataset", "RealWorld", "XSENS"
        ] = None,
        realtime_sim=False,
        enable_translation=True,
        visualization_prec_level=25,
        verbose=False,
    ):
        """

        :param E: Environment parameters
        :param B: Body parameters
        :param P: Placement parameters
        :param D: Dynamics parameters
        :param H: Hardware parameters
        :param device: Device to be used for torch
        :param dataset_name: Name of the dataset
        :param realtime_sim: Enable realtime simulation
        :param enable_translation: Enable translation of the joints
        :param visualization_prec_level: Precision level of the visualization. The higher, the better, but it will be slower
        """
        if isinstance(E, dict):
            self.E = self.Environment(**E)
        else:
            self.E = E
        if isinstance(B, dict):
            self.B = self.Body(**B)
        else:
            self.B = B
        if isinstance(P, dict):
            self.P = self.Placement(**P)
        else:
            self.P = P
        if isinstance(D, dict):
            self.D = self.Dynamics(**D)
        else:
            self.D = D
        if isinstance(H, dict):
            self.H = self.Hardware(**H)
        else:
            self.H = H

        # Initialize the noise dictionary for acc and gyro
        if self.D.batch_size > 1:
            self.H.init_noise_dict((self.D.batch_size, self.D.n_samples, 3))
        else:
            self.H.init_noise_dict((self.D.n_samples, 3))

        self.dataset_name = dataset_name
        self.verbose = verbose
        # For PyBullet visualization
        self.realtime_sim = realtime_sim
        self.translation_enabled = enable_translation
        self.pybullet_client_id = -1  # PyBullet client ID
        self.humanoid_id = -1  # PyBullet humanoid ID
        self.visualization_prec_level = visualization_prec_level
        self.humanoid_params = (
            c.HUMANOID_PARAMS_DEFAULT
        )  # To be set in launch_pybullet_client()

        # TODO: The keys of joint_info should be readable names, not PyBullet defined names.
        self.joint_info: dict = {}  # PyBullet defined joint_name -> joint_id
        self.link_joint_id_dict: dict = {}  # User defined link_name -> joint_id

        self.p_IMU_obs_pybullet_dict: dict = {}  # IMU name -> p_IMU_obs
        self.q_IMU_obs_pybullet_dict: dict = {}  # IMU name -> q_IMU_obs

        self.simulated_IMU_dict: dict = {}  # IMU name -> simulated IMU (acc, gyro)

        # TODO: Change this somehow. Current implementation are too dataset specific
        if dataset_name is not None:
            # Show warning that dataset_name is to be removed
            warnings.warn("dataset_name is to be removed", UserWarning)

        if dataset_name == "DigitDrawingDataset":
            # Show warning

            # The joint pairs of the humanoid (This should be given by Body)
            self.joint_child_parent_pairs = digit_consts.JOINT_CHILD_PARENT_PAIRS
            # Note: From now on, switch to self.B.child_parent_pairs as much as possible

            assert self.B.child_parent_pairs == self.joint_child_parent_pairs
            # These params are required for placing IMUs on the humanoid
            self.imu_names = digit_consts.IMU_NAMES
            self.imu_sizes = digit_consts.IMU_SIZES
            self.imu_colors = digit_consts.IMU_COLORS
            self.joint_link_dict = digit_consts.JOINT_WIMUSIM_LINK_DICT

            # The step size of the simulation p.stepSimulation()
            self.sampling_rate = digit_consts.SAMPLING_RATE

            self.joint_imu_pairs = digit_consts.JOINT_IMU_PAIRS

        elif dataset_name == "REALDISP":
            # These params are required for placing IMUs on the humanoid
            self.imu_names = realdisp_consts.IMU_NAMES  # This should be given by P
            self.imu_sizes = realdisp_consts.IMU_SIZES
            self.imu_colors = realdisp_consts.IMU_COLORS
            self.joint_link_dict = realdisp_consts.JOINT_WIMUSIM_LINK_DICT

            # The step size of the simulation p.stepSimulation()
            self.sampling_rate = realdisp_consts.SAMPLING_RATE

            self.joint_imu_pairs = realdisp_consts.JOINT_IMU_PAIRS

            # TODO: Check if the D params are in quaternion for REALDISP

        elif dataset_name == "H3.6M":
            self.joint_link_dict = h36m_consts.JOINT_WIMUSIM_LINK_DICT  #
            self.sampling_rate = 30  # 30 Hz
            self.imu_sizes = realdisp_consts.IMU_SIZES
            self.imu_colors = realdisp_consts.IMU_COLORS
            warnings.warn(
                "H3.6M dataset is not configured yet. May not work at some point",
                UserWarning,
            )
        elif dataset_name == "XSENS":
            self.joint_child_parent_pairs = xsens_consts.JOINT_CHILD_PARENT_PAIRS
            self.imu_names = xsens_consts.IMU_NAMES
            self.imu_sizes = xsens_consts.IMU_SIZES
            self.imu_colors = xsens_consts.IMU_COLORS
            self.joint_link_dict = xsens_consts.JOINT_WIMUSIM_LINK_DICT
            self.sampling_rate = self.D.sample_rate
            self.joint_imu_pairs = xsens_consts.JOINT_IMU_PAIRS
        else:
            self.joint_link_dict = h36m_consts.JOINT_WIMUSIM_LINK_DICT  #
            self.imu_sizes = realdisp_consts.IMU_SIZES
            self.imu_colors = realdisp_consts.IMU_COLORS

            #
            self.sampling_rate = self.D.sample_rate  # 30 Hz

            # raise ValueError("Invalid dataset_name", dataset_name)
            # warnings.warn(
            #     "Dataset name is not given. Use default humanoid model for PyBullet.",
            #     UserWarning,
            # )

        self.timestep = 1.0 / self.sampling_rate / self.visualization_prec_level

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                if self.verbose:
                    print("Using GPU:", torch.cuda.get_device_name(0))
            else:
                self.device = torch.device("cpu")
                if self.verbose:
                    print("CUDA is not available. Using CPU.")
        else:
            self.device = device

        # Validation
        self.validate_B_params()  # TODO: Move these in the __init__ of each class
        self.validate_P_params()  # TODO: Move these in the __init__ of each class

    def close_pybullet_client(self):
        """
        Closes the PyBullet client.
        :return:
        """
        if self.pybullet_client_id != -1:
            p.disconnect(self.pybullet_client_id)
            self.pybullet_client_id = -1
            self.humanoid_id = -1
        else:
            warnings.warn("PyBullet client is not connected. Ignoring...", UserWarning)

    def launch_pybullet_client(self):
        if self.pybullet_client_id != -1:
            warnings.warn(
                "PyBullet client is already connected."
                "Close the current pybullet client with .close_pybullet_client() first. Ignoring...",
                UserWarning,
            )

            return

        if self.dataset_name == "XSENS":
            raise NotImplementedError(
                "XSENS dataset config currently supports simulation and parameterisation, "
                "but PyBullet visualization is not implemented for the 23-segment skeleton."
            )

        self.pybullet_client_id = p.connect(p.GUI)
        p.setGravity(*self.E.g)
        if self.dataset_name == "REALDISP":
            self.convert_BP_to_humanoid_params()
        elif self.dataset_name == "DigitDrawingDataset":
            self.set_humanoid_params()
            print("Use default humanoid params.")
        elif self.dataset_name == "H3.6M":
            self.convert_BP_to_humanoid_params()
        else:
            self.convert_BP_to_humanoid_params()
            # raise ValueError("Invalid dataset_name", self.dataset_name)

        # Deploy the humanoid in the PyBullet environment
        self.deploy_humanoid()
        p.setRealTimeSimulation(0)
        p.setTimeStep(self.timestep)
        warnings.WarningMessage(
            f"TimeStep is currently set {self.timestep} sec to avoid unstable movement with the spherical joint",
            category=UserWarning,
            filename="",
            lineno=0,
        )

    def validate_D_params(self):
        for key in self.D.orientation.keys():
            assert (
                key in self.joint_link_dict.keys()
            ), f"Invalid joint name: D.orientation[{key}]"

    def validate_B_params(self):
        """
        Validates the B parameters.
        Confirm that child_parent_pairs are consistent with the B.rp keys.
        :return:
        """
        for parent_joint, child_joint in self.B.rp.keys():
            assert (
                child_joint,
                parent_joint,
            ) in self.B.child_parent_pairs, f"Invalid joint pair: {parent_joint} -> {child_joint}. Check the B.child_parent_pairs."

    def validate_P_params(self):
        """
        Validates the P parameters.
        :return:
        """
        assert (
            self.P.rp.keys() == self.P.ro.keys()
        ), "Inconsistent keys for P parameters."
        for parent_joint, child_IMU in self.P.rp.keys():
            assert (
                parent_joint in self.joint_link_dict.keys()
            ), f"Invalid joint name: {parent_joint}"
            assert child_IMU in self.P.imu_names, f"Invalid IMU name: {child_IMU}"

    def set_humanoid_params(self, params: OrderedDict = c.HUMANOID_PARAMS_DEFAULT):
        """
        Sets the parameters of the humanoid.
        param params: Parameters defining the humanoid's body parts.
        :return:
        """
        # This params should be obtained by self.B.
        # TODO: Some function like self.convert_B_to_humanoid_params() should be implemented
        # TODO: add some validation
        self.humanoid_params = params

    def convert_BP_to_humanoid_params(self):
        """
        Converts B and P params to humanoid_params.
        convert_B_and_P
        :return:
        """
        # TODO: Implement this function
        if self.dataset_name == "DigitDrawingDataset":
            # NOTE: Currently, the y-axis of the R_CLAVICLE2R_SHOULDER is not reflected to the humanoid_params.
            self.humanoid_params["torso_1"]["length"] = (
                self.B.rp["PELVIS2R_CLAVICLE"][2] / 2.0
            )
            self.humanoid_params["torso_2"]["length"] = (
                self.B.rp["PELVIS2R_CLAVICLE"][2] / 2.0
            )
            self.humanoid_params["right_clavicle"]["length"] = self.B.rp[
                "R_CLAVICLE2R_SHOULDER"
            ][0]
            self.humanoid_params["right_upperarm"]["length"] = self.B.rp[
                "R_SHOULDER2R_ELBOW"
            ][0]

            self.humanoid_params["torso"]["radius"] = self.P.rp["PELVIS2TRS"][1]
            self.humanoid_params["right_upperarm"]["radius"] = self.P.rp[
                "R_SHOULDER2RUA"
            ][2]
            self.humanoid_params["right_lowerarm"]["radius"] = self.P.rp["R_ELBOW2RWR"][
                2
            ]

        elif self.dataset_name in ["REALDISP"]:
            # B params
            self.humanoid_params["torso_1"]["length"] = (
                self.B.rp[("PELVIS", "R_CLAVICLE")][2] / 2.0
            )
            self.humanoid_params["torso_2"]["length"] = (
                self.B.rp[("PELVIS", "R_CLAVICLE")][2] / 2.0
            )
            self.humanoid_params["right_clavicle"]["length"] = self.B.rp[
                ("R_CLAVICLE", "R_SHOULDER")
            ][0]
            self.humanoid_params["right_upperarm"]["length"] = self.B.rp[
                ("R_SHOULDER", "R_ELBOW")
            ][0]
            self.humanoid_params["left_clavicle"]["length"] = -self.B.rp[
                ("L_CLAVICLE", "L_SHOULDER")
            ][0]
            self.humanoid_params["left_upperarm"]["length"] = -self.B.rp[
                ("L_SHOULDER", "L_ELBOW")
            ][0]
            self.humanoid_params["pelvis"]["length"] = (
                self.B.rp[("BASE", "R_HIP")][0] - self.B.rp[("BASE", "L_HIP")][0]
            ) * 1.5
            self.humanoid_params["right_upperleg"]["length"] = -self.B.rp[
                ("R_HIP", "R_KNEE")
            ][2]
            self.humanoid_params["right_upperleg"]["length"] = -self.B.rp[
                ("L_HIP", "L_KNEE")
            ][2]

            # P params
            # TODO: Refactor this part. P shouldn't affect the humanoid's radius.
            # self.humanoid_params["torso_1"]["radius"] = torch.tensor(0.05, device=self.device)  # -self.P.rp[("PELVIS", "BACK")][1]
            self.humanoid_params["right_upperarm"]["radius"] = self.P.rp[
                ("R_SHOULDER", "RUA")
            ][2]
            self.humanoid_params["right_lowerarm"]["radius"] = self.P.rp[("R_ELBOW", "RLA")][
                2
            ]
            self.humanoid_params["left_upperarm"]["radius"] = self.P.rp[
                ("L_SHOULDER", "LUA")
            ][2]
            self.humanoid_params["left_lowerarm"]["radius"] = self.P.rp[("L_ELBOW", "LLA")][
                2
            ]
            self.humanoid_params["right_upperleg"]["radius"] = self.P.rp[("R_HIP", "RT")][1]
            self.humanoid_params["right_lowerleg"]["radius"] = self.P.rp[("R_KNEE", "RC")][1]
            self.humanoid_params["left_upperleg"]["radius"] = self.P.rp[("L_HIP", "LT")][1]
            self.humanoid_params["left_lowerleg"]["radius"] = self.P.rp[("L_KNEE", "LC")][1]
        # elif self.dataset_name in ["H3.6M"]:
        else:
            # B params
            self.humanoid_params["torso_1"]["length"] = self.B.rp[("PELVIS", "BELLY")][
                2
            ]
            self.humanoid_params["torso_2"]["length"] = self.B.rp[("BELLY", "NECK")][2]
            self.humanoid_params["right_clavicle"]["length"] = self.B.rp[
                ("R_CLAVICLE", "R_SHOULDER")
            ][0]
            self.humanoid_params["right_upperarm"]["length"] = self.B.rp[
                ("R_SHOULDER", "R_ELBOW")
            ][0]
            self.humanoid_params["right_lowerarm"]["length"] = self.B.rp[
                ("R_ELBOW", "R_WRIST")
            ][0]
            self.humanoid_params["left_clavicle"]["length"] = -self.B.rp[
                ("L_CLAVICLE", "L_SHOULDER")
            ][0]
            self.humanoid_params["left_upperarm"]["length"] = -self.B.rp[
                ("L_SHOULDER", "L_ELBOW")
            ][0]
            self.humanoid_params["left_lowerarm"]["length"] = -self.B.rp[
                ("L_ELBOW", "L_WRIST")
            ][0]
            self.humanoid_params["pelvis"]["length"] = (
                self.B.rp[("BASE", "R_HIP")][0] - self.B.rp[("BASE", "L_HIP")][0]
            ) * 1.5
            self.humanoid_params["right_upperleg"]["length"] = -self.B.rp[
                ("R_HIP", "R_KNEE")
            ][2]
            self.humanoid_params["right_lowerleg"]["length"] = -self.B.rp[
                ("R_KNEE", "R_ANKLE")
            ][2]
            self.humanoid_params["left_upperleg"]["length"] = -self.B.rp[
                ("L_HIP", "L_KNEE")
            ][2]
            self.humanoid_params["left_lowerleg"]["length"] = -self.B.rp[
                ("L_KNEE", "L_ANKLE")
            ][2]
        # else:
        #     raise ValueError("Invalid dataset_name", self.dataset_name)

    def run_visualization(
        self, pause: float = 0.0, record_video: bool = False, filepath: str = "out.mp4"
    ):
        """
        Runs the visualization of the simulation.
        :param pause:
        :param record_video:
        :param filepath:
        :return:
        """
        if self.pybullet_client_id == -1:
            raise ValueError("PyBullet client is not connected")

        # Initialize the observation arrays
        self.p_IMU_obs_pybullet_dict = {
            imu_name: np.zeros((self.D.n_samples, 3)) for imu_name in self.P.imu_names
        }
        self.q_IMU_obs_pybullet_dict = {
            imu_name: np.zeros((self.D.n_samples, 4)) for imu_name in self.P.imu_names
        }

        if record_video:
            assert filepath.endswith(".mp4"), "Invalid file extension. Must be .mp4"
            assert pathlib.Path(filepath).parent.exists(), "Invalid directory path"
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, filepath)

        # Convert the quaternion to PyBullet's format
        joint_params_xyzw = {
            k: v[:, [1, 2, 3, 0]].detach().cpu().numpy()
            for k, v in self.D.orientation.items()
            # if v.shape[1] == 4  # Quaternion form
        }

        joint_indices = [
            self.link_joint_id_dict[wimusim_link_name]
            for (_, wimusim_link_name) in self.joint_link_dict.items()
        ]
        target_positions = np.array(
            [
                joint_params_xyzw[joint_name]
                for (joint_name, _) in self.joint_link_dict.items()
            ]
        )

        # Specify the joint target positions and step the simulation for each frame
        for i in range(self.D.n_samples):
            p.setJointMotorControlMultiDofArray(
                self.humanoid_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_positions[:, i, :],
            )
            if self.translation_enabled:
                p.setJointMotorControlArray(
                    self.humanoid_id,
                    jointIndices=[
                        self.link_joint_id_dict[link_name]
                        for link_name in ["trans_x", "trans_y", "trans_z"]
                    ],
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=self.D.translation["XYZ"][i, :],
                )
            [
                p.stepSimulation()
                for _ in range(int(1 / self.sampling_rate / self.timestep))
            ]
            time.sleep(pause)

            # Store the IMU positions and orientations for each frame
            self.update_IMU_pq_observation(frame_idx=i)

        if record_video:
            p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)

    def simulate(
        self, mode: Literal["parameterise", "generate"] = "parameterise"
    ) -> Dict[str, torch.Tensor]:
        """
        Simulates the IMU data.

        :param mode: parameterise or generate.
            In "parameterise" mode, hardware noise H.eta_a, H_eta_b are directly used.
            In "generate" mode, the hardware noise are generated based on the H.sa and H.sg.
        :return:
        """

        self.simulated_IMU_dict = {}  # Initialize the simulated IMU dict

        resolved_joint_pose_dict = {
            "BASE": (
                self.D.translation["XYZ"],
                self.D.orientation["BASE"],
            )
        }
        resolved_imu_pose_dict = {}

        # Resolve pose (position and orientation) of humanoid's joints
        for child_name, parent_name in self.B.child_parent_pairs:
            resolved_joint_pose_dict[child_name] = utils.resolve_child_pose(
                resolved_joint_pose_dict[parent_name][0],
                resolved_joint_pose_dict[parent_name][1],
                self.B.rp[(parent_name, child_name)],
                self.D.orientation[child_name],
            )

        for joint_name, imu_name in self.P.joint_imu_pairs:
            resolved_imu_pose_dict[imu_name] = utils.resolve_child_pose(
                resolved_joint_pose_dict[joint_name][0],
                resolved_joint_pose_dict[joint_name][1],
                self.P.rp[(joint_name, imu_name)],
                self.P.ro[(joint_name, imu_name)],
                child_ori_type="euler",
            )

        for imu_name, (p_IMU, q_IMU) in resolved_imu_pose_dict.items():
            acc_bias = self.H.ba[imu_name]
            gyro_bias = self.H.bg[imu_name]

            if mode == "parameterise":
                # Use the bias and noise parameters directly
                acc_noise = self.H.eta_a[imu_name]
                gyro_noise = self.H.eta_g[imu_name]

                self.simulated_IMU_dict[imu_name] = utils.simulate_imu(
                    p_IMU,
                    q_IMU,
                    ba=acc_bias,
                    bg=gyro_bias,
                    eta_a=acc_noise,
                    eta_g=gyro_noise,
                    g=self.E.g,
                    ts=1 / self.sampling_rate,
                )

                # Update the std parameters based on the current noise
                self.H.sa[imu_name] = self.H.eta_a[imu_name].std(dim=0)
                self.H.sg[imu_name] = self.H.eta_a[imu_name].std(dim=0)

            elif mode == "generate":
                # Generate the noise for acc and gyro based on the std parameter
                if self.D.batch_size > 1:
                    # New code hasn't been tested yet. Delete this if that works.
                    # acc_noise = self.H.gen_3d_noise(
                    #     (self.D.batch_size, self.D.n_samples, 3), self.H.sa[imu_name]
                    # )
                    # gyro_noise = self.H.gen_3d_noise(
                    #     (self.D.batch_size, self.D.n_samples, 3), self.H.sg[imu_name]
                    # )
                    acc_noise = (
                        torch.randn(
                            size=(self.D.batch_size, self.D.n_samples, 3),
                            device=self.device,
                        )
                        * self.H.sa[imu_name]
                    )
                    gyro_noise = (
                        torch.randn(
                            size=(self.D.batch_size, self.D.n_samples, 3),
                            device=self.device,
                        )
                        * self.H.sg[imu_name]
                    )
                else:
                    acc_noise = (
                        torch.randn(size=(self.D.n_samples, 3), device=self.device)
                        * self.H.sa[imu_name]
                    )
                    gyro_noise = (
                        torch.randn(size=(self.D.n_samples, 3), device=self.device)
                        * self.H.sg[imu_name]
                    )

                self.simulated_IMU_dict[imu_name] = utils.simulate_imu(
                    p_IMU,
                    q_IMU,
                    ba=acc_bias,
                    bg=gyro_bias,
                    eta_a=acc_noise,
                    eta_g=gyro_noise,
                    g=self.E.g,
                    ts=1 / self.sampling_rate,
                )

        return self.simulated_IMU_dict

    def update_IMU_pq_observation(self, frame_idx):
        """
        Updates the IMU position and orientation observations in PyBullet.
        :param frame_idx: Frame index
        :return:
        """
        imus_link_states_i = p.getLinkStates(
            self.humanoid_id,
            linkIndices=[
                self.link_joint_id_dict[imu_name] for imu_name in self.P.imu_names
            ],
        )
        for imu_name, imu_link_state in zip(self.P.imu_names, imus_link_states_i):
            self.p_IMU_obs_pybullet_dict[imu_name][frame_idx] = imu_link_state[0]
            self.q_IMU_obs_pybullet_dict[imu_name][frame_idx] = imu_link_state[1]

    def deploy_humanoid(self):
        """
        Deploys the humanoid in the PyBullet environment.
        :return:
        """

        enable_translation = self.translation_enabled

        if self.pybullet_client_id == -1:
            raise ValueError("PyBullet client is not connected")
        if self.humanoid_id != -1:
            raise ValueError("Humanoid is already deployed")

        humanoid_params = self.humanoid_params

        # List of humanoid body parts
        link_list = list(c.HUMANOID_PARAMS_DEFAULT.keys())

        # Initialize an ordered dictionary to store humanoid specifications
        humanoid_specs = OrderedDict()  # This specifies info passed p.createMultiBody()

        # Assign IDs and default link info to each body part
        for i, link_name in enumerate(link_list):
            humanoid_specs[link_name] = c.DEFAULT_LINK_INFO.copy()
            humanoid_specs[link_name]["id"] = i + 1  # col_id = 0 for the base

        # Set parent ID for each link of the humanoid
        for c_link_name, p_link_name in c.CHILD_PARENT_LINK_PAIRS:
            if p_link_name == "base":
                humanoid_specs[c_link_name]["parent_id"] = 0
            else:
                humanoid_specs[c_link_name]["parent_id"] = humanoid_specs[p_link_name][
                    "id"
                ]
        # print(humanoid_specs)

        # Set joint types for each link
        for link in link_list:
            if link in [f"trans_{axis}" for axis in "xyz"]:
                if enable_translation:
                    humanoid_specs[link]["joint_type"] = p.JOINT_PRISMATIC
                    if link == "trans_x":
                        humanoid_specs[link]["joint_axis"] = (1, 0, 0)
                    elif link == "trans_y":
                        humanoid_specs[link]["joint_axis"] = (0, 1, 0)
                    elif link == "trans_z":
                        humanoid_specs[link]["joint_axis"] = (0, 0, 1)
                    else:
                        raise ValueError("Invalid link name")
                else:
                    pass  # Use default value (p.JOINT_FIXED)
            elif link not in [
                # "pelvis",  # Comment out to allow rotation for adjusting the facing direction
                "head",
                "right_hand",
                "left_hand",
                "right_foot",
                "left_foot",
            ]:
                humanoid_specs[link]["joint_type"] = p.JOINT_SPHERICAL
                humanoid_specs[link]["joint_axis"] = (1, 1, 1)
            else:
                pass  # Use default value (p.JOINT_FIXED)

        # Rotation of 90 degrees along X-axis
        # noinspection PyArgumentList
        q_x90 = R.from_euler("XYZ", [0, 90, 0], degrees=True).as_quat()
        buff = 0.01  # Buffer between joints

        # Create base collision shape
        # Collision frame position specifies the position of the collision shape relative to the associated joint's frame.
        # E.g. when the collisionFramePosition is (0, 0, 0), the collision shape's center is same as the joint's frame.
        col_base_id = p.createCollisionShape(
            p.GEOM_SPHERE, radius=0.01
        )  # reserve id = 0 for base

        # Create collision shapes for prismatic joints
        col_prismatic = p.createCollisionShape(
            p.GEOM_SPHERE, radius=humanoid_params["trans_x"]["radius"]
        )  # reuse the same one for all prismatic joints

        # Example: Creating Pelvis Collision Shape
        humanoid_specs["pelvis"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["pelvis"]["radius"],
            height=humanoid_params["pelvis"]["length"]
            - humanoid_params["pelvis"]["radius"] * 2,
            collisionFrameOrientation=q_x90,
        )

        torso_1_link_len = humanoid_params["torso_1"]["length"] - (
            humanoid_params["pelvis"]["radius"] + buff
        )
        torso_1_link_offset = (
            humanoid_params["pelvis"]["radius"] + torso_1_link_len / 2 + buff
        )
        humanoid_specs["torso_1"]["col_id"] = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=torso_1_link_len/2,
            collisionFramePosition=(0, 0, torso_1_link_offset),
        )
        torso_2_link_len = humanoid_params["torso_2"]["length"]
        torso_2_link_offset = torso_2_link_len / 2 + buff
        humanoid_specs["torso_2"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["torso_2"]["radius"],
            height=torso_2_link_len - 2 * humanoid_params["torso_2"]["radius"],
            collisionFramePosition=(0, 0, torso_2_link_offset),
        )

        humanoid_specs["head"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["head"]["radius"],
            height=humanoid_params["head"]["length"],
            collisionFramePosition=(
                0,
                0,
                humanoid_params["head"]["radius"]
                + humanoid_params["head"]["length"] / 2
                + buff,
            ),
        )

        humanoid_specs["right_clavicle"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["right_clavicle"]["radius"],
            height=humanoid_params["right_clavicle"]["radius"],  # Correct
            collisionFramePosition=(
                humanoid_params["torso_2"]["radius"]
                + humanoid_params["right_clavicle"]["radius"]
                + buff,
                0,
                0,
            ),
            collisionFrameOrientation=q_x90,
        )

        humanoid_specs["right_upperarm"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["right_upperarm"]["radius"],
            height=humanoid_params["right_upperarm"]["length"]
            - humanoid_params["right_upperarm"]["radius"] * 2,
            collisionFramePosition=(
                humanoid_params["right_upperarm"]["length"] / 2,
                0,
                0,
            ),
            collisionFrameOrientation=q_x90,
        )

        humanoid_specs["right_lowerarm"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["right_lowerarm"]["radius"],
            height=humanoid_params["right_lowerarm"]["length"]
            - humanoid_params["right_lowerarm"]["radius"] * 2
            - buff,
            collisionFramePosition=(
                humanoid_params["right_lowerarm"]["length"] / 2 + buff,
                0,
                0,
            ),
            collisionFrameOrientation=q_x90,
        )

        humanoid_specs["right_hand"]["col_id"] = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                humanoid_params["right_hand"]["length"] / 2 - buff,
                humanoid_params["right_hand"]["width"] / 2,
                humanoid_params["right_hand"]["depth"] / 2,
            ],
            collisionFramePosition=(
                humanoid_params["right_hand"]["length"] / 2 + buff,
                0,
                0,
            ),
        )

        humanoid_specs["left_clavicle"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["left_clavicle"]["radius"],
            height=humanoid_params["left_clavicle"]["radius"],  # Correct
            collisionFramePosition=(
                -(
                    humanoid_params["torso_2"]["radius"]
                    + humanoid_params["left_clavicle"]["radius"]
                    + buff
                ),
                0,
                0,
            ),
            collisionFrameOrientation=q_x90,
        )

        humanoid_specs["left_upperarm"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["left_upperarm"]["radius"],
            height=humanoid_params["left_upperarm"]["length"]
            - humanoid_params["left_upperarm"]["radius"] * 2,
            collisionFramePosition=(
                -humanoid_params["left_upperarm"]["length"] / 2,
                0,
                0,
            ),
            collisionFrameOrientation=q_x90,
        )

        humanoid_specs["left_lowerarm"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["left_lowerarm"]["radius"],
            height=humanoid_params["left_lowerarm"]["length"]
            - humanoid_params["left_lowerarm"]["radius"] * 2
            - buff,
            collisionFramePosition=(
                -(humanoid_params["left_lowerarm"]["length"] / 2 + buff),
                0,
                0,
            ),
            collisionFrameOrientation=q_x90,
        )

        humanoid_specs["left_hand"]["col_id"] = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                humanoid_params["left_hand"]["length"] / 2 - buff,
                humanoid_params["left_hand"]["width"] / 2,
                humanoid_params["left_hand"]["depth"] / 2,
            ],
            collisionFramePosition=(
                -(humanoid_params["left_hand"]["length"] / 2 + buff),
                0,
                0,
            ),
        )

        right_upperleg_link_len = humanoid_params["right_upperleg"]["length"] - (
            humanoid_params["right_upperleg"]["radius"] + buff
        )
        right_upperleg_link_offset = (
            humanoid_params["pelvis"]["radius"] + right_upperleg_link_len / 2
        )
        humanoid_specs["right_upperleg"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["right_upperleg"]["radius"],
            height=right_upperleg_link_len
            - humanoid_params["right_upperleg"]["radius"] * 2,
            collisionFramePosition=(0, 0, -right_upperleg_link_offset),
        )

        humanoid_specs["right_lowerleg"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["right_lowerleg"]["radius"],
            height=humanoid_params["right_lowerleg"]["length"]
            - humanoid_params["right_lowerleg"]["radius"] * 2
            - buff,
            collisionFramePosition=(
                0,
                0,
                -(humanoid_params["right_lowerleg"]["length"] / 2 + buff),
            ),
        )

        humanoid_specs["right_foot"]["col_id"] = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                humanoid_params["right_foot"]["width"] / 2,
                humanoid_params["right_foot"]["length"] / 2,
                humanoid_params["right_foot"]["depth"] / 2 - buff,
            ],
            collisionFramePosition=(
                0,
                humanoid_params["right_foot"]["length"] / 3,
                -humanoid_params["right_foot"]["depth"] + buff,
            ),
        )

        left_upperleg_link_len = humanoid_params["left_upperleg"]["length"] - (
            humanoid_params["left_upperleg"]["radius"] + buff
        )
        left_upperleg_link_offset = (
            humanoid_params["pelvis"]["radius"] + left_upperleg_link_len / 2
        )
        humanoid_specs["left_upperleg"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["left_upperleg"]["radius"],
            height=left_upperleg_link_len
            - humanoid_params["left_upperleg"]["radius"] * 2,
            collisionFramePosition=(0, 0, -left_upperleg_link_offset),
        )

        humanoid_specs["left_lowerleg"]["col_id"] = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=humanoid_params["left_lowerleg"]["radius"],
            height=humanoid_params["left_lowerleg"]["length"]
            - humanoid_params["left_lowerleg"]["radius"] * 2
            - buff,
            collisionFramePosition=(
                0,
                0,
                -(humanoid_params["left_lowerleg"]["length"] / 2 + buff),
            ),
        )

        humanoid_specs["left_foot"]["col_id"] = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                humanoid_params["left_foot"]["width"] / 2,
                humanoid_params["left_foot"]["length"] / 2,
                humanoid_params["left_foot"]["depth"] / 2 - buff,
            ],
            collisionFramePosition=(
                0,
                humanoid_params["left_foot"]["length"] / 3,
                -humanoid_params["left_foot"]["depth"] + buff,
            ),
        )

        # Specify the joint starting positions
        # Example: Specifying Head Joint Position
        humanoid_specs["torso_2"]["position"] = (
            0.0,
            0.0,
            humanoid_params["torso_1"]["length"],
        )
        humanoid_specs["head"]["position"] = (
            0.0,
            0.0,
            humanoid_params["torso_2"]["length"],
        )
        humanoid_specs["right_clavicle"]["position"] = (
            0.0,
            0.0,
            humanoid_params["torso_2"]["length"],
        )
        humanoid_specs["right_upperarm"]["position"] = (
            humanoid_params["right_clavicle"]["length"],
            0,
            0,
        )
        humanoid_specs["right_lowerarm"]["position"] = (
            humanoid_params["right_upperarm"]["length"],
            0,
            0,
        )
        humanoid_specs["right_hand"]["position"] = (
            humanoid_params["right_lowerarm"]["length"],
            0,
            0,
        )
        humanoid_specs["left_clavicle"]["position"] = (
            0.0,
            0.0,
            humanoid_params["torso_2"]["length"],
        )
        humanoid_specs["left_upperarm"]["position"] = (
            -humanoid_params["left_clavicle"]["length"],
            0,
            0,
        )
        humanoid_specs["left_lowerarm"]["position"] = (
            -humanoid_params["left_upperarm"]["length"],
            0,
            0,
        )
        humanoid_specs["left_hand"]["position"] = (
            -humanoid_params["left_lowerarm"]["length"],
            0,
            0,
        )
        humanoid_specs["right_upperleg"]["position"] = (
            humanoid_params["pelvis"]["length"] / 4,
            0,
            0,
        )  # TODO: Check this parameter
        humanoid_specs["right_lowerleg"]["position"] = (
            0,
            0,
            -humanoid_params["right_upperleg"]["length"],
        )
        humanoid_specs["right_foot"]["position"] = (
            0,
            0,
            -humanoid_params["right_lowerleg"]["length"],
        )
        humanoid_specs["left_upperleg"]["position"] = (
            -humanoid_params["pelvis"]["length"] / 4,
            0,
            0,
        )
        humanoid_specs["left_lowerleg"]["position"] = (
            0,
            0,
            -humanoid_params["left_upperleg"]["length"],
        )
        humanoid_specs["left_foot"]["position"] = (
            0,
            0,
            -humanoid_params["left_lowerleg"]["length"],
        )

        # Place IMUs on the humanoid
        # List of IMU names
        imu_specs = OrderedDict()  # This specifies info passed p.createMultiBody()
        last_link_key = next(reversed(humanoid_specs))
        P = self.P.as_numpy()  # Ensure that the P parameters are in numpy format
        # noinspection PyArgumentList
        for i, P_key in enumerate(P["ro"].keys()):
            joint_name, imu_name = P_key
            imu_specs[imu_name] = c.DEFAULT_LINK_INFO.copy()
            imu_specs[imu_name]["id"] = i + 1 + humanoid_specs[last_link_key]["id"]

            q_IMU = R.from_euler("XYZ", P["ro"][P_key], degrees=False).as_quat()
            imu_specs[imu_name]["col_id"] = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=self.imu_sizes[i] / 2 - 0.001,
                collisionFrameOrientation=q_IMU,
            )
            imu_specs[imu_name]["vis_id"] = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=self.imu_sizes[i] / 2,
                visualFrameOrientation=q_IMU,
                rgbaColor=self.imu_colors[i],
            )

            imu_specs[imu_name]["parent_id"] = humanoid_specs[
                self.joint_link_dict[joint_name]
            ]["id"]
            imu_specs[imu_name]["position"] = P["rp"][P_key]
            imu_specs[imu_name]["inertial_frame_orientation"] = q_IMU

        # Deploy the humanoid body specified by humanoid_specs in the PyBullet environment
        base_specs = c.DEFAULT_LINK_INFO.copy()
        base_specs["mass"] = 0

        # The IDs of the links are assigned in the order of the keys of humanoid_specs
        self.humanoid_id = p.createMultiBody(
            baseMass=base_specs["mass"],
            baseCollisionShapeIndex=base_specs["col_id"],
            baseVisualShapeIndex=base_specs["visual_shape_id"],
            basePosition=base_specs["position"],
            baseOrientation=base_specs["orientation"],
            baseInertialFramePosition=base_specs["inertial_frame_position"],
            baseInertialFrameOrientation=base_specs["inertial_frame_orientation"],
            linkMasses=[humanoid_specs[name]["mass"] for name in humanoid_specs.keys()]
            + [imu_specs[name]["mass"] for name in imu_specs.keys()],
            linkCollisionShapeIndices=[
                humanoid_specs[name]["col_id"] for name in humanoid_specs.keys()
            ]
            + [imu_specs[name]["col_id"] for name in imu_specs.keys()],
            linkVisualShapeIndices=[
                humanoid_specs[name]["visual_shape_id"]
                for name in humanoid_specs.keys()
            ]
            + [imu_specs[name]["vis_id"] for name in imu_specs.keys()],
            linkPositions=[
                humanoid_specs[name]["position"] for name in humanoid_specs.keys()
            ]
            + [imu_specs[name]["position"] for name in imu_specs.keys()],
            linkOrientations=[
                humanoid_specs[name]["orientation"] for name in humanoid_specs.keys()
            ]
            + [imu_specs[name]["orientation"] for name in imu_specs.keys()],
            linkInertialFramePositions=[
                humanoid_specs[name]["inertial_frame_position"]
                for name in humanoid_specs.keys()
            ]
            + [imu_specs[name]["inertial_frame_position"] for name in imu_specs.keys()],
            linkInertialFrameOrientations=[
                humanoid_specs[name]["inertial_frame_orientation"]
                for name in humanoid_specs.keys()
            ]
            + [
                imu_specs[name]["inertial_frame_orientation"]
                for name in imu_specs.keys()
            ],
            linkParentIndices=[
                humanoid_specs[name]["parent_id"] for name in humanoid_specs.keys()
            ]
            + [imu_specs[name]["parent_id"] for name in imu_specs.keys()],
            linkJointTypes=[
                humanoid_specs[name]["joint_type"] for name in humanoid_specs.keys()
            ]
            + [imu_specs[name]["joint_type"] for name in imu_specs.keys()],
            linkJointAxis=[
                humanoid_specs[name]["joint_axis"] for name in humanoid_specs.keys()
            ]
            + [imu_specs[name]["joint_axis"] for name in imu_specs.keys()],
        )
        self.set_joint_info(
            humanoid_specs, imu_specs
        )  # Associate the understandable names of the joints with the joint IDs.

    def set_joint_info(self, humanoid_specs: dict, imu_specs) -> None:
        """
        Associate the understandable names of the joints with the joint IDs.

        :param humanoid_specs: dict of humanoid specifications generated in deploy_humanoid()
        :param imu_specs: dict of IMU specifications generated in deploy_humanoid()
        :return:
        """
        # Check if the humanoid is already deployed
        assert self.humanoid_id != -1, "Humanoid is not deployed yet"
        assert self.humanoid_params != OrderedDict(), "Humanoid parameters are not set"
        joint_info = OrderedDict(
            {
                joint_info[1].decode("utf-8"): {
                    "id": joint_info[0],
                    "joint_name_p": joint_info[1].decode(
                        "utf-8"
                    ),  # PyBullet defined joint_name
                    "joint_type": joint_info[2],
                    "joint_lower_limit": joint_info[8],
                    "joint_upper_limit": joint_info[9],
                    "link_name_p": joint_info[12].decode(
                        "utf-8"
                    ),  # PyBullet defined link_name
                    "parent_id": joint_info[16],
                }
                for joint_info in [
                    p.getJointInfo(
                        self.humanoid_id, i, physicsClientId=self.pybullet_client_id
                    )
                    for i in range(
                        p.getNumJoints(
                            self.humanoid_id, physicsClientId=self.pybullet_client_id
                        )
                    )
                ]
            }
        )

        for spec, vals in humanoid_specs.items():
            joint_info[f"joint{vals['id']}"]["link_name_u"] = spec
        for spec, vals in imu_specs.items():
            joint_info[f"joint{vals['id']}"]["link_name_u"] = spec

        self.joint_info = joint_info
        self.link_joint_id_dict = {
            v["link_name_u"]: v["id"] for v in joint_info.values()
        }

    def plot_simulated_imu(
        self,
        imu_name: str,
        target_imu_dict: dict = None,
        start: int = 0,
        end: int = None,
        batch_idx: int = 0,
    ):
        """
        Plots the simulated IMU measurements.

        :param imu_name: imu_name to plot (must be one of the self.imu_names)
        :param target_imu_dict: target IMU measurements to compare with
        :param start: start index
        :param end: end index. plot the entire sequence if None
        :return:
        """
        # validate the imu_name
        assert imu_name in self.imu_names

        fig, axs = plt.subplots(3, 2, figsize=(15, 3))
        fig.tight_layout()

        acc_sim, gyro_sim = self.simulated_IMU_dict[imu_name]

        if self.D.batch_size > 1:
            acc_sim = acc_sim[batch_idx]
            gyro_sim = gyro_sim[batch_idx]

        # Convert to numpy array
        if isinstance(acc_sim, torch.Tensor):
            acc_sim = acc_sim.detach().cpu().numpy()
        if isinstance(gyro_sim, torch.Tensor):
            gyro_sim = gyro_sim.detach().cpu().numpy()
        if end is None:
            end = self.D.n_samples

        # Prepare the target IMU measurements if target_imu_dict is passed
        if target_imu_dict is not None:
            acc_target, gyro_target = target_imu_dict[imu_name]
            if isinstance(acc_target, torch.Tensor):
                acc_target = acc_target.detach().cpu().numpy()
            if isinstance(gyro_target, torch.Tensor):
                gyro_target = gyro_target.detach().cpu().numpy()

            axs[0][0].set_title(
                f"{imu_name} Acc Comparison (RMSE: {utils.calc_rmse(acc_target, acc_sim):.3f})"
            )
            axs[0][1].set_title(
                f"{imu_name} Gyro Comparison (RMSE: {utils.calc_rmse(gyro_target, gyro_sim):.3f})"
            )
        else:
            axs[0][0].set_title(f"{imu_name} Acc")
            axs[0][1].set_title(f"{imu_name} Gyro")

        # Plot the IMU measurements
        for i in range(3):
            if target_imu_dict is not None:
                axs[i][0].plot(acc_target[start:end, i], label="target")
                axs[i][1].plot(gyro_target[start:end, i], label="target")

            axs[i][0].plot(acc_sim[start:end, i], label="sim")
            axs[i][1].plot(gyro_sim[start:end, i], label="sim")

        axs[0][0].legend()
        axs[0][1].legend()
        plt.show()
