import numpy as np
from collections import OrderedDict

DATASET_NAMES = ["DigitDrawingDataset", "REALDISP", "XSENS"]

QUAT_NUMPY_FORMAT = ("W", "X", "Y", "Z")
QUAT_MATLAB_FORMAT = QUAT_NUMPY_FORMAT  # a + bi + cj + dk
QUAT_PYTORCH3D_FORMAT = QUAT_NUMPY_FORMAT

QUAT_SCIPY_FORMAT = ("X", "Y", "Z", "W")
QUAT_PYBULLET_FORMAT = QUAT_SCIPY_FORMAT
QUAT_ROMA_FORMAT = QUAT_SCIPY_FORMAT

POS_DEFAULT = (0.0, 0.0, 0.0)
ORI_DEFAULT = (0.0, 0.0, 0.0, 1.0)  # Unit quaternion in XYZW format

E_DEFAULT = {"g": np.array([0, 0, -9.8])}

# default body params (unit: m)
# This is for DigitDrawingDataset
B_DEFAULT = {
    "rp": {
        "world2pelvis": np.array([0.0, 0.0, 0.0]),
        "pelvis2shoulder": np.array([0.22, -0.09, 0.43]),
        "shoulder2elbow": np.array([0.27, 0.0, 0.0]),
    },
    "ro": {
        "world2pelvis": np.array([0.0, 0.0, 0.0, 1.0]),
        "pelvis2shoulder": np.array([0.0, 0.0, 0.0, 1.0]),
        "shoulder2elbow": np.array([0.0, 0.0, 0.0, 1.0]),
    },
}

# For matlab's IMU simulation model
H_DEFAULT = {
    "acc_config": {
        "MeasurementRange": None,
        "Resolution": None,
        "ConstantBias": None,
        "AxesMisalignment": None,
        "NoiseDensity": None,
        "BiasInstabilityCoefficients": None,
        "BiasInstability": None,
        "RandomWalk": None,
        "NoiseType": "double-sided",
        "TemperatureBias": None,
        "TemperatureScaleFactor": None,
    },
    "gyro_config": {
        "MeasurementRange": None,
        "Resolution": None,
        "ConstantBias": None,
        "AxesMisalignment": None,
        "NoiseDensity": None,
        "BiasInstabilityCoefficients": None,
        "BiasInstability": None,
        "RandomWalk": None,
        "NoiseType": "double-sided",
        "TemperatureBias": None,
        "TemperatureScaleFactor": None,
        "AccelerationBias": None,
    },
}

_B_NEAR_ZERO = [-0.01, 0.01]
B_RANGE_DEFAULT = {
    "rp": {
        "BASE2PELVIS": np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, _B_NEAR_ZERO]),
        "PELVIS2R_CLAVICLE": np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, [0.40, 0.60]]),
        "R_CLAVICLE2R_SHOULDER": np.array([[0.15, 0.35], [-0.10, 0.00], _B_NEAR_ZERO]),
        "R_SHOULDER2R_ELBOW": np.array([[0.20, 0.30], _B_NEAR_ZERO, _B_NEAR_ZERO]),
        "PELVIS2L_CLAVICLE": np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, [0.40, 0.60]]),
        "L_CLAVICLE2L_SHOULDER": np.array(
            [[-0.35, -0.15], [-0.10, 0.00], _B_NEAR_ZERO]
        ),
        "L_SHOULDER2L_ELBOW": np.array([[-0.30, -0.20], _B_NEAR_ZERO, _B_NEAR_ZERO]),
        "BASE2R_HIP": np.array([[0.10, 0.20], _B_NEAR_ZERO, _B_NEAR_ZERO]),
        "R_HIP2R_KNEE": np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, [-0.50, -0.30]]),
        "BASE2L_HIP": np.array([[-0.20, -0.10], _B_NEAR_ZERO, _B_NEAR_ZERO]),
        "L_HIP2L_KNEE": np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, [-0.50, -0.30]]),
    },
    "ro": {},
}

# The following consts are for visualization with PyBullet
XSENS_IMU_SIZE = (0.058, 0.058, 0.033)

# Related to visualization with PyBullet
DEFAULT_LINK_INFO = {
    "mass": 0.1,
    "col_id": 0,
    "visual_shape_id": -1,
    "position": POS_DEFAULT,
    "orientation": ORI_DEFAULT,
    "inertial_frame_position": POS_DEFAULT,
    "inertial_frame_orientation": ORI_DEFAULT,
    "parent_id": 0,
    "joint_type": 4, # p.JOINT_FIXED,
    "joint_axis": (0, 0, 0),
}

# Related to visualization with PyBullet
# TODO: This should be moved to realdisp.consts
IMU_PARAMS_DEFAULT = ["BACK", "RUA", "RLA", "LUA", "LLA", "RT", "RC", "LT", "LC"]

ROM_RANGE_ADJ = [-20, 20]  # ADJ: Adjustment\
# "JOINT_NAME": np.array([[min, max], [min, max], [min, max]])
JOINT_ROM_DEFAULT = {
    "BASE": np.deg2rad(
        np.array([[-180, 180], [-90, 90], [-180, 180]])
    ),  # No limit for the base
    "PELVIS": np.deg2rad(
        np.array([[-60, 30], [-50, 50], [-45, 45]])
    ),  # np.deg2rad(np.array([[-45, 30], [-50, 50], [-40, 40]])), # default values
    "BELLY": np.deg2rad(
        np.array([[-45, 30], [-40, 40], [-45, 45]])
    ),  # Needs to be adjusted
    "NECK": np.deg2rad(np.array([[-50, 60], [-60, 60], [-50, 50]])),
    "R_CLAVICLE": np.deg2rad(np.array([ROM_RANGE_ADJ, [-20, +10], [-20, 20]])),
    "L_CLAVICLE": np.deg2rad(np.array([ROM_RANGE_ADJ, [-10, +20], [-20, 20]])),
    "R_SHOULDER": np.deg2rad(np.array([[-140, 90], [-90, 90], [-30, 135]])),
    "L_SHOULDER": np.deg2rad(np.array([[-140, 90], [-90, 90], [-135, 30]])),
    "R_ELBOW": np.deg2rad(np.array([[-90, 90], ROM_RANGE_ADJ, [-5, 145]])),
    "L_ELBOW": np.deg2rad(np.array([[-90, 90], ROM_RANGE_ADJ, [-145, 5]])),
    "R_HIP": np.deg2rad(np.array([[-15, 125], [-45, 20], [-45, 45]])),
    "L_HIP": np.deg2rad(np.array([[-15, 125], [-20, 45], [-45, 45]])),
    "R_KNEE": np.deg2rad(np.array([[-130, 0], ROM_RANGE_ADJ, ROM_RANGE_ADJ])),
    "L_KNEE": np.deg2rad(np.array([[-130, 0], ROM_RANGE_ADJ, ROM_RANGE_ADJ])),
    # Keep it as it is
    "NOSE": np.deg2rad(np.array([ROM_RANGE_ADJ, ROM_RANGE_ADJ, ROM_RANGE_ADJ])),
    "HEAD": np.deg2rad(np.array([ROM_RANGE_ADJ, ROM_RANGE_ADJ, ROM_RANGE_ADJ])),
    "R_WRIST": np.deg2rad(np.array([ROM_RANGE_ADJ, ROM_RANGE_ADJ, ROM_RANGE_ADJ])),
    "L_WRIST": np.deg2rad(np.array([ROM_RANGE_ADJ, ROM_RANGE_ADJ, ROM_RANGE_ADJ])),
    "R_ANKLE": np.deg2rad(np.array([ROM_RANGE_ADJ, ROM_RANGE_ADJ, ROM_RANGE_ADJ])),
    "L_ANKLE": np.deg2rad(np.array([ROM_RANGE_ADJ, ROM_RANGE_ADJ, ROM_RANGE_ADJ])),
}


SYMMETRY_KEY_PAIRS = [
    ("PELVIS2R_CLAVICLE", "PELVIS2L_CLAVICLE"),
    ("R_CLAVICLE2R_SHOULDER", "L_CLAVICLE2L_SHOULDER"),
    ("R_SHOULDER2R_ELBOW", "L_SHOULDER2L_ELBOW"),
    ("BASE2R_HIP", "BASE2L_HIP"),
    ("R_HIP2R_KNEE", "L_HIP2L_KNEE"),
]



# Related to visualization with PyBullet
trans_prismatic_joint_radius = 0.00001
HUMANOID_PARAMS_DEFAULT = OrderedDict(
    {
        # trans_{xyz} are for realizing the translation of the humanoid
        "trans_x": {
            # p.GEOM_PRISMATIC
            "radius": trans_prismatic_joint_radius,
        },
        "trans_y": {
            # p.GEOM_PRISMATIC
            "radius": trans_prismatic_joint_radius,
        },
        "trans_z": {
            # p.GEOM_PRISMATIC
            "radius": trans_prismatic_joint_radius,
        },
        "pelvis": {
            # p.GEOM_CAPSULE
            "length": 0.40,
            "radius": 0.08,
        },
        "torso_1": {
            # p.GEOM_CAPSULE
            "length": 0.25,
            "radius": 0.10,
        },
        "torso_2": {
            # p.GEOM_CAPSULE
            "length": 0.25,
            "radius": 0.10,
        },
        "head": {
            # p.GEOM_CAPSULE
            "radius": 0.10,
            "length": 0.05,  # Check this value later
        },
        "right_clavicle": {
            # p.GEOM_CAPSULE
            "radius": 0.025,
            "length": 0.20,
        },
        "right_upperarm": {
            # p.GEOM_CAPSULE
            "radius": 0.04,
            "length": 0.25,
        },
        "right_lowerarm": {
            # p.GEOM_CAPSULE
            "radius": 0.03,
            "length": 0.25,
        },
        "right_hand": {
            # p.GEOM_BOX
            "length": 0.16,
            "width": 0.08,
            "depth": 0.03,
        },
        "left_clavicle": {
            # p.GEOM_CAPSULE
            "radius": 0.025,
            "length": 0.20,
        },
        "left_upperarm": {
            # p.GEOM_CAPSULE
            "radius": 0.04,
            "length": 0.25,
        },
        "left_lowerarm": {
            # p.GEOM_CAPSULE
            "radius": 0.03,
            "length": 0.25,
        },
        "left_hand": {"length": 0.16, "width": 0.08, "depth": 0.03},
        "right_upperleg": {
            # p.GEOM_CAPSULE
            "radius": 0.06,
            "length": 0.4,
        },
        "right_lowerleg": {
            # p.GEOM_CAPSULE
            "radius": 0.04,
            "length": 0.4,
        },
        "right_foot": {
            # p.GEOM_BOX
            "length": 0.23,
            "width": 0.08,
            "depth": 0.06,
        },
        "left_upperleg": {
            # p.GEOM_CAPSULE
            "radius": 0.06,
            "length": 0.4,
        },
        "left_lowerleg": {
            # p.GEOM_CAPSULE
            "radius": 0.04,
            "length": 0.4,
        },
        "left_foot": {
            # p.GEOM_BOX
            "length": 0.23,
            "width": 0.08,
            "depth": 0.06,
        },
    }
)

HUMANOID_JOINT_NAMES = list(HUMANOID_PARAMS_DEFAULT.keys())

CHILD_PARENT_LINK_PAIRS = [
    # "child": "parent"
    ("trans_x", "base"),
    ("trans_y", "trans_x"),
    ("trans_z", "trans_y"),
    ("pelvis", "trans_z"),
    ("torso_1", "pelvis"),
    ("torso_2", "torso_1"),
    ("head", "torso_2"),
    ("right_clavicle", "torso_2"),
    ("right_upperarm", "right_clavicle"),
    ("right_lowerarm", "right_upperarm"),
    ("right_hand", "right_lowerarm"),
    ("left_clavicle", "torso_2"),
    ("left_upperarm", "left_clavicle"),
    ("left_lowerarm", "left_upperarm"),
    ("left_hand", "left_lowerarm"),
    ("right_upperleg", "pelvis"),
    ("right_lowerleg", "right_upperleg"),
    ("right_foot", "right_lowerleg"),
    ("left_upperleg", "pelvis"),
    ("left_lowerleg", "left_upperleg"),
    ("left_foot", "left_lowerleg"),
]
