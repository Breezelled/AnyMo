import numpy as np


def deg2rad(deg):
    """
    Convert degrees to radians.
    :param deg: degrees in float or numpy array.
    :return: degrees in radians.
    """
    return deg * np.pi / 180


SAMPLING_RATE = 50  # Hz
N_SUBJECT = 17

SCENARIO_NAMES = ["ideal", "self", "mutual"]

# DO NOT change the order as it is used in the code when loading the dataset.
IMU_NAMES = ["RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC"]
IMU_COLUMNS = [
    f"{sensor}{axis}" for sensor in ["Acc", "Gyro", "Mag"] for axis in "XYZ"
] + [f"Quat{axis}" for axis in "WXYZ"]
REALDISP_COLUMNS = (
    ["timestamp_s", "timestamp_us"]
    + [f"{imu_name}_{imu_col}" for imu_name in IMU_NAMES for imu_col in IMU_COLUMNS]
    + ["ActivityLabel"]
)

XSENS_HWD = np.array([0.058, 0.058, 0.033])  # Height, Width, and Depth
RGBA_ORANGE = (249 / 255, 105 / 255, 14 / 255, 255 / 255)
IMU_SIZES = [XSENS_HWD for _ in IMU_NAMES]
IMU_COLORS = [RGBA_ORANGE for _ in IMU_NAMES]

JOINT_IMU_PAIRS = [
    ("PELVIS", "BACK"),
    ("R_SHOULDER", "RUA"),
    ("R_ELBOW", "RLA"),
    ("L_SHOULDER", "LUA"),
    ("L_ELBOW", "LLA"),
    ("R_HIP", "RT"),
    ("R_KNEE", "RC"),
    ("L_HIP", "LT"),
    ("L_KNEE", "LC"),
]

JOINT_CHILD_PARENT_PAIRS = [
    ("PELVIS", "BASE"),
    ("R_CLAVICLE", "PELVIS"),
    ("R_SHOULDER", "R_CLAVICLE"),
    ("R_ELBOW", "R_SHOULDER"),
    ("L_CLAVICLE", "PELVIS"),
    ("L_SHOULDER", "L_CLAVICLE"),
    ("L_ELBOW", "L_SHOULDER"),
    ("R_HIP", "BASE"),
    ("L_HIP", "BASE"),
    ("R_KNEE", "R_HIP"),
    ("L_KNEE", "L_HIP"),
]

JOINT_WIMUSIM_LINK_PAIRS = [
    ("BASE", "pelvis"),
    ("PELVIS", "torso_1"),
    ("R_CLAVICLE", "right_clavicle"),
    ("R_SHOULDER", "right_upperarm"),
    ("R_ELBOW", "right_lowerarm"),
    ("L_CLAVICLE", "left_clavicle"),
    ("L_SHOULDER", "left_upperarm"),
    ("L_ELBOW", "left_lowerarm"),
    ("R_HIP", "right_upperleg"),
    ("R_KNEE", "right_lowerleg"),
    ("L_HIP", "left_upperleg"),
    ("L_KNEE", "left_lowerleg"),
]
JOINT_WIMUSIM_LINK_DICT = {
    joint_name: link_name for joint_name, link_name in JOINT_WIMUSIM_LINK_PAIRS
}

# This should ideally be [-0, 0] # but due to the limitation of the current implementation, it is set to [-20, 20].
ROM_RANGE_ADJ = [-20, 20]  # ADJ: Adjustment
JOINT_ROM_DICT = {  # "JOINT_NAME": np.array([[min, max], [min, max], [min, max]])
    "BASE": np.deg2rad(
        np.array([[-180, 180], [-90, 90], [-180, 180]])
    ),  # No limit for the base
    "PELVIS": np.deg2rad(
        np.array([[-60, 30], [-50, 50], [-40, 40]])
    ),  # np.deg2rad(np.array([[-45, 30], [-50, 50], [-40, 40]])), # default values
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
}

# "world2pelvis": np.array([0.0, 0.0, 0.0]),  # No need to be tuned
# "pelvis2shoulder": np.array([0.22, -0.09, 0.43]),
# "shoulder2elbow": np.array([0.27, 0.0, 0.0]),
# Key naming rule: f"{JOINT_NAME(PARENT)}2{JOINT_NAME(CHILD)}"
B_DEFAULT = {
    "rp": {
        ("BASE", "PELVIS"): np.array([0.0, 0.0, 0.0]),
        ("PELVIS", "R_CLAVICLE"): np.array([0.0, 0.0, 0.46]),
        ("R_CLAVICLE", "R_SHOULDER"): np.array([0.22, -0.09, 0.0]),
        ("R_SHOULDER", "R_ELBOW"): np.array([0.27, 0.0, 0.0]),
        ("PELVIS", "L_CLAVICLE"): np.array([0.0, 0.0, 0.46]),
        ("L_CLAVICLE", "L_SHOULDER"): np.array([-0.22, -0.09, 0.0]),
        ("L_SHOULDER", "L_ELBOW"): np.array([-0.27, 0.0, 0.0]),
        ("BASE", "R_HIP"): np.array([0.14, 0.0, 0.0]),
        ("R_HIP", "R_KNEE"): np.array([0.0, 0.0, -0.4]),
        ("BASE", "L_HIP"): np.array([-0.14, 0.0, 0.0]),
        ("L_HIP", "L_KNEE"): np.array([0.0, 0.0, -0.4]),
    },
    "ro": {},
}

# Use B_DEFAULT values to specify the initial rp params.
# Key naming rule: f"{JOINT_NAME(PARENT)}2{IMU_NAME(CHILD)}"
P_DEFAULT = {
    "rp": {
        ("PELVIS", "BACK"): np.array([0.0, -0.12, 0.30]),
        ("R_SHOULDER", "RUA"): np.array([0.12, 0.0, 0.06]),
        ("R_ELBOW", "RLA"): np.array([0.20, 0.0, 0.04]),
        ("L_SHOULDER", "LUA"): np.array([-0.12, 0.0, 0.06]),
        ("L_ELBOW", "LLA"): np.array([-0.20, 0.0, 0.04]),
        ("R_HIP", "RT"): np.array([0, 0.06, -0.2]),
        ("R_KNEE", "RC"): np.array([0, 0.04, -0.2]),
        ("L_HIP", "LT"): np.array([0, 0.06, -0.2]),
        ("L_KNEE", "LC"): np.array([0, 0.04, -0.2]),
    },
    "ro": {
        ("PELVIS", "BACK"): deg2rad(np.array([90.0, 0, -90.0])),
        ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
        ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
        ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
        ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
        ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
        ("R_KNEE", "RC"): deg2rad(np.array([-90.0, 0.0, 90.0])),
        ("L_HIP", "LT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
        ("L_KNEE", "LC"): deg2rad(np.array([-90.0, 0.0, 90.0])),
    },
}


# [[min, max], [min, max], [min, max]] for each joint
_B_NEAR_ZERO = [-0.01, 0.01]
B_RANGE_DEFAULT = {
    "rp": {
        ("BASE", "PELVIS"): np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, _B_NEAR_ZERO]),
        ("PELVIS", "R_CLAVICLE"): np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, [0.40, 0.60]]),
        ("R_CLAVICLE", "R_SHOULDER"): np.array(
            [[0.15, 0.35], [-0.10, 0.00], _B_NEAR_ZERO]
        ),
        ("R_SHOULDER", "R_ELBOW"): np.array([[0.20, 0.30], _B_NEAR_ZERO, _B_NEAR_ZERO]),
        ("PELVIS", "L_CLAVICLE"): np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, [0.40, 0.60]]),
        ("L_CLAVICLE", "L_SHOULDER"): np.array(
            [[-0.35, -0.15], [-0.10, 0.00], _B_NEAR_ZERO]
        ),
        ("L_SHOULDER", "L_ELBOW"): np.array(
            [[-0.30, -0.20], _B_NEAR_ZERO, _B_NEAR_ZERO]
        ),
        ("BASE", "R_HIP"): np.array([[0.10, 0.20], _B_NEAR_ZERO, _B_NEAR_ZERO]),
        ("R_HIP", "R_KNEE"): np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, [-0.50, -0.30]]),
        ("BASE", "L_HIP"): np.array([[-0.20, -0.10], _B_NEAR_ZERO, _B_NEAR_ZERO]),
        ("L_HIP", "L_KNEE"): np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, [-0.50, -0.30]]),
    },
    "ro": {},
}

_Pp_NEAR_ZERO = [-0.03, 0.03]
_Po_NEAR_ZERO = [-5.0, 5.0]
P_RANGE_DEFAULT = {
    "rp": {
        ("PELVIS", "BACK"): np.array(
            [_Pp_NEAR_ZERO, [-0.16, -0.08], [0.20, 0.40]]
        ),  # np.array([0.0, -0.12, 0.30]),
        ("R_SHOULDER", "RUA"): np.array(
            [[0.05, 0.25], _Pp_NEAR_ZERO, [0.04, 0.08]]
        ),  # np.array([0.12, 0.0, 0.06]),
        ("R_ELBOW", "RLA"): np.array(
            [[0.1, 0.3], _Pp_NEAR_ZERO, [0.02, 0.06]]
        ),  # np.array([0.20, 0.0, 0.04]),
        ("L_SHOULDER", "LUA"): np.array(
            [[-0.25, -0.05], _Pp_NEAR_ZERO, [0.04, 0.08]]
        ),  # np.array([-0.12, 0.0, 0.06]),
        ("L_ELBOW", "LLA"): np.array(
            [[-0.3, -0.1], _Pp_NEAR_ZERO, [0.02, 0.06]]
        ),  # np.array([-0.20, 0.0, 0.04]),
        ("R_HIP", "RT"): np.array(
            [_Pp_NEAR_ZERO, [0.04, 0.10], [-0.3, -0.15]]
        ),  # np.array([0, 0.06, -0.2]),
        ("R_KNEE", "RC"): np.array(
            [_Pp_NEAR_ZERO, [0.02, 0.08], [-0.3, -0.1]]
        ),  # np.array([0, 0.04, -0.2]),
        ("L_HIP", "LT"): np.array(
            [_Pp_NEAR_ZERO, [0.04, 0.10], [-0.3, -0.15]]
        ),  # np.array([0, 0.06, -0.2]),
        ("L_KNEE", "LC"): np.array(
            [_Pp_NEAR_ZERO, [0.02, 0.08], [-0.3, -0.1]]
        ),  # np.array([0, 0.04, -0.2]),
    },  # TODO: Specify default values
    "ro": {
        ("PELVIS", "BACK"): deg2rad(
            np.array([[85.0, 95.0], [-5.0, 5.0], [-95.0, -85.0]])
        ),  # np.array([90.0, 0, -90.0])
        ("R_SHOULDER", "RUA"): deg2rad(
            np.array([[-5.0, 5.0], [0.0, 10.0], [-5.0, 5.0]])
        ),  # np.array([0.0, 0.0, 0.0])
        ("R_ELBOW", "RLA"): deg2rad(
            np.array([[-5.0, 5.0], [0.0, 10.0], [-5.0, 5.0]])
        ),  # np.array([0.0, 0.0, 0.0])
        ("L_SHOULDER", "LUA"): deg2rad(
            np.array([[-5.0, 5.0], [0.0, 10.0], [175.0, 185.0]])
        ),  # np.array([0.0, 0.0, 180.0])
        ("L_ELBOW", "LLA"): deg2rad(
            np.array([[-5.0, 5.0], [0.0, 10.0], [175.0, 185.0]])
        ),  # np.array([0.0, 0.0, 180.0])
        ("R_HIP", "RT"): deg2rad(
            np.array([[-95.0, -85.0], [0.0, 15.0], [85.0, 95.0]])
        ),  # np.array([-90.0, 0.0, 90.0])
        ("R_KNEE", "RC"): deg2rad(
            np.array([[-100.0, -80.0], [0.0, 15.0], [85.0, 95.0]])
        ),  # np.array([-90.0, 0.0, 90.0])
        ("L_HIP", "LT"): deg2rad(
            np.array([[-95.0, -85.0], [0.0, 15.0], [85.0, 95.0]])
        ),  # np.array([-90.0, 0.0, 90.0])
        ("L_KNEE", "LC"): deg2rad(
            np.array([[-100.0, -80.0], [0.0, 15.0], [85.0, 95.0]])
        ),  # np.array([-90.0, 0.0, 90.0])
    },
}

ACTIVITY_NAME_ID_DICT = {
    "Null": 0,
    "Walking": 1,
    "Jogging": 2,
    "Running": 3,
    "Jump Up": 4,
    "Jump Front & Back": 5,
    "Jump Sideways": 6,
    "Jump Leg/Arms Open/Closed": 7,
    "Jump Rope": 8,
    "Trunk Twist (arms outstretched)": 9,
    "Trunk Twist (elbows bended)": 10,
    "Waist Bends Forward": 11,
    "Waist Rotation": 12,
    "Waist Bends (reach foot with opposite hand)": 13,
    "Reach Heels Backwards": 14,
    "Lateral Bend (left or right)": 15,
    "Lateral Bend Arm Up (left or right)": 16,
    "Repetitive Forward Stretching": 17,
    "Upper Trunk and Lower Body Opposite Twist": 18,
    "Arms Lateral Elevation": 19,
    "Arms Frontal Elevation": 20,
    "Frontal Hand Claps": 21,
    "Arms Frontal Crossing": 22,
    "Shoulders High Amplitude Rotation": 23,
    "Shoulders Low Amplitude Rotation": 24,
    "Arms Inner Rotation": 25,
    "Knees (alternatively) to the Breast": 26,
    "Heels (alternatively) to the Backside": 27,
    "Knees Bending (crouching)": 28,
    "Knees (alternatively) Bend Forward": 29,
    "Rotation on the Knees": 30,
    "Rowing": 31,
    "Elliptic Bike": 32,
    "Cycling": 33,
}

P_SELF_DEFAULT_DICT = {
    "P_001": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 0, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([-90, 0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0, 0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(np.array([-90.0, -90.0, -90.0])),
            ("L_HIP", "LT"): deg2rad(np.array([-90.0, 0.0, -90.0])),
            ("L_KNEE", "LC"): deg2rad(np.array([-90, -90.0, 90.0])),
        },
    },
    "P_002": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 0, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([135, 0, 0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(np.array([-90.0, 0, 90.0])),
            ("L_HIP", "LT"): deg2rad(np.array([-90.0, 0.0, -90.0])),
            ("L_KNEE", "LC"): deg2rad(np.array([-90, 0.0, 90.0])),
        },
    },
    "P_003": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(
                np.array([90.0, -90, -90.0])
            ),  # [90.0, 0, -90.0]
            ("R_SHOULDER", "RUA"): deg2rad(np.array([45, 0.0, 180])),  # [0.0, 0.0, 0.0]
            ("R_ELBOW", "RLA"): deg2rad(np.array([90, 0, 0])),  # OK
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),  # OK
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([90.0, 0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    "P_004": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 0, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 00.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([180.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(
                np.array([0.0, 0.0, 180.0])
            ),  # (0, 0, -45) ?
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 180.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(np.array([-90.0, 180.0, 90.0])),  # Hasn't fixed
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, -45.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    "P_005": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, -120, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([45.0, -45.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([135.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    # P_006 has no data
    "P_007": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 0, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    "P_008": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, -30, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, -90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, -90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    "P_009": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, -45, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([-90.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    "P_010": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 0, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 180.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    "P_011": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 135, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 90.0, -90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    "P_012": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 0, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([180.0, 0.0, 0.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, -90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    # P013 no data
    "P_014": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 0, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([180.0, 0.0, 180.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([-90.0, 0.0, 0.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, -90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    "P_015": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 90, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 90.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    "P_016": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 135, 90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
    "P_017": {
        "rp": P_DEFAULT["rp"],
        "ro": {
            ("PELVIS", "BACK"): deg2rad(np.array([90.0, 0, -90.0])),
            ("R_SHOULDER", "RUA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("R_ELBOW", "RLA"): deg2rad(np.array([0.0, 0.0, 0.0])),
            ("L_SHOULDER", "LUA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("L_ELBOW", "LLA"): deg2rad(np.array([0.0, 0.0, 180.0])),
            ("R_HIP", "RT"): deg2rad(np.array([-90.0, 0.0, 90.0])),
            ("R_KNEE", "RC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
            ("L_HIP", "LT"): deg2rad(
                np.array([-90.0, 0, 90.0])
            ),  # default val: [-90.0, 0, 90.0]
            ("L_KNEE", "LC"): deg2rad(
                np.array([-90.0, 0.0, 90.0])
            ),  # default val: [-90.0, 0.0, 90.0]
        },
    },
}
