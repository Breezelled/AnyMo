import numpy as np

# Consts of the humanoid model
## E parameters
g = np.array([0, 0, -9.81])  # Gravity vector. The world frame is ENU (East-North-Up)

SAMPLING_RATE = 100  # Hz

# B parameters '_r' means radius, '_l' means length
HEAD_r = 0.1
SPINE_l = 1.765 / 2  # 0.883
SPINE_r = 0.06
SHOULDER1_r = 0.05
BLUESENSE_HWD = np.array([0.03, 0.03, 0.01])  # Height, Width, and Depth
CLAVICLE_l = 0.16  # Collarbone which connects an arm to the body
CLAVICLE_r = 0.03
HUMERUS_l = 0.25  # Upper arm
HUMERUS_r = 0.03
RADIUS_ULNA_l = 0.25  # Lower arm consists of radius and ulna
RADIUS_ULNA_r = 0.025

p_WORLD = np.array([0.0, 0.0, 0.0])
o_WORLD = np.array([0.0, 0.0, 0.0])

# This will be tuned during optimization.

# P parameters '_p' means position, '_q' means orientation
rp_WORLD2PELVIS = np.array([0.0, 0.0, 0.0])
rp_PELVIS2R_SHOULDER = np.array([SPINE_r + CLAVICLE_l, 0.0, SPINE_l - HEAD_r * 3])
rp_R_SHOULDER2R_ELBOW = np.array([HUMERUS_l, 0.0, 0.0])
# rp_XXX is the relative position of the joint/link in the parent link frame.
rp_PELVIS2TRS = np.array(
    [0.0, SPINE_r + BLUESENSE_HWD[2] / 2, SPINE_l - HEAD_r * 4]
)  # NOTE: z coordinate is too high
rp_SHOULDER2RUA = np.array([HUMERUS_l * 0.75, 0, HUMERUS_r + BLUESENSE_HWD[2] / 2])
rp_ELBOW2RWR = np.array([RADIUS_ULNA_l * 0.8, 0, RADIUS_ULNA_r + BLUESENSE_HWD[2] / 2])

RGBA_BLUE = (135 / 255, 206 / 255, 250 / 255, 255 / 255)
IMU_NAMES = ["TRS", "RUA", "RWR"]
IMU_SIZES = [BLUESENSE_HWD for _ in IMU_NAMES]
IMU_COLORS = [RGBA_BLUE for _ in IMU_NAMES]

JOINT_IMU_PAIRS = [
    ("PELVIS", "TRS"),
    ("R_SHOULDER", "RUA"),
    ("R_ELBOW", "RWR"),
]

JOINT_CHILD_PARENT_PAIRS = [
    ("PELVIS", "BASE"),
    ("R_CLAVICLE", "PELVIS"),
    ("R_SHOULDER", "R_CLAVICLE"),
    ("R_ELBOW", "R_SHOULDER"),
]

JOINT_WIMUSIM_LINK_PAIRS = [
    ("BASE", "pelvis"),
    ("PELVIS", "torso_1"),
    ("R_CLAVICLE", "right_clavicle"),
    ("R_SHOULDER", "right_upperarm"),
    ("R_ELBOW", "right_lowerarm"),
]

JOINT_WIMUSIM_LINK_DICT = {
    joint_name: link_name for joint_name, link_name in JOINT_WIMUSIM_LINK_PAIRS
}

ROM_RANGE_ADJ = [-20, 20]  # ADJ: Adjustment
JOINT_ROM_DICT = {  # "JOINT_NAME": np.array([[min, max], [min, max], [min, max]])
    "BASE": np.deg2rad(
        np.array([[-180, 180], [-90, 90], [-180, 180]])
    ),  # No limit for the base
    "PELVIS": np.deg2rad(
        np.array([[-60, 30], [-50, 50], [-40, 40]])
    ),  # np.deg2rad(np.array([[-45, 30], [-50, 50], [-40, 40]])), # default values
    "R_CLAVICLE": np.deg2rad(np.array([ROM_RANGE_ADJ, [-20, +10], [-20, 20]])),
    "R_SHOULDER": np.deg2rad(np.array([[-140, 90], [-90, 90], [-30, 135]])),
    "R_ELBOW": np.deg2rad(np.array([[-90, 90], ROM_RANGE_ADJ, [-5, 145]])),
}

# rp_XXX is the relative position of the joint/link in the parent link frame.
B_DEFAULT = {
    "rp": {
        ("BASE", "PELVIS"): np.array([0.0, 0.0, 0.0]),  # No need to be tuned
        ("PELVIS", "R_CLAVICLE"): np.array([0.0, 0.0, 0.43]),
        ("R_CLAVICLE", "R_SHOULDER"): np.array([0.22, 0.0, 0.0]),
        ("R_SHOULDER", "R_ELBOW"): np.array([0.27, 0.0, 0.0]),
    },
    "ro": {},
}

# Use B_DEFAULT values to specify the initial rp params.
P_DEFAULT = {
    "rp": {
        ("PELVIS", "TRS"): np.array([0.0, 0.10, 0.24]),
        ("R_SHOULDER", "RUA"): np.array([0.24, 0.0, 0.06]),
        ("R_ELBOW", "RWR"): np.array([0.23, 0.0, 0.04]),
    },
    "ro": {
        ("PELVIS", "TRS"): np.array([-np.pi / 2, 0.0, 0.0]),
        ("R_SHOULDER", "RUA"): np.array([0.0, 0.0, 0.0]),
        ("R_ELBOW", "RWR"): np.array([0.0, 0.0, 0.0]),
    },
}

# [[min, max], [min, max], [min, max]] for each joint
_B_NEAR_ZERO = [-0.03, 0.03]
B_RANGE_DEFAULT = {
    "rp": {
        ("BASE", "PELVIS"): np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, _B_NEAR_ZERO]),
        ("PELVIS", "R_CLAVICLE"): np.array([_B_NEAR_ZERO, _B_NEAR_ZERO, [0.40, 0.60]]),
        ("R_CLAVICLE", "R_SHOULDER"): np.array(
            [[0.15, 0.35], [-0.10, 0.00], _B_NEAR_ZERO]
        ),
        ("R_SHOULDER", "R_ELBOW"): np.array([[0.20, 0.30], _B_NEAR_ZERO, _B_NEAR_ZERO]),
    },
    "ro": {},
}

_Pp_NEAR_ZERO = [-0.03, 0.03]
_Po_NEAR_ZERO = [-5.0, 5.0]
P_RANGE_DEFAULT = {
    "rp": {
        ("PELVIS", "TRS"): np.array(
            [_Pp_NEAR_ZERO, [-0.05, 0.10], [0.20, 0.50]]
        ),  # np.array([0.0, -0.12, 0.30]),
        ("R_SHOULDER", "RUA"): np.array(
            [[0.05, 0.25], _Pp_NEAR_ZERO, [0.03, 0.08]]
        ),  # np.array([0.12, 0.0, 0.06]),
        ("R_ELBOW", "RWR"): np.array(
            [[0.1, 0.3], _Pp_NEAR_ZERO, [0.02, 0.06]]
        ),  # np.array([0.20, 0.0, 0.04]),
    },  # TODO: Specify default values
    "ro": {
        ("PELVIS", "TRS"): np.deg2rad(
            np.array([[-95, -85], [-5.0, 5.0], [-5, 5]])
        ),  # np.array([90.0, 0, -90.0])
        ("R_SHOULDER", "RUA"): np.deg2rad(
            np.array([[-5.0, 5.0], [-10, 0], [-5.0, 5.0]])
        ),  # np.array([0.0, 0.0, 0.0])
        ("R_ELBOW", "RWR"): np.deg2rad(
            np.array([[-5.0, 5], [-5.0, 5.0], [-5.0, 5.0]])
        ),  # np.array([0.0, 0.0, 0.0])
    },
}
