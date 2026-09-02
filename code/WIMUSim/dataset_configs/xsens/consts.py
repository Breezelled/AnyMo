import numpy as np


def deg2rad(deg):
    return deg * np.pi / 180.0


SAMPLING_RATE = 60  # Hz

PART_NAMES = [
    "Pelvis",
    "L5",
    "L3",
    "T12",
    "T8",
    "Neck",
    "Head",
    "R_Shoulder",
    "R_UpperArm",
    "R_Forearm",
    "R_Hand",
    "L_Shoulder",
    "L_UpperArm",
    "L_Forearm",
    "L_Hand",
    "R_UpperLeg",
    "R_LowerLeg",
    "R_Foot",
    "R_Toe",
    "L_UpperLeg",
    "L_LowerLeg",
    "L_Foot",
    "L_Toe",
]

IMU_NAMES = list(PART_NAMES)
JOINT_NAMES = ["BASE"] + PART_NAMES

BODY_PARENT = {
    "Pelvis": "BASE",
    "L5": "Pelvis",
    "L3": "L5",
    "T12": "L3",
    "T8": "T12",
    "Neck": "T8",
    "Head": "Neck",
    "R_Shoulder": "T8",
    "R_UpperArm": "R_Shoulder",
    "R_Forearm": "R_UpperArm",
    "R_Hand": "R_Forearm",
    "L_Shoulder": "T8",
    "L_UpperArm": "L_Shoulder",
    "L_Forearm": "L_UpperArm",
    "L_Hand": "L_Forearm",
    "R_UpperLeg": "Pelvis",
    "R_LowerLeg": "R_UpperLeg",
    "R_Foot": "R_LowerLeg",
    "R_Toe": "R_Foot",
    "L_UpperLeg": "Pelvis",
    "L_LowerLeg": "L_UpperLeg",
    "L_Foot": "L_LowerLeg",
    "L_Toe": "L_Foot",
}

JOINT_PARENT_CHILD_PAIRS = [
    (parent, child) for child, parent in BODY_PARENT.items()
]
JOINT_CHILD_PARENT_PAIRS = [
    (child, parent) for parent, child in JOINT_PARENT_CHILD_PAIRS
]

JOINT_ID_DICT = {"BASE": 0}
for idx, part_name in enumerate(PART_NAMES, start=1):
    JOINT_ID_DICT[part_name] = idx


def _link_name(joint_name: str) -> str:
    return joint_name.lower()


JOINT_WIMUSIM_LINK_PAIRS = [
    (joint_name, _link_name(joint_name)) for joint_name in JOINT_NAMES
]
JOINT_WIMUSIM_LINK_DICT = {
    joint_name: link_name for joint_name, link_name in JOINT_WIMUSIM_LINK_PAIRS
}

JOINT_IMU_PAIRS = [(joint_name, joint_name) for joint_name in IMU_NAMES]

XSENS_IMU_SIZE = np.array([0.058, 0.058, 0.033], dtype=np.float32)
RGBA_BLUE = (68 / 255, 114 / 255, 196 / 255, 1.0)
IMU_SIZES = [XSENS_IMU_SIZE for _ in IMU_NAMES]
IMU_COLORS = [RGBA_BLUE for _ in IMU_NAMES]

_FULL_ROM = np.deg2rad(np.array([[-180, 180], [-180, 180], [-180, 180]], dtype=np.float32))
_TORSO_ROM = np.deg2rad(np.array([[-45, 45], [-40, 40], [-45, 45]], dtype=np.float32))
_PELVIS_ROM = np.deg2rad(np.array([[-60, 30], [-50, 50], [-45, 45]], dtype=np.float32))
_NECK_ROM = np.deg2rad(np.array([[-50, 60], [-60, 60], [-50, 50]], dtype=np.float32))
_HEAD_ROM = np.deg2rad(np.array([[-20, 20], [-20, 20], [-20, 20]], dtype=np.float32))
_CLAVICLE_R_ROM = np.deg2rad(np.array([[-20, 20], [-20, 10], [-20, 20]], dtype=np.float32))
_CLAVICLE_L_ROM = np.deg2rad(np.array([[-20, 20], [-10, 20], [-20, 20]], dtype=np.float32))
_SHOULDER_R_ROM = np.deg2rad(np.array([[-140, 90], [-90, 90], [-30, 135]], dtype=np.float32))
_SHOULDER_L_ROM = np.deg2rad(np.array([[-140, 90], [-90, 90], [-135, 30]], dtype=np.float32))
_ELBOW_R_ROM = np.deg2rad(np.array([[-90, 90], [-20, 20], [-5, 145]], dtype=np.float32))
_ELBOW_L_ROM = np.deg2rad(np.array([[-90, 90], [-20, 20], [-145, 5]], dtype=np.float32))
_WRIST_ROM = np.deg2rad(np.array([[-40, 40], [-40, 40], [-60, 60]], dtype=np.float32))
_HIP_R_ROM = np.deg2rad(np.array([[-15, 125], [-45, 20], [-45, 45]], dtype=np.float32))
_HIP_L_ROM = np.deg2rad(np.array([[-15, 125], [-20, 45], [-45, 45]], dtype=np.float32))
_KNEE_ROM = np.deg2rad(np.array([[-130, 0], [-20, 20], [-20, 20]], dtype=np.float32))
_ANKLE_ROM = np.deg2rad(np.array([[-45, 45], [-30, 30], [-30, 30]], dtype=np.float32))
_TOE_ROM = np.deg2rad(np.array([[-30, 30], [-20, 20], [-20, 20]], dtype=np.float32))

JOINT_ROM_DICT = {
    "BASE": _FULL_ROM,
    "Pelvis": _PELVIS_ROM,
    "L5": _TORSO_ROM,
    "L3": _TORSO_ROM,
    "T12": _TORSO_ROM,
    "T8": _TORSO_ROM,
    "Neck": _NECK_ROM,
    "Head": _HEAD_ROM,
    "R_Shoulder": _CLAVICLE_R_ROM,
    "R_UpperArm": _SHOULDER_R_ROM,
    "R_Forearm": _ELBOW_R_ROM,
    "R_Hand": _WRIST_ROM,
    "L_Shoulder": _CLAVICLE_L_ROM,
    "L_UpperArm": _SHOULDER_L_ROM,
    "L_Forearm": _ELBOW_L_ROM,
    "L_Hand": _WRIST_ROM,
    "R_UpperLeg": _HIP_R_ROM,
    "R_LowerLeg": _KNEE_ROM,
    "R_Foot": _ANKLE_ROM,
    "R_Toe": _TOE_ROM,
    "L_UpperLeg": _HIP_L_ROM,
    "L_LowerLeg": _KNEE_ROM,
    "L_Foot": _ANKLE_ROM,
    "L_Toe": _TOE_ROM,
}

B_DEFAULT = {
    "rp": {
        ("BASE", "Pelvis"): np.array([0.0, 0.0, 0.0], dtype=np.float32),
        ("Pelvis", "L5"): np.array([0.0, 0.0, 0.10], dtype=np.float32),
        ("L5", "L3"): np.array([0.0, 0.0, 0.10], dtype=np.float32),
        ("L3", "T12"): np.array([0.0, 0.0, 0.10], dtype=np.float32),
        ("T12", "T8"): np.array([0.0, 0.0, 0.12], dtype=np.float32),
        ("T8", "Neck"): np.array([0.0, 0.0, 0.16], dtype=np.float32),
        ("Neck", "Head"): np.array([0.0, 0.0, 0.18], dtype=np.float32),
        ("T8", "R_Shoulder"): np.array([0.18, 0.0, 0.04], dtype=np.float32),
        ("R_Shoulder", "R_UpperArm"): np.array([0.08, 0.0, 0.0], dtype=np.float32),
        ("R_UpperArm", "R_Forearm"): np.array([0.28, 0.0, 0.0], dtype=np.float32),
        ("R_Forearm", "R_Hand"): np.array([0.24, 0.0, 0.0], dtype=np.float32),
        ("T8", "L_Shoulder"): np.array([-0.18, 0.0, 0.04], dtype=np.float32),
        ("L_Shoulder", "L_UpperArm"): np.array([-0.08, 0.0, 0.0], dtype=np.float32),
        ("L_UpperArm", "L_Forearm"): np.array([-0.28, 0.0, 0.0], dtype=np.float32),
        ("L_Forearm", "L_Hand"): np.array([-0.24, 0.0, 0.0], dtype=np.float32),
        ("Pelvis", "R_UpperLeg"): np.array([0.10, 0.0, 0.0], dtype=np.float32),
        ("R_UpperLeg", "R_LowerLeg"): np.array([0.0, 0.0, -0.42], dtype=np.float32),
        ("R_LowerLeg", "R_Foot"): np.array([0.0, 0.0, -0.43], dtype=np.float32),
        ("R_Foot", "R_Toe"): np.array([0.0, 0.12, -0.03], dtype=np.float32),
        ("Pelvis", "L_UpperLeg"): np.array([-0.10, 0.0, 0.0], dtype=np.float32),
        ("L_UpperLeg", "L_LowerLeg"): np.array([0.0, 0.0, -0.42], dtype=np.float32),
        ("L_LowerLeg", "L_Foot"): np.array([0.0, 0.0, -0.43], dtype=np.float32),
        ("L_Foot", "L_Toe"): np.array([0.0, 0.12, -0.03], dtype=np.float32),
    },
    "ro": {},
}


def _vector_range(vec: np.ndarray, frac: float = 0.35, floor: float = 0.02) -> np.ndarray:
    delta = np.maximum(np.abs(vec) * frac, floor).astype(np.float32)
    return np.stack([vec - delta, vec + delta], axis=1).astype(np.float32)


B_RANGE_DEFAULT = {
    "rp": {key: _vector_range(value) for key, value in B_DEFAULT["rp"].items()},
    "ro": {},
}

_CHILDREN = {}
for parent, child in JOINT_PARENT_CHILD_PAIRS:
    _CHILDREN.setdefault(parent, []).append(child)


def _default_imu_offset(joint_name: str) -> np.ndarray:
    children = _CHILDREN.get(joint_name, [])
    if children:
        return (0.5 * B_DEFAULT["rp"][(joint_name, children[0])]).astype(np.float32)
    parent = BODY_PARENT[joint_name]
    if parent is None:
        return np.array([0.0, 0.0, 0.05], dtype=np.float32)
    return (-0.25 * B_DEFAULT["rp"][(parent, joint_name)]).astype(np.float32)


P_DEFAULT = {
    "rp": {
        (joint_name, joint_name): _default_imu_offset(joint_name)
        for joint_name in IMU_NAMES
    },
    "ro": {
        (joint_name, joint_name): np.zeros(3, dtype=np.float32)
        for joint_name in IMU_NAMES
    },
}

P_RANGE_DEFAULT = {
    "rp": {key: _vector_range(value, frac=0.50, floor=0.03) for key, value in P_DEFAULT["rp"].items()},
    "ro": {
        key: np.deg2rad(
            np.array([[-180, 180], [-180, 180], [-180, 180]], dtype=np.float32)
        )
        for key in P_DEFAULT["ro"].keys()
    },
}

