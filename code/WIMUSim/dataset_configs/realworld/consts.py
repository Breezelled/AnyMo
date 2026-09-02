import numpy as np

# VIDEO_FRAME_RATE = 23.98 # 23.98, 25, 29.67 fps
IMU_FRAME_RATE = 50

ACTIVITY_LIST = [
    "walking",
    "sitting",
    "lying",
    "climbingdown",
    "running",
    "standing",
    "climbingup",
    "jumping",
]
PLACEMENT_LIST = ["shin", "thigh", "waist", "head", "chest", "forearm", "upperarm"]
IMU_NAMES = ["LHD", "WST", "LCH", "LUA", "LLA", "LTH", "LSH"]

SUBJECT_INFO = {
    "p001": {
        "gender": "F",
        "age": 52,
        "height": 1.63,
        "weight": 48.0,
        "occupation": "Industrial Management Assistant",
    },
    "p002": {
        "gender": "M",
        "age": 26,
        "height": 1.79,
        "weight": 70.0,
        "occupation": "Business Informatics",
    },
    "p003": {
        "gender": "M",
        "age": 27,
        "height": 1.76,
        "weight": 81.0,
        "occupation": "Business Informatics",
    },
    "p004": {
        "gender": "M",
        "age": 26,
        "height": 1.83,
        "weight": 82.0,
        "occupation": "Business Informatics",
    },
    "p005": {
        "gender": "M",
        "age": 62,
        "height": 1.70,
        "weight": 70.0,
        "occupation": "Telecommunication Engineer",
    },
    "p006": {
        "gender": "F",
        "age": 24,
        "height": 1.74,
        "weight": 65.0,
        "occupation": "Geriatric Nurse Assistant",
    },
    "p007": {
        "gender": "M",
        "age": 26,
        "height": 1.80,
        "weight": 81.0,
        "occupation": "Business Informatics",
    },
    "p008": {
        "gender": "F",
        "age": 36,
        "height": 1.65,
        "weight": 95.0,
        "occupation": "Nationally Certified Physical Therapist",
    },
    "p009": {
        "gender": "M",
        "age": 26,
        "height": 1.79,
        "weight": 95.0,
        "occupation": "Plumbing and Heating Installer",
    },
    "p010": {
        "gender": "M",
        "age": 26,
        "height": 1.70,
        "weight": 90.0,
        "occupation": "Chemical Laboratory Assistant",
    },
    "p011": {
        "gender": "F",
        "age": 48,
        "height": 1.75,
        "weight": 71.0,
        "occupation": "Industrial Management Assistant",
    },
    "p012": {
        "gender": "F",
        "age": 16,
        "height": 1.64,
        "weight": 54.0,
        "occupation": "Pupil",
    },
    "p013": {
        "gender": "F",
        "age": 27,
        "height": 1.70,
        "weight": 65.0,
        "occupation": "Japanology",
    },
    "p014": {
        "gender": "M",
        "age": 26,
        "height": 1.83,
        "weight": 78.0,
        "occupation": "Business Informatics",
    },
    "p015": {
        "gender": "F",
        "age": 30,
        "height": 1.65,
        "weight": 66.0,
        "occupation": "Geologist",
    },
}

# Configuration for WIMUSim parameter optimization
ROM_RANGE_ADJ = [-20, 20]  # ADJ: Adjustment
JOINT_ROM_DICT = {  # "JOINT_NAME": np.array([[min, max], [min, max], [min, max]])
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

# Keys are the names used in WIMUSIM, values are the names used in the dataset
IMU_NAME_PAIRS_DICT = {
    "LCH": "chest",
    "LLA": "forearm",
    "LHD": "head",
    "LSH": "shin",
    "LTH": "thigh",
    "LUA": "upperarm",
    "WST": "waist",
}
