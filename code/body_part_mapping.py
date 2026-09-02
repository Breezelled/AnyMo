"""
Body-part mappings for the zero-shot HAR datasets evaluated by AnyMo.

This file uses the numbered body diagram the user provided to interpret the
joint indices used by the shared 22-slot preprocessing format, then maps those
locations onto this project's `SEGMENT_NAMES` in `code/data.py`.

Important:
- These are the body-part slots used when each array-format dataset is placed
  into the shared 22-joint template.
- They are not guaranteed to be the original dataset's official sensor names.
- Left/right here follows the provided front-view body diagram.
"""

from __future__ import annotations

from collections import OrderedDict


SEGMENT_NAMES_REFERENCE = (
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
)


def _humanize_segment_name(name: str) -> str:
    if name.startswith("L_"):
        return "Left " + name[2:].replace("_", " ")
    if name.startswith("R_"):
        return "Right " + name[2:].replace("_", " ")
    return name


ARRAY_JOINT_TO_BODY_PART = OrderedDict(
    {
        0: {
            "figure_location": "pelvis / hip center",
            "project_segment_name": "Pelvis",
            "segment_name": _humanize_segment_name("Pelvis"),
        },
        1: {
            "figure_location": "left hip / upper leg",
            "project_segment_name": "L_UpperLeg",
            "segment_name": _humanize_segment_name("L_UpperLeg"),
        },
        2: {
            "figure_location": "left lower leg / shin",
            "project_segment_name": "L_LowerLeg",
            "segment_name": _humanize_segment_name("L_LowerLeg"),
        },
        3: {
            "figure_location": "left foot / ankle",
            "project_segment_name": "L_Foot",
            "segment_name": _humanize_segment_name("L_Foot"),
        },
        4: {
            "figure_location": "left toe / forefoot",
            "project_segment_name": "L_Toe",
            "segment_name": _humanize_segment_name("L_Toe"),
        },
        5: {
            "figure_location": "right hip / upper leg",
            "project_segment_name": "R_UpperLeg",
            "segment_name": _humanize_segment_name("R_UpperLeg"),
        },
        6: {
            "figure_location": "right lower leg / shin",
            "project_segment_name": "R_LowerLeg",
            "segment_name": _humanize_segment_name("R_LowerLeg"),
        },
        7: {
            "figure_location": "right foot / ankle",
            "project_segment_name": "R_Foot",
            "segment_name": _humanize_segment_name("R_Foot"),
        },
        8: {
            "figure_location": "right toe / forefoot",
            "project_segment_name": "R_Toe",
            "segment_name": _humanize_segment_name("R_Toe"),
        },
        9: {
            "figure_location": "lower spine / waist",
            "project_segment_name": "L5",
            "segment_name": _humanize_segment_name("L5"),
        },
        10: {
            "figure_location": "mid spine / abdomen",
            "project_segment_name": "L3",
            "segment_name": _humanize_segment_name("L3"),
        },
        11: {
            "figure_location": "upper trunk / chest / sternum",
            "project_segment_name": "T8",
            "segment_name": _humanize_segment_name("T8"),
            "note": "Approximate torso mapping; the shared array format has fewer torso slots than the full spine graph.",
        },
        12: {
            "figure_location": "base of neck / upper thorax",
            "project_segment_name": "Neck",
            "segment_name": _humanize_segment_name("Neck"),
        },
        13: {
            "figure_location": "head",
            "project_segment_name": "Head",
            "segment_name": _humanize_segment_name("Head"),
        },
        14: {
            "figure_location": "left shoulder",
            "project_segment_name": "L_Shoulder",
            "segment_name": _humanize_segment_name("L_Shoulder"),
        },
        15: {
            "figure_location": "left upper arm",
            "project_segment_name": "L_UpperArm",
            "segment_name": _humanize_segment_name("L_UpperArm"),
        },
        16: {
            "figure_location": "left forearm",
            "project_segment_name": "L_Forearm",
            "segment_name": _humanize_segment_name("L_Forearm"),
        },
        17: {
            "figure_location": "left hand / wrist",
            "project_segment_name": "L_Hand",
            "segment_name": _humanize_segment_name("L_Hand"),
        },
        18: {
            "figure_location": "right shoulder",
            "project_segment_name": "R_Shoulder",
            "segment_name": _humanize_segment_name("R_Shoulder"),
        },
        19: {
            "figure_location": "right upper arm",
            "project_segment_name": "R_UpperArm",
            "segment_name": _humanize_segment_name("R_UpperArm"),
        },
        20: {
            "figure_location": "right forearm",
            "project_segment_name": "R_Forearm",
            "segment_name": _humanize_segment_name("R_Forearm"),
        },
        21: {
            "figure_location": "right hand / wrist",
            "project_segment_name": "R_Hand",
            "segment_name": _humanize_segment_name("R_Hand"),
        },
    }
)


def _slot(
    joint_index: int,
    channels: str = "6D acc+gyro",
    *,
    note: str | None = None,
) -> dict[str, str | int]:
    joint_info = ARRAY_JOINT_TO_BODY_PART[joint_index]
    payload: dict[str, str | int] = {
        "joint_index": joint_index,
        "figure_location": str(joint_info["figure_location"]),
        "segment_name": str(joint_info["segment_name"]),
        "project_segment_name": str(joint_info["project_segment_name"]),
        "channels": channels,
    }
    joint_note = joint_info.get("note")
    if joint_note:
        payload["joint_note"] = str(joint_note)
    if note:
        payload["note"] = note
    return payload


def _sensor_slot(
    sensor_id: str,
    official_position: str,
    project_segment_name: str,
    channels: str = "6D acc+gyro",
    *,
    note: str | None = None,
) -> dict[str, str]:
    payload = {
        "sensor_id": str(sensor_id),
        "official_position": str(official_position),
        "figure_location": str(official_position).lower(),
        "segment_name": _humanize_segment_name(project_segment_name),
        "project_segment_name": str(project_segment_name),
        "channels": str(channels),
    }
    if note:
        payload["note"] = note
    return payload


OPENPACK_SENSOR_TO_BODY_PARTS = OrderedDict(
    {
        "atr01": _sensor_slot(
            "atr01",
            "Right Wrist",
            "R_Forearm",
            note="OpenPack mounts this IMU at the wrist; AnyMo maps wrist-mounted motion to the forearm segment.",
        ),
        "atr02": _sensor_slot(
            "atr02",
            "Left Wrist",
            "L_Forearm",
            note="OpenPack mounts this IMU at the wrist; AnyMo maps wrist-mounted motion to the forearm segment.",
        ),
        "atr03": _sensor_slot("atr03", "Right Upper Arm", "R_UpperArm"),
        "atr04": _sensor_slot("atr04", "Left Upper Arm", "L_UpperArm"),
    }
)


OPENPACK_DATASET_TO_BODY_PARTS = OrderedDict(
    {
        "OpenPack": list(OPENPACK_SENSOR_TO_BODY_PARTS.values()),
    }
)


OPENPACK_DATASET_TO_SEGMENT_NAMES = OrderedDict(
    (dataset, [str(entry["segment_name"]) for entry in entries])
    for dataset, entries in OPENPACK_DATASET_TO_BODY_PARTS.items()
)


ARRAY_HAR_DATASET_TO_BODY_PARTS = OrderedDict(
    {
        "PAMAP": [
            _slot(21, note="wrist-like slot in the shared template"),
            _slot(11, note="chest / torso-like slot in the shared template"),
            _slot(7, note="ankle / foot-like slot in the shared template"),
        ],
        "USCHAD": [
            _slot(5),
        ],
        "UCIHAR": [
            _slot(9, note="single torso / waist-like slot in the shared template"),
        ],
        "Opp_g": [
            _slot(10),
            _slot(19),
            _slot(20),
            _slot(15),
            _slot(16),
        ],
        "WISDM": [
            _slot(21),
        ],
        "DSADS": [
            _slot(11),
            _slot(21),
            _slot(17),
            _slot(6),
            _slot(2),
        ],
        "UTD-MHAD": [
            _slot(21, note="used when real_labels < 21"),
            _slot(5, note="used when real_labels >= 21"),
        ],
        "w-HAR": [
            _slot(7),
        ],
        "realworld": [
            _slot(14),
            _slot(16),
            _slot(13),
            _slot(3),
            _slot(1),
            _slot(15),
            _slot(9),
        ],
        "TNDA-HAR": [
            _slot(20),
            _slot(2),
            _slot(21),
            _slot(3),
            _slot(11),
        ],
    }
)


ARRAY_HAR_DATASET_TO_SEGMENT_NAMES = OrderedDict(
    (dataset, [str(entry["segment_name"]) for entry in entries])
    for dataset, entries in ARRAY_HAR_DATASET_TO_BODY_PARTS.items()
)


EGO_DATASET_TO_BODY_PARTS = OrderedDict(
    {
        "ego4d": [
            _slot(13, note="head-mounted IMU"),
        ],
        "mmea": [
            _slot(13, note="head-mounted IMU"),
        ],
        "egoexo4d": [
            _slot(13, note="head-mounted IMU"),
        ],
    }
)


EGO_DATASET_TO_SEGMENT_NAMES = OrderedDict(
    (dataset, [str(entry["segment_name"]) for entry in entries])
    for dataset, entries in EGO_DATASET_TO_BODY_PARTS.items()
)


DATASET_TO_BODY_PARTS = OrderedDict(
    [
        *ARRAY_HAR_DATASET_TO_BODY_PARTS.items(),
        *EGO_DATASET_TO_BODY_PARTS.items(),
        *OPENPACK_DATASET_TO_BODY_PARTS.items(),
    ]
)


DATASET_TO_SEGMENT_NAMES = OrderedDict(
    (dataset, [str(entry["segment_name"]) for entry in entries])
    for dataset, entries in DATASET_TO_BODY_PARTS.items()
)


__all__ = [
    "SEGMENT_NAMES_REFERENCE",
    "ARRAY_JOINT_TO_BODY_PART",
    "ARRAY_HAR_DATASET_TO_BODY_PARTS",
    "ARRAY_HAR_DATASET_TO_SEGMENT_NAMES",
    "EGO_DATASET_TO_BODY_PARTS",
    "EGO_DATASET_TO_SEGMENT_NAMES",
    "OPENPACK_SENSOR_TO_BODY_PARTS",
    "OPENPACK_DATASET_TO_BODY_PARTS",
    "OPENPACK_DATASET_TO_SEGMENT_NAMES",
    "DATASET_TO_BODY_PARTS",
    "DATASET_TO_SEGMENT_NAMES",
]
