M36H_JOINT_ID_DICT = {
    "BASE": 0,
    "PELVIS": 0,
    "R_HIP": 1,
    "R_KNEE": 2,
    "R_ANKLE": 3,
    "L_HIP": 4,
    "L_KNEE": 5,
    "L_ANKLE": 6,
    "BELLY": 7,
    "NECK": 8,
    "R_CLAVICLE": 8,
    "L_CLAVICLE": 8,
    "NOSE": 9,
    "HEAD": 10,
    "L_SHOULDER": 11,
    "L_ELBOW": 12,
    "L_WRIST": 13,
    "R_SHOULDER": 14,
    "R_ELBOW": 15,
    "R_WRIST": 16,
}

M36H_JOINT_PARENT_CHILD_PAIRS = [
    ("BASE", "R_HIP"),
    ("R_HIP", "R_KNEE"),
    ("R_KNEE", "R_ANKLE"),
    ("BASE", "L_HIP"),
    ("L_HIP", "L_KNEE"),
    ("L_KNEE", "L_ANKLE"),
    ("BASE", "PELVIS"),
    ("PELVIS", "BELLY"),
    ("BELLY", "NECK"),
    ("BELLY", "R_CLAVICLE"),
    ("R_CLAVICLE", "R_SHOULDER"),
    ("R_SHOULDER", "R_ELBOW"),
    ("R_ELBOW", "R_WRIST"),
    ("BELLY", "L_CLAVICLE"),
    ("L_CLAVICLE", "L_SHOULDER"),
    ("L_SHOULDER", "L_ELBOW"),
    ("L_ELBOW", "L_WRIST"),
    ("NECK", "NOSE"),
    ("NOSE", "HEAD"),
    ("NECK", "HEAD"),
]

# For visualization with PyBullet
# e.g. Orientation of the "BASE" joint determines the orientation of the pelvis (a link defined in PyBullet environment).
# the list of links is defined HUMANOID_PARAMS_DEFAULT in wimusim/consts.py
JOINT_WIMUSIM_LINK_PAIRS = [
    ("BASE", "pelvis"),
    ("PELVIS", "torso_1"),
    ("BELLY", "torso_2"),
    ("NECK", "head"),
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

M36H_JOINT_PAIR_DICT = {
    (parent, child): (M36H_JOINT_ID_DICT[parent], M36H_JOINT_ID_DICT[child])
    for parent, child in M36H_JOINT_PARENT_CHILD_PAIRS
}

M36H_JOINT_PARENT_CHILD_DICT = {
    parent: child for (parent, child) in M36H_JOINT_PARENT_CHILD_PAIRS
}
M36H_JOINT_CHILD_PARENT_DICT = {
    child: parent for (parent, child) in M36H_JOINT_PARENT_CHILD_PAIRS
}
