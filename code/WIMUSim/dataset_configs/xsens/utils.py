import numpy as np

from dataset_configs.xsens import consts


def generate_default_placement_params(B_rp_dict):
    children = {}
    for parent, child in consts.JOINT_PARENT_CHILD_PAIRS:
        children.setdefault(parent, []).append(child)

    rp = {}
    ro = {}
    for joint_name in consts.IMU_NAMES:
        child_names = children.get(joint_name, [])
        if child_names:
            ref_child = child_names[0]
            rel = np.asarray(B_rp_dict[(joint_name, ref_child)], dtype=np.float32)
            rp[(joint_name, joint_name)] = 0.5 * rel
        else:
            parent = consts.BODY_PARENT[joint_name]
            rel = np.asarray(B_rp_dict[(parent, joint_name)], dtype=np.float32)
            rp[(joint_name, joint_name)] = -0.25 * rel
        ro[(joint_name, joint_name)] = np.zeros(3, dtype=np.float32)

    return {"rp": rp, "ro": ro}


def generate_B_range(B_rp_dict, frac: float = 0.35, floor: float = 0.02):
    rp_range = {}
    for key, value in B_rp_dict.items():
        value = np.asarray(value, dtype=np.float32)
        delta = np.maximum(np.abs(value) * frac, floor).astype(np.float32)
        rp_range[key] = np.stack([value - delta, value + delta], axis=1).astype(
            np.float32
        )
    return {"rp": rp_range, "ro": {}}
