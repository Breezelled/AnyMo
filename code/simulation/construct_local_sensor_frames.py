#!/usr/bin/env python3
"""Construct tangent/binormal/normal local frames at body-surface placements."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_BASE_DIR = Path(os.environ.get("ANYMO_DATA_ROOT", "<PATH_TO_NYMERIA>"))
DEFAULT_INPUT_NPZ = DEFAULT_BASE_DIR / "body_surface_candidates.npz"
DEFAULT_OUTPUT_NPZ = DEFAULT_BASE_DIR / "body_surface_candidates_with_local_frames.npz"

SEGMENT_CHILDREN = {
    "Pelvis": ["L5", "R_UpperLeg", "L_UpperLeg"],
    "L5": ["L3"],
    "L3": ["T12"],
    "T12": ["T8"],
    "T8": ["Neck", "R_Shoulder", "L_Shoulder"],
    "Neck": ["Head"],
    "Head": [],
    "R_Shoulder": ["R_UpperArm"],
    "R_UpperArm": ["R_Forearm"],
    "R_Forearm": ["R_Hand"],
    "R_Hand": [],
    "L_Shoulder": ["L_UpperArm"],
    "L_UpperArm": ["L_Forearm"],
    "L_Forearm": ["L_Hand"],
    "L_Hand": [],
    "R_UpperLeg": ["R_LowerLeg"],
    "R_LowerLeg": ["R_Foot"],
    "R_Foot": ["R_Toe"],
    "R_Toe": [],
    "L_UpperLeg": ["L_LowerLeg"],
    "L_LowerLeg": ["L_Foot"],
    "L_Foot": ["L_Toe"],
    "L_Toe": [],
}
SEGMENT_PARENT = {child: parent for parent, children in SEGMENT_CHILDREN.items() for child in children}
GLOBAL_REFERENCE_AXES = (
    np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct tangent/binormal/normal local sensor frames for selected body-surface placements."
    )
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_selection_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def unit_vector(vec: np.ndarray, *, name: str) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"{name} has near-zero norm")
    return (vec / norm).astype(np.float32)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    normals = np.zeros_like(vertices, dtype=np.float32)
    tris = vertices[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]).astype(np.float32)
    for face_idx, face in enumerate(faces):
        normals[face] += face_normals[face_idx]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)):
        raise ValueError("vertex normals contain non-finite values")
    if np.any(norms < 1e-8):
        raise ValueError("some template vertices have degenerate normals")
    return (normals / norms).astype(np.float32)


def compute_segment_centroids(selection_payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    segment_names = [str(x) for x in selection_payload["segment_names"].tolist()]
    template_vertices = np.asarray(selection_payload["template_vertices"], dtype=np.float32)
    selected_ids = np.asarray(selection_payload["selected_vertex_ids"], dtype=np.int32)
    selected_mask = np.asarray(selection_payload["selected_vertex_mask"], dtype=bool)
    selected_weights = np.asarray(selection_payload["selected_vertex_weights"], dtype=np.float32)

    centroids: dict[str, np.ndarray] = {}
    for seg_idx, segment_name in enumerate(segment_names):
        mask = selected_mask[seg_idx]
        ids = selected_ids[seg_idx][mask]
        weights = selected_weights[seg_idx][mask]
        if ids.size == 0:
            raise ValueError(f"segment {segment_name} has no selected vertices")
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError(f"segment {segment_name} has invalid centroid weights")
        centroid = np.average(template_vertices[ids], axis=0, weights=weights).astype(np.float32)
        centroids[segment_name] = centroid
    return centroids


def _find_available_descendant(segment_name: str, centroids: dict[str, np.ndarray]) -> str | None:
    for child in SEGMENT_CHILDREN.get(segment_name, []):
        if child in centroids:
            return child
        found = _find_available_descendant(child, centroids)
        if found is not None:
            return found
    return None


def _find_available_ancestor(segment_name: str, centroids: dict[str, np.ndarray]) -> str | None:
    parent = SEGMENT_PARENT.get(segment_name)
    while parent is not None:
        if parent in centroids:
            return parent
        parent = SEGMENT_PARENT.get(parent)
    return None


def compute_segment_axes(segment_names: list[str], centroids: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    axes: dict[str, np.ndarray] = {}
    for segment_name in segment_names:
        if segment_name == "Pelvis":
            target_name = _find_available_descendant("L5", centroids) if "L5" not in centroids else "L5"
            if target_name is None:
                raise ValueError("Pelvis requires an available descendant along the spine chain")
            raw_axis = centroids[target_name] - centroids[segment_name]
        else:
            descendant = _find_available_descendant(segment_name, centroids)
            if descendant is not None:
                raw_axis = centroids[descendant] - centroids[segment_name]
            else:
                ancestor = _find_available_ancestor(segment_name, centroids)
                if ancestor is None:
                    raise ValueError(f"segment {segment_name} has no available neighbor to define anatomical axis")
                raw_axis = centroids[segment_name] - centroids[ancestor]
        axes[segment_name] = unit_vector(raw_axis, name=f"{segment_name} anatomical axis")
    return axes


def project_tangent(axis: np.ndarray, normal: np.ndarray) -> np.ndarray:
    tangent = np.asarray(axis, dtype=np.float32) - float(np.dot(axis, normal)) * np.asarray(normal, dtype=np.float32)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm >= 1e-8 and np.isfinite(tangent_norm):
        return (tangent / tangent_norm).astype(np.float32)
    for ref_axis in GLOBAL_REFERENCE_AXES:
        tangent = ref_axis - float(np.dot(ref_axis, normal)) * normal
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm >= 1e-8 and np.isfinite(tangent_norm):
            return (tangent / tangent_norm).astype(np.float32)
    raise ValueError("failed to construct tangent direction from projected anatomical axis")


def attach_local_sensor_frames(selection_payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    segment_names = [str(x) for x in selection_payload["segment_names"].tolist()]
    template_faces = np.asarray(selection_payload["template_faces"], dtype=np.int32)
    template_vertices = np.asarray(selection_payload["template_vertices"], dtype=np.float32)
    selected_ids = np.asarray(selection_payload["selected_vertex_ids"], dtype=np.int32)
    selected_mask = np.asarray(selection_payload["selected_vertex_mask"], dtype=bool)

    vertex_normals = compute_vertex_normals(template_vertices, template_faces)
    centroids = compute_segment_centroids(selection_payload)
    segment_axes = compute_segment_axes(segment_names, centroids)

    rot = np.tile(np.eye(3, dtype=np.float32), (selected_ids.shape[0], selected_ids.shape[1], 1, 1))
    normals = np.zeros((selected_ids.shape[0], selected_ids.shape[1], 3), dtype=np.float32)
    tangents = np.zeros((selected_ids.shape[0], selected_ids.shape[1], 3), dtype=np.float32)

    for seg_idx, segment_name in enumerate(segment_names):
        axis = segment_axes[segment_name]
        for placement_idx, is_valid in enumerate(selected_mask[seg_idx]):
            if not bool(is_valid):
                continue
            vertex_id = int(selected_ids[seg_idx, placement_idx])
            normal = unit_vector(vertex_normals[vertex_id], name=f"{segment_name} vertex normal")
            tangent = project_tangent(axis, normal)
            binormal = unit_vector(np.cross(normal, tangent), name=f"{segment_name} binormal")
            tangent = unit_vector(np.cross(binormal, normal), name=f"{segment_name} tangent re-orthogonalized")
            rotation = np.stack([tangent, binormal, normal], axis=1).astype(np.float32)
            if np.linalg.det(rotation) <= 0.0:
                raise ValueError(f"{segment_name} rotation matrix is not right-handed")
            rot[seg_idx, placement_idx] = rotation
            normals[seg_idx, placement_idx] = normal
            tangents[seg_idx, placement_idx] = tangent

    payload = dict(selection_payload)
    payload["selected_vertex_rotation_matrix"] = rot
    payload["selected_vertex_normal"] = normals
    payload["selected_vertex_tangent"] = tangents
    return payload


def main() -> int:
    args = parse_args()
    if args.output_npz.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_npz}. Pass --overwrite to replace it.")
    payload = load_selection_payload(args.input_npz)
    output_payload = attach_local_sensor_frames(payload)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **output_payload)
    print(f"surface_placements_with_local_frames={args.output_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
