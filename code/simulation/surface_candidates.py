"""Select dense body-surface candidate placements for 23 anatomical segments."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BASE_DIR = Path(os.environ.get("ANYMO_DATA_ROOT", "<PATH_TO_NYMERIA>"))
DEFAULT_SUMMARY_CSV = DEFAULT_BASE_DIR / "nymeria_mesh_60hz_summary.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_BASE_DIR / "body_surface_candidates.npz"
DEFAULT_NYMERIA_PKG_ROOT = Path(
    os.environ.get("NYMERIA_TOOLS_ROOT", "<PATH_TO_NYMERIA_TOOLS>")
)

BODY_SUBDIR = "body"
SYNC_SUBDIR = "multimodal_sync_60hz"
XDATA_FILENAME = "xdata.npz"
GLB_FILENAME = "xdata_blueman.glb"
MESH_60HZ_FILENAME = "momentum_mesh_60hz.npz"
SELECTION_MODE = "body_surface"
SELECTION_TOP_K = 2

SEGMENT_DEFINITIONS: list[tuple[str, tuple[str, ...]]] = [
    ("Pelvis", ("p_pelvis", "p_l_glut", "p_r_glut", "p_l_rect", "p_r_rect")),
    ("L5", ("b_spine0",)),
    ("L3", ("b_spine1",)),
    ("T12", ("b_spine2",)),
    ("T8", ("b_spine3", "p_sternum", "p_navel")),
    ("Neck", ("b_neck0", "p_neck_twist")),
    ("Head", ("b_head", "b_l_eye", "b_r_eye", "b_jaw", "b_teeth", "b_tongue")),
    ("R_Shoulder", ("b_r_shoulder", "p_r_delt", "p_r_scap")),
    ("R_UpperArm", ("b_r_arm", "p_r_arm_twist")),
    ("R_Forearm", ("b_r_forearm", "p_r_forearm_twist", "b_r_wrist_twist")),
    ("R_Hand", ("b_r_wrist", "b_r_index", "b_r_middle", "b_r_ring", "b_r_pinky", "b_r_thumb")),
    ("L_Shoulder", ("b_l_shoulder", "p_l_delt", "p_l_scap")),
    ("L_UpperArm", ("b_l_arm", "p_l_arm_twist")),
    ("L_Forearm", ("b_l_forearm", "p_l_forearm_twist", "b_l_wrist_twist")),
    ("L_Hand", ("b_l_wrist", "b_l_index", "b_l_middle", "b_l_ring", "b_l_pinky", "b_l_thumb")),
    ("R_UpperLeg", ("b_r_upleg", "p_r_upleg_twist")),
    ("R_LowerLeg", ("b_r_leg", "p_r_leg_twist")),
    (
        "R_Foot",
        ("b_r_foot_twist", "b_r_foot", "b_r_talocrural", "b_r_subtalar", "b_r_transversetarsal"),
    ),
    ("R_Toe", ("b_r_ball",)),
    ("L_UpperLeg", ("b_l_upleg", "p_l_upleg_twist")),
    ("L_LowerLeg", ("b_l_leg", "p_l_leg_twist")),
    (
        "L_Foot",
        ("b_l_foot_twist", "b_l_foot", "b_l_talocrural", "b_l_subtalar", "b_l_transversetarsal"),
    ),
    ("L_Toe", ("b_l_ball",)),
]


def _ensure_imports(nymeria_pkg_root: str) -> None:
    pkg_root = str(Path(nymeria_pkg_root))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _required_paths(sample_dir: Path) -> dict[str, Path]:
    return {
        "xdata_npz": sample_dir / BODY_SUBDIR / XDATA_FILENAME,
        "xdata_glb": sample_dir / BODY_SUBDIR / GLB_FILENAME,
        "mesh_60hz": sample_dir / SYNC_SUBDIR / MESH_60HZ_FILENAME,
    }


def discover_reference_sample_dir(summary_csv: Path) -> Path:
    with summary_csv.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") not in {"written", "skipped_existing"}:
                continue
            sample_dir = Path(row["sample_dir"])
            if all(path.is_file() for path in _required_paths(sample_dir).values()):
                return sample_dir
    raise RuntimeError(f"no usable reference sample found in {summary_csv}")


def load_reference_assets(sample_dir: Path, nymeria_pkg_root: str) -> dict[str, Any]:
    _ensure_imports(nymeria_pkg_root)
    from nymeria.body_motion_provider import BodyDataProvider

    paths = _required_paths(sample_dir)
    body_provider = BodyDataProvider(
        npzfile=str(paths["xdata_npz"]), glbfile=str(paths["xdata_glb"])
    )
    if body_provider.character is None or body_provider.momentum_template_mesh is None:
        raise RuntimeError("BodyDataProvider did not load character mesh")

    with np.load(paths["mesh_60hz"], allow_pickle=False) as data:
        template_vertices = np.asarray(data["template_vertices"], dtype=np.float32)
        template_faces = np.asarray(data["template_faces"], dtype=np.int32)

    return {
        "template_vertices": template_vertices,
        "template_faces": template_faces,
        "joint_names": list(body_provider.character.skeleton.joint_names),
        "skin_index": np.asarray(body_provider.character.skin_weights.index, dtype=np.int32),
        "skin_weight": np.asarray(body_provider.character.skin_weights.weight, dtype=np.float32),
    }


def segment_joint_indices(
    joint_names: list[str],
    definitions: list[tuple[str, tuple[str, ...]]] = SEGMENT_DEFINITIONS,
) -> list[np.ndarray]:
    joint_indices: list[np.ndarray] = []
    for segment_name, prefixes in definitions:
        indices = [
            idx
            for idx, name in enumerate(joint_names)
            if any(name == prefix or name.startswith(prefix) for prefix in prefixes)
        ]
        if not indices:
            raise RuntimeError(f"no Momentum joints matched segment {segment_name}")
        joint_indices.append(np.asarray(indices, dtype=np.int32))
    return joint_indices


def build_candidate_masks(
    skin_index: np.ndarray,
    skin_weight: np.ndarray,
    segment_joint_idx: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if skin_index.shape != skin_weight.shape:
        raise ValueError("skin index and weight shapes must match")

    num_segments = len(segment_joint_idx)
    num_vertices = skin_index.shape[0]
    positive = skin_weight > 0
    dominant_slot = np.argmax(skin_weight, axis=1)
    dominant_joint = skin_index[np.arange(num_vertices), dominant_slot]
    dominant_valid = skin_weight[np.arange(num_vertices), dominant_slot] > 0
    dominant_masks = np.zeros((num_segments, num_vertices), dtype=bool)
    any_masks = np.zeros((num_segments, num_vertices), dtype=bool)

    for segment_idx, joint_idx in enumerate(segment_joint_idx):
        joint_match = np.isin(skin_index, joint_idx)
        any_masks[segment_idx] = np.any(joint_match & positive, axis=1)
        dominant_masks[segment_idx] = dominant_valid & np.isin(dominant_joint, joint_idx)
    return dominant_masks, any_masks


def build_surface_masks(
    skin_index: np.ndarray,
    skin_weight: np.ndarray,
    segment_joint_idx: list[np.ndarray],
) -> np.ndarray:
    order = np.argsort(skin_weight, axis=1)[:, ::-1]
    top_slots = order[:, : min(SELECTION_TOP_K, skin_index.shape[1])]
    top_joints = np.take_along_axis(skin_index, top_slots, axis=1)
    top_weights = np.take_along_axis(skin_weight, top_slots, axis=1)
    masks = np.zeros((len(segment_joint_idx), skin_index.shape[0]), dtype=bool)
    for segment_idx, joint_idx in enumerate(segment_joint_idx):
        masks[segment_idx] = np.any(np.isin(top_joints, joint_idx) & (top_weights > 0), axis=1)
    return masks


def build_mesh_adjacency(num_vertices: int, faces: np.ndarray) -> list[list[int]]:
    neighbors: list[set[int]] = [set() for _ in range(num_vertices)]
    for face in np.asarray(faces, dtype=np.int64):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [sorted(items) for items in neighbors]


def hop_distance_to_core(
    adjacency: list[list[int]], core_mask: np.ndarray, bank_mask: np.ndarray
) -> np.ndarray:
    distances = np.full(len(adjacency), np.inf, dtype=np.float32)
    core_indices = np.flatnonzero(core_mask)
    if core_indices.size == 0:
        core_indices = np.flatnonzero(bank_mask)
    if core_indices.size == 0:
        raise ValueError("candidate bank must contain at least one vertex")

    queue: deque[int] = deque(int(index) for index in core_indices.tolist())
    for index in core_indices:
        distances[int(index)] = 0.0
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1.0
        for neighbor in adjacency[current]:
            if next_distance < distances[neighbor]:
                distances[neighbor] = next_distance
                queue.append(neighbor)

    bank_indices = np.flatnonzero(bank_mask)
    if not np.all(np.isfinite(distances[bank_indices])):
        finite = distances[bank_indices][np.isfinite(distances[bank_indices])]
        fill_value = float(finite.max() + 1.0) if finite.size else 0.0
        distances[bank_indices[~np.isfinite(distances[bank_indices])]] = fill_value
    return distances


def compute_sampling_weights(
    adjacency: list[list[int]],
    core_mask: np.ndarray,
    bank_mask: np.ndarray,
    inverse_density: np.ndarray,
    gamma: float,
) -> np.ndarray:
    distances = hop_distance_to_core(adjacency, core_mask, bank_mask)
    weights = np.zeros(bank_mask.shape[0], dtype=np.float32)
    weights[bank_mask] = np.exp(-gamma * distances[bank_mask]) * inverse_density[bank_mask]
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("sampling weights must sum to > 0")
    return weights / total


def pad_vertex_rows(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    max_length = max(int(row.shape[0]) for row in rows)
    ids = np.full((len(rows), max_length), -1, dtype=np.int32)
    mask = np.zeros((len(rows), max_length), dtype=bool)
    for row_index, row in enumerate(rows):
        ids[row_index, : row.shape[0]] = row
        mask[row_index, : row.shape[0]] = True
    return ids, mask


def build_body_surface_candidates(
    *,
    template_vertices: np.ndarray,
    template_faces: np.ndarray,
    joint_names: list[str],
    skin_index: np.ndarray,
    skin_weight: np.ndarray,
    gamma: float,
    seed: int,
    reference_sample_dir: Path,
) -> dict[str, np.ndarray]:
    segment_names = np.asarray([name for name, _ in SEGMENT_DEFINITIONS], dtype="<U32")
    joint_indices = segment_joint_indices(joint_names)
    dominant_masks, any_masks = build_candidate_masks(skin_index, skin_weight, joint_indices)
    surface_masks = build_surface_masks(skin_index, skin_weight, joint_indices)
    adjacency = build_mesh_adjacency(template_vertices.shape[0], template_faces)
    inverse_density = 1.0 / np.asarray(
        [max(1, len(neighbors)) for neighbors in adjacency], dtype=np.float32
    )

    weights = np.zeros((len(segment_names), template_vertices.shape[0]), dtype=np.float32)
    selected_rows: list[np.ndarray] = []
    selected_weight_rows: list[np.ndarray] = []
    for segment_idx in range(len(segment_names)):
        weights[segment_idx] = compute_sampling_weights(
            adjacency,
            dominant_masks[segment_idx],
            any_masks[segment_idx],
            inverse_density,
            gamma,
        )
        ids = np.flatnonzero(surface_masks[segment_idx]).astype(np.int32)
        selected_rows.append(ids)
        selected_weight_rows.append(weights[segment_idx, ids].astype(np.float32))

    selected_ids, selected_mask = pad_vertex_rows(selected_rows)
    selected_weights = np.zeros(selected_ids.shape, dtype=np.float32)
    for row_index, row in enumerate(selected_weight_rows):
        selected_weights[row_index, : row.shape[0]] = row

    return {
        "segment_names": segment_names,
        "selection_mode": np.asarray(SELECTION_MODE),
        "top_k": np.asarray(SELECTION_TOP_K, dtype=np.int32),
        "samples_per_segment": np.asarray(-1, dtype=np.int32),
        "template_vertices": np.asarray(template_vertices, dtype=np.float32),
        "template_faces": np.asarray(template_faces, dtype=np.int32),
        "candidate_any_masks": np.asarray(any_masks, dtype=bool),
        "candidate_dominant_masks": np.asarray(dominant_masks, dtype=bool),
        "candidate_any_counts": any_masks.sum(axis=1).astype(np.int32),
        "candidate_dominant_counts": dominant_masks.sum(axis=1).astype(np.int32),
        "sampling_weights": weights,
        "sampled_vertex_ids": np.empty((len(segment_names), 0), dtype=np.int32),
        "sampled_vertex_weights": np.empty((len(segment_names), 0), dtype=np.float32),
        "selected_vertex_ids": selected_ids,
        "selected_vertex_mask": selected_mask,
        "selected_vertex_weights": selected_weights,
        "reference_sample_dir": np.asarray(str(reference_sample_dir)),
        "seed": np.asarray(seed, dtype=np.int32),
        "gamma": np.asarray(gamma, dtype=np.float32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select dense body-surface candidate placements for 23 anatomical segments."
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--reference-sample-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--nymeria-pkg-root", type=Path, default=DEFAULT_NYMERIA_PKG_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_sample_dir = args.reference_sample_dir or discover_reference_sample_dir(
        args.summary_csv
    )
    assets = load_reference_assets(reference_sample_dir, str(args.nymeria_pkg_root))
    payload = build_body_surface_candidates(
        **assets,
        gamma=args.gamma,
        seed=args.seed,
        reference_sample_dir=reference_sample_dir,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_path, **payload)

    print(f"reference_sample_dir={reference_sample_dir}")
    print(f"output_path={args.output_path}")
    print(f"selection_mode={SELECTION_MODE} gamma={args.gamma} seed={args.seed}")
    for segment_idx, segment_name in enumerate(payload["segment_names"].tolist()):
        print(
            f"{segment_name}: dominant={int(payload['candidate_dominant_counts'][segment_idx])} "
            f"any_positive={int(payload['candidate_any_counts'][segment_idx])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
