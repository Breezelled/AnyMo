from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = Path(os.environ.get("ANYMO_DATA_ROOT", PROJECT_DIR / "data")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("ANYMO_OUTPUT_ROOT", PROJECT_DIR / "outputs")).expanduser()
CACHE_ROOT = Path(os.environ.get("ANYMO_CACHE_ROOT", PROJECT_DIR / ".cache")).expanduser()
PRETRAIN_LOSS_CHOICES = ("masked_mse", "predictive_infonce", "predictive_infonce_tokenwise", "graphview_infonce")
SIMULATION_SUMMARY = DEFAULT_BASE_DIR / "geometry_aware_imu_summary.csv"
ENCODER_OUTPUT_DIR = OUTPUT_ROOT / "encoder"
DEFAULT_ENCODER_CHECKPOINT = ENCODER_OUTPUT_DIR / "encoder_best.pt"
TOKENIZER_OUTPUT_DIR = OUTPUT_ROOT / "tokenizer"
MOTION_LANGUAGE_DATA_DIR = OUTPUT_ROOT / "motion_language_pretraining"
INSTRUCTION_DATA_DIR = OUTPUT_ROOT / "instruction_tuning"
HAR_EVALUATION_DIR = OUTPUT_ROOT / "har_evaluation"
NYMERIA_HELDOUT_DIR = OUTPUT_ROOT / "nymeria_heldout_evaluation"
EGOEXO4D_PREPARED_DIR = OUTPUT_ROOT / "egoexo4d_prepared"
EGOEXO4D_EVALUATION_DIR = OUTPUT_ROOT / "egoexo4d_zero_shot_evaluation"
MOTION_LANGUAGE_MODEL_DIR = OUTPUT_ROOT / "motion_language_model"
ANYMO_MODEL_DIR = OUTPUT_ROOT / "anymo"


@dataclass
class TrainSTGCNConfig:
    summary_csv: Path = SIMULATION_SUMMARY
    base_dir: Path = DEFAULT_BASE_DIR
    output_dir: Path = ENCODER_OUTPUT_DIR
    split_cache: Path = ENCODER_OUTPUT_DIR / "data_split.json"
    candidate_name: str = "body_surface_placements"
    real1_device_suffix: str = "imu_1202_1"
    real2_device_suffix: str = "imu_1202_2"
    sample_rate: int = 60
    window_seconds: float = 5.0
    window_size: int = 300
    num_nodes: int = 23
    in_channels: int = 6
    latent_dim: int = 256
    batch_size: int = 64
    eval_batch_size: int = 64
    epochs: int = 10
    patience: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.0
    num_workers: int = 4
    seed: int = 42
    dropout: float = 0.0
    use_sigreg: bool = False
    pretrain_loss: str = "predictive_infonce"
    lambda_sigreg: float = 0.09
    infonce_temperature: float = 0.1
    surface_rotation_augment: bool = True
    surface_rotation_inplane_max_deg: float = 180.0
    surface_rotation_tilt_max_deg: float = 10.0
    predictor_depth: int = 6
    predictor_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    graphview_proj_dim: int = 128
    max_visible_nodes: int = 5


@dataclass
class EvalConfig:
    encoder_ckpt: Path = DEFAULT_ENCODER_CHECKPOINT
    summary_csv: Path = SIMULATION_SUMMARY
    base_dir: Path = DEFAULT_BASE_DIR
    split_cache: Path = ENCODER_OUTPUT_DIR / "evaluation_split.json"
    output_dir: Path = ENCODER_OUTPUT_DIR / "evaluation"
    candidate_name: str = "body_surface_placements"
    real1_device_suffix: str = "imu_1202_1"
    real2_device_suffix: str = "imu_1202_2"
    sample_rate: int = 60
    window_seconds: float = 5.0
    window_size: int = 300
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 8
    seed: int = 42
    dropout: float = 0.0
    use_sigreg: bool = False
    pretrain_loss: str = "predictive_infonce"
    max_visible_nodes: int = 5
    eval_train_rotation_augment: bool = True
    eval_train_rotation_augment_prob: float = 0.5


TrainConfig = TrainSTGCNConfig
