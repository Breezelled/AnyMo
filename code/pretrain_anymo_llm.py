from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

import config


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
DEFAULT_DATASET_DIR = config.MOTION_LANGUAGE_DATA_DIR
DEFAULT_CACHE_ROOT = config.CACHE_ROOT
DEFAULT_HF_HOME = DEFAULT_CACHE_ROOT / 'huggingface'
DEFAULT_HF_HUB_CACHE = DEFAULT_HF_HOME / 'hub'
DEFAULT_MODELSCOPE_CACHE = DEFAULT_CACHE_ROOT / 'modelscope'

BACKBONE_SPECS = {
    'qwen2_5': {
        'model': 'Qwen/Qwen2.5-0.5B',
        'model_type': 'anymo_qwen2_5',
        'output_dir': config.MOTION_LANGUAGE_MODEL_DIR,
        'extra_args': [],
    },
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Launch ms-swift AnyMo-token pretraining with runtime projector replacement.')
    parser.add_argument('--swift-bin', type=str, default='swift')
    parser.add_argument('--backbone', type=str, choices=sorted(BACKBONE_SPECS), default='qwen2_5')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument('--val-dataset', type=Path, default=None)
    parser.add_argument('--use-val', dest='use_val', action='store_true')
    parser.add_argument('--no-val', dest='use_val', action='store_false')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--eval-steps', type=int, default=2000)
    parser.add_argument('--eval-strategy', type=str, default='steps')
    parser.add_argument('--save-strategy', type=str, default='best')
    parser.add_argument('--save-total-limit', type=int, default=1)
    parser.add_argument('--load-best-model-at-end', type=str, default='true')
    parser.add_argument('--metric-for-best-model', type=str, default='loss')
    parser.add_argument('--greater-is-better', type=str, default='false')
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--torch-dtype', type=str, default='bfloat16')
    parser.add_argument('--check-model', type=str, default='false')
    parser.add_argument('--attn-impl', '--attn_impl', dest='attn_impl', type=str, default=None)
    parser.add_argument('--gradient-checkpointing', '--gradient_checkpointing', dest='gradient_checkpointing', type=str, default='false')
    parser.add_argument('--safe-serialization', type=str, default='false')
    parser.add_argument('--projection-hidden-size', type=int, default=512)
    parser.add_argument('--init-anymo-lm-head-from-projector', type=str, default='true')
    parser.add_argument('--use-hf', type=str, default='true')
    parser.add_argument('--hf-home', type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument('--hf-hub-cache', type=Path, default=DEFAULT_HF_HUB_CACHE)
    parser.add_argument('--modelscope-cache', type=Path, default=DEFAULT_MODELSCOPE_CACHE)
    parser.add_argument('--dry-run', action='store_true')
    parser.set_defaults(use_val=False)
    return parser


def resolve_backbone_config(args: argparse.Namespace) -> dict[str, object]:
    spec = BACKBONE_SPECS[args.backbone]
    return {
        'model': args.model or spec['model'],
        'model_type': spec['model_type'],
        'output_dir': Path(args.output_dir) if args.output_dir is not None else spec['output_dir'],
        'extra_args': list(spec['extra_args']),
    }


def build_swift_pt_command(args: argparse.Namespace) -> list[str]:
    dataset_dir = Path(args.dataset_dir)
    train_path = dataset_dir / 'train.jsonl'
    codebook_artifact = dataset_dir / 'imu_codebook_lookup.pt'
    model_kwargs = {
        'anymo_codebook_artifact': str(codebook_artifact),
        'anymo_projection_hidden_size': int(args.projection_hidden_size),
        'init_anymo_lm_head_from_projector': str(args.init_anymo_lm_head_from_projector).lower() == 'true',
    }
    code_dir = CODE_DIR
    backbone_config = resolve_backbone_config(args)
    use_val = bool(args.use_val)
    val_path = Path(args.val_dataset) if args.val_dataset is not None else dataset_dir / 'val.jsonl'
    eval_strategy = str(args.eval_strategy) if use_val else 'no'
    save_strategy = str(args.save_strategy)
    if not use_val and save_strategy == 'best':
        save_strategy = 'epoch'
    cmd = [
        args.swift_bin,
        'pt',
        '--model',
        str(backbone_config['model']),
        '--model_type',
        str(backbone_config['model_type']),
        '--dataset',
        str(train_path),
        '--output_dir',
        str(backbone_config['output_dir']),
        '--tuner_type',
        'full',
        '--num_train_epochs',
        str(args.epochs),
        '--learning_rate',
        str(args.learning_rate),
        '--per_device_train_batch_size',
        str(args.batch_size),
        '--eval_strategy',
        str(eval_strategy),
        '--save_strategy',
        str(save_strategy),
        '--save_total_limit',
        str(args.save_total_limit),
        '--max_length',
        str(args.max_length),
        '--torch_dtype',
        str(args.torch_dtype),
        '--check_model',
        str(args.check_model),
        '--custom_register_path',
        str(code_dir / 'anymo_swift_register.py'),
        '--external_plugins',
        str(code_dir / 'anymo_swift_plugin.py'),
        '--callbacks',
        'anymo_row_freeze',
        '--model_kwargs',
        json.dumps(model_kwargs),
    ]
    if use_val:
        cmd.extend([
            '--val_dataset',
            str(val_path),
            '--per_device_eval_batch_size',
            str(args.eval_batch_size),
            '--eval_steps',
            str(args.eval_steps),
            '--load_best_model_at_end',
            str(args.load_best_model_at_end),
            '--metric_for_best_model',
            str(args.metric_for_best_model),
            '--greater_is_better',
            str(args.greater_is_better),
        ])
    if args.attn_impl is not None:
        cmd.extend(['--attn_impl', str(args.attn_impl)])
    cmd.extend(['--gradient_checkpointing', str(args.gradient_checkpointing)])
    cmd.extend(['--safe_serialization', str(args.safe_serialization)])
    cmd.extend(str(arg) for arg in backbone_config['extra_args'])
    return cmd


def build_runtime_env(args: argparse.Namespace) -> dict[str, str]:
    return {
        'USE_HF': '1' if str(args.use_hf).lower() == 'true' else '0',
        'HF_HOME': str(args.hf_home),
        'HUGGINGFACE_HUB_CACHE': str(args.hf_hub_cache),
        'MODELSCOPE_CACHE': str(args.modelscope_cache),
    }


def ensure_runtime_dirs(runtime_env: dict[str, str]) -> None:
    for key in ('HF_HOME', 'HUGGINGFACE_HUB_CACHE', 'MODELSCOPE_CACHE'):
        Path(runtime_env[key]).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = build_argparser().parse_args()
    cmd = build_swift_pt_command(args)
    runtime_env = build_runtime_env(args)
    ensure_runtime_dirs(runtime_env)
    merged_env = os.environ.copy()
    merged_env.update(runtime_env)
    if args.dry_run:
        env_prefix = ' '.join(f'{key}={shlex.quote(value)}' for key, value in runtime_env.items())
        print(f"{env_prefix} {' '.join(shlex.quote(part) for part in cmd)}")
        return 0
    subprocess.run(cmd, check=True, env=merged_env)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
