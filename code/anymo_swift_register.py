from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import imu_token_utils
from anymo_swift_model import AnyMoRuntimeReplacementWrapper
from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.model.models.qwen import QwenLoader


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def resolve_anymo_runtime_config(model_kwargs=None) -> dict[str, object]:
    model_kwargs = dict(model_kwargs or {})
    artifact_path = model_kwargs.pop('anymo_codebook_artifact', None) or os.environ.get('ANYMO_CODEBOOK_ARTIFACT')
    if artifact_path is None:
        raise ValueError("AnyMo loader requires 'anymo_codebook_artifact' via model_kwargs or ANYMO_CODEBOOK_ARTIFACT env.")
    resume_checkpoint = model_kwargs.pop('anymo_resume_checkpoint', None)
    if resume_checkpoint is None:
        resume_checkpoint = os.environ.get('ANYMO_RESUME_CHECKPOINT')
    projection_hidden_size = model_kwargs.pop('anymo_projection_hidden_size', None)
    if projection_hidden_size is None:
        projection_hidden_size = os.environ.get('ANYMO_PROJECTION_HIDDEN_SIZE', 512)
    init_anymo_lm_head_from_projector = model_kwargs.pop('init_anymo_lm_head_from_projector', None)
    if init_anymo_lm_head_from_projector is None:
        init_anymo_lm_head_from_projector = os.environ.get('INIT_ANYMO_LM_HEAD_FROM_PROJECTOR', 'true')
    freeze_non_anymo_token_rows = model_kwargs.pop('freeze_non_anymo_token_rows', None)
    if freeze_non_anymo_token_rows is None:
        freeze_non_anymo_token_rows = os.environ.get('FREEZE_NON_ANYMO_TOKEN_ROWS', 'true')
    return {
        'artifact_path': str(artifact_path),
        'resume_checkpoint': str(resume_checkpoint) if resume_checkpoint else None,
        'projection_hidden_size': int(projection_hidden_size),
        'init_anymo_lm_head_from_projector': _to_bool(init_anymo_lm_head_from_projector),
        'freeze_non_anymo_token_rows': _to_bool(freeze_non_anymo_token_rows),
        'remaining_model_kwargs': model_kwargs,
    }


def resolve_anymo_special_tokens(new_special_tokens, codebook_payload) -> list[str]:
    if new_special_tokens:
        return list(new_special_tokens)
    return [
        codebook_payload['special_tokens']['bos'],
        codebook_payload['special_tokens']['eos'],
        *codebook_payload['imu_code_tokens'],
    ]


class _AnyMoLoaderMixin:
    def __init__(self, *args, new_special_tokens=None, model_kwargs=None, **kwargs):
        runtime_config = resolve_anymo_runtime_config(model_kwargs)
        self.anymo_codebook_artifact = Path(runtime_config['artifact_path'])
        self.anymo_resume_checkpoint = (
            Path(runtime_config['resume_checkpoint'])
            if runtime_config['resume_checkpoint']
            else None
        )
        self.projection_hidden_size = int(runtime_config['projection_hidden_size'])
        self.init_anymo_lm_head_from_projector = bool(runtime_config['init_anymo_lm_head_from_projector'])
        self.freeze_non_anymo_token_rows = bool(runtime_config['freeze_non_anymo_token_rows'])
        self.codebook_payload = torch.load(self.anymo_codebook_artifact, map_location='cpu')
        resolved_special_tokens = resolve_anymo_special_tokens(new_special_tokens, self.codebook_payload)
        super().__init__(*args, new_special_tokens=resolved_special_tokens, model_kwargs=runtime_config['remaining_model_kwargs'], **kwargs)

    def load(self):
        model, processor = super().load()
        if model is None:
            return None, processor
        tokenizer = self._get_tokenizer(processor)
        bos_token = self.codebook_payload['special_tokens']['bos']
        eos_token = self.codebook_payload['special_tokens']['eos']
        imu_code_tokens = self.codebook_payload['imu_code_tokens']
        imu_token_ids = tokenizer.convert_tokens_to_ids(imu_code_tokens)
        imu_bos_token_id = tokenizer.convert_tokens_to_ids(bos_token)
        imu_eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        trainable_ids = [imu_bos_token_id, imu_eos_token_id, *imu_token_ids]
        missing_tokens = [
            token for token, token_id in [(bos_token, imu_bos_token_id), (eos_token, imu_eos_token_id), *list(zip(imu_code_tokens, imu_token_ids))]
            if token_id is None or token_id < 0
        ]
        if missing_tokens:
            raise ValueError(f'AnyMo tokens were not registered in the tokenizer. Missing token ids for: {missing_tokens[:8]}')
        base_vocab_size = min(trainable_ids)
        wrapper = AnyMoRuntimeReplacementWrapper(
            llm=model,
            flat_codebook=self.codebook_payload['flat_codebook'],
            imu_token_ids=imu_token_ids,
            imu_bos_token_id=imu_bos_token_id,
            imu_eos_token_id=imu_eos_token_id,
            base_vocab_size=base_vocab_size,
            tokenizer_vocab_size=len(tokenizer),
            projection_hidden_size=self.projection_hidden_size,
            init_anymo_lm_head_from_projector=self.init_anymo_lm_head_from_projector,
            freeze_non_anymo_token_rows=self.freeze_non_anymo_token_rows,
        )
        wrapper.load_runtime_state(self.model_info.model_dir)
        if self.anymo_resume_checkpoint is not None:
            self._load_anymo_resume_checkpoint(wrapper, self.anymo_resume_checkpoint)
        return wrapper, processor

    @staticmethod
    def _resolve_anymo_resume_state_path(checkpoint_path: Path) -> Path:
        if checkpoint_path.is_file():
            return checkpoint_path
        state_path = checkpoint_path / 'pytorch_model.bin'
        if state_path.exists():
            return state_path
        raise FileNotFoundError(
            f'AnyMo resume checkpoint must be a pytorch_model.bin file or a directory containing one: {checkpoint_path}'
        )

    @classmethod
    def _load_anymo_resume_checkpoint(cls, wrapper: AnyMoRuntimeReplacementWrapper, checkpoint_path: Path) -> None:
        state_path = cls._resolve_anymo_resume_state_path(Path(checkpoint_path))
        state_dict = torch.load(state_path, map_location='cpu')
        missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                'AnyMo resume checkpoint did not match the runtime wrapper. '
                f'Missing keys: {missing[:8]}; unexpected keys: {unexpected[:8]}'
            )
        wrapper._align_runtime_state_to_llm()


class AnyMoQwenLoader(_AnyMoLoaderMixin, QwenLoader):
    pass


register_model(
    ModelMeta(
        model_type='anymo_qwen2_5',
        model_groups=[ModelGroup([Model('Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B')])],
        loader=AnyMoQwenLoader,
        template='qwen2_5',
        architectures=['Qwen2ForCausalLM'],
        additional_saved_files=['anymo_runtime_projector.pt'],
    ),
    exist_ok=True,
)
