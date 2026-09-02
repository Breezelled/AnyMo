from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


ANYMO_RUNTIME_STATE_FILENAME = "anymo_runtime_projector.pt"


def _resolve_llm_hidden_size(llm: nn.Module) -> int:
    config = getattr(llm, "config", None)
    if config is not None and hasattr(config, "hidden_size"):
        return int(getattr(config, "hidden_size"))
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "hidden_size"):
        return int(getattr(text_config, "hidden_size"))
    raise AttributeError("Could not resolve LLM hidden_size from config or config.text_config.")


class AnyMoRuntimeReplacementWrapper(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        flat_codebook: torch.Tensor,
        imu_token_ids: list[int],
        imu_bos_token_id: int,
        imu_eos_token_id: int,
        base_vocab_size: int,
        tokenizer_vocab_size: int,
        projection_hidden_size: int = 512,
        init_anymo_lm_head_from_projector: bool = True,
        freeze_non_anymo_token_rows: bool = True,
    ):
        super().__init__()
        if flat_codebook.ndim != 2:
            raise ValueError(f"Expected flat_codebook to have shape [K, D], got {tuple(flat_codebook.shape)}")
        self.llm = llm
        self.base_vocab_size = int(base_vocab_size)
        self.tokenizer_vocab_size = int(max(tokenizer_vocab_size, imu_bos_token_id + 1, imu_eos_token_id + 1, *(tid + 1 for tid in imu_token_ids)))
        self.imu_token_ids = [int(token_id) for token_id in imu_token_ids]
        self.imu_bos_token_id = int(imu_bos_token_id)
        self.imu_eos_token_id = int(imu_eos_token_id)
        self.init_anymo_lm_head_from_projector = bool(init_anymo_lm_head_from_projector)
        self.freeze_non_anymo_token_rows = bool(freeze_non_anymo_token_rows)
        self.register_buffer("flat_codebook", flat_codebook.detach().cpu().float())
        mapping = torch.full((self.tokenizer_vocab_size,), -1, dtype=torch.long)
        for global_code_id, token_id in enumerate(self.imu_token_ids):
            mapping[token_id] = int(global_code_id)
        self.register_buffer("token_id_to_code_index", mapping)
        hidden_size = _resolve_llm_hidden_size(llm)
        codebook_dim = int(flat_codebook.shape[-1])
        self.projector = nn.Sequential(
            nn.Linear(codebook_dim, int(projection_hidden_size)),
            nn.GELU(),
            nn.Linear(int(projection_hidden_size), hidden_size),
        )
        self.last_inputs_embeds: torch.Tensor | None = None
        self._freeze_hook_handles: list[Any] = []
        self._align_runtime_state_to_llm()
        if self.freeze_non_anymo_token_rows:
            self._install_row_freeze_hooks()
        if self.init_anymo_lm_head_from_projector:
            self.initialize_anymo_rows_from_projector()

    @property
    def config(self):
        return self.llm.config

    @property
    def device(self) -> torch.device:
        return self.get_input_embeddings().weight.device

    @property
    def runtime_dtype(self) -> torch.dtype:
        return self.get_input_embeddings().weight.dtype

    def _align_runtime_state_to_llm(self) -> None:
        device = self.device
        dtype = self.runtime_dtype
        self.projector.to(device=device, dtype=dtype)
        self.flat_codebook = self.flat_codebook.to(device=device)
        self.token_id_to_code_index = self.token_id_to_code_index.to(device=device)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            return self.llm.gradient_checkpointing_enable(**kwargs)
        return None

    def enable_input_require_grads(self):
        if hasattr(self.llm, "enable_input_require_grads"):
            return self.llm.enable_input_require_grads()
        return None

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def _trainable_token_ids(self) -> list[int]:
        return [*self.imu_token_ids, self.imu_bos_token_id, self.imu_eos_token_id]

    def _build_row_mask(self, num_rows: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.zeros(num_rows, device=device, dtype=dtype)
        for token_id in self._trainable_token_ids():
            if 0 <= token_id < num_rows:
                mask[token_id] = 1
        return mask

    def _install_row_freeze_hooks(self) -> None:
        seen: set[int] = set()
        params: list[torch.nn.Parameter] = []
        embed = self.get_input_embeddings()
        if embed is not None and getattr(embed, "weight", None) is not None:
            params.append(embed.weight)
        lm_head = self.get_output_embeddings()
        if lm_head is not None and getattr(lm_head, "weight", None) is not None:
            params.append(lm_head.weight)
        for param in params:
            if id(param) in seen:
                continue
            seen.add(id(param))

            def _hook(grad: torch.Tensor, self=self):
                if grad is None:
                    return None
                row_mask = self._build_row_mask(grad.shape[0], device=grad.device, dtype=grad.dtype)
                view_shape = (grad.shape[0],) + (1,) * (grad.ndim - 1)
                return grad * row_mask.view(view_shape)

            self._freeze_hook_handles.append(param.register_hook(_hook))

    def project_codebook_vectors(self, code_indices: torch.Tensor) -> torch.Tensor:
        code_indices = code_indices.to(device=self.flat_codebook.device, dtype=torch.long)
        codebook_vectors = self.flat_codebook.index_select(0, code_indices)
        return self.projector(codebook_vectors.to(device=self.device, dtype=self.runtime_dtype))

    def initialize_anymo_rows_from_projector(self) -> None:
        if not self.imu_token_ids:
            return
        code_indices = torch.arange(len(self.imu_token_ids), device=self.device, dtype=torch.long)
        projected = self.project_codebook_vectors(code_indices).detach()
        output_weight = self.get_output_embeddings().weight
        with torch.no_grad():
            output_weight[self.imu_token_ids] = projected.to(dtype=output_weight.dtype, device=output_weight.device)
            input_weight = self.get_input_embeddings().weight
            if input_weight.data_ptr() != output_weight.data_ptr():
                input_weight[self.imu_token_ids] = projected.to(dtype=input_weight.dtype, device=input_weight.device)

    def _replace_anymo_inputs(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        code_indices = torch.full_like(input_ids, -1)
        in_range = input_ids < self.token_id_to_code_index.numel()
        if in_range.any():
            code_indices[in_range] = self.token_id_to_code_index[input_ids[in_range]]
        imu_mask = code_indices >= 0
        if not bool(imu_mask.any()):
            return inputs_embeds
        projected = self.project_codebook_vectors(code_indices[imu_mask])
        replaced = inputs_embeds.clone()
        replaced[imu_mask] = projected.to(dtype=replaced.dtype, device=replaced.device)
        return replaced

    def _forward_backbone_last_hidden_state(
        self,
        *,
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        backbone = getattr(self.llm, "model", None)
        if backbone is None:
            backbone = getattr(self.llm, "transformer", None)
        if backbone is None:
            raise AttributeError(
                "The wrapped LLM does not expose a backbone `.model` or "
                "`.transformer` module."
            )
        kwargs.pop("labels", None)
        kwargs.pop("logits_to_keep", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("use_cache", None)
        outputs = backbone(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]

    def collect_anymo_diagnostics(self) -> dict[str, float]:
        output_weight = self.get_output_embeddings().weight.detach()
        imu_norm = float(output_weight[self.imu_token_ids].norm().item()) if self.imu_token_ids else 0.0
        old_norm = float(output_weight[: self.base_vocab_size].norm().item()) if self.base_vocab_size > 0 else 0.0
        return {
            "imu_rows": float(len(self.imu_token_ids)),
            "base_vocab_size": float(self.base_vocab_size),
            "imu_weight_norm": imu_norm,
            "base_weight_norm": old_norm,
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_backbone_last_hidden_state: bool = False,
        **kwargs,
    ):
        if input_ids is None:
            if return_backbone_last_hidden_state:
                if inputs_embeds is None:
                    raise ValueError("inputs_embeds is required when input_ids is None and returning backbone hidden states.")
                return self._forward_backbone_last_hidden_state(
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    **kwargs,
                )
            return self.llm(attention_mask=attention_mask, labels=labels, inputs_embeds=inputs_embeds, **kwargs)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        replaced = self._replace_anymo_inputs(input_ids, inputs_embeds)
        self.last_inputs_embeds = replaced.detach().cpu()
        if return_backbone_last_hidden_state:
            return self._forward_backbone_last_hidden_state(
                attention_mask=attention_mask,
                inputs_embeds=replaced,
                **kwargs,
            )
        return self.llm(attention_mask=attention_mask, labels=labels, inputs_embeds=replaced, **kwargs)

    def save_pretrained(self, output_dir: str | Path, **kwargs) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.llm, "save_pretrained"):
            self.llm.save_pretrained(output_dir, **kwargs)
        torch.save(
            {
                "projector_state_dict": self.projector.state_dict(),
                "base_vocab_size": self.base_vocab_size,
                "tokenizer_vocab_size": self.tokenizer_vocab_size,
                "imu_token_ids": self.imu_token_ids,
                "imu_bos_token_id": self.imu_bos_token_id,
                "imu_eos_token_id": self.imu_eos_token_id,
                "init_anymo_lm_head_from_projector": self.init_anymo_lm_head_from_projector,
                "freeze_non_anymo_token_rows": self.freeze_non_anymo_token_rows,
            },
            output_dir / ANYMO_RUNTIME_STATE_FILENAME,
        )

    def load_runtime_state(self, model_dir: str | Path) -> bool:
        runtime_path = Path(model_dir) / ANYMO_RUNTIME_STATE_FILENAME
        if not runtime_path.exists():
            return False
        obj = torch.load(runtime_path, map_location="cpu")
        self.projector.load_state_dict(obj["projector_state_dict"])
        self._align_runtime_state_to_llm()
        return True

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
