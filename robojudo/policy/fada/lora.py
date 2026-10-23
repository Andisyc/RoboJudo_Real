# Adapted from TTT src/unilab/algos/torch/distill/fada/adaptation.py
# (revision 0f8a410b). Licensed under Apache-2.0; see LICENSE-TTT.
"""IDM-only Q/V attention LoRA, without training-framework dependencies."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterator, cast

import torch
from torch import nn
from torch.nn import functional as F

from .model import FADAPlannerIDMPolicy


@dataclass(frozen=True)
class FADALoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if any(not isinstance(name, str) or not name for name in self.target_modules):
            raise ValueError("LoRA target names must be non-empty strings")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("LoRA target names must be unique")


class FADALoRAQVMultiheadAttention(nn.Module):
    """Frozen packed QKV projections with trainable Q/V residual branches."""

    def __init__(self, base: nn.MultiheadAttention, *, rank: int,
                 alpha: float, dropout: float) -> None:
        super().__init__()
        if not base._qkv_same_embed_dim or base.in_proj_weight is None:
            raise ValueError("FADA Q/V LoRA requires packed same-dimension QKV attention")
        if base.in_proj_weight.shape != (3 * base.embed_dim, base.embed_dim):
            raise ValueError("FADA Q/V LoRA packed projection shape is incompatible")
        self.base = base.requires_grad_(False)
        self.batch_first = bool(base.batch_first)
        # The fused encoder would bypass the adapter's forward method.
        self.in_proj_bias = None
        factory = {"device": base.in_proj_weight.device, "dtype": base.in_proj_weight.dtype}
        self.lora_q_A = nn.Linear(base.embed_dim, rank, bias=False, **factory)
        self.lora_q_B = nn.Linear(rank, base.embed_dim, bias=False, **factory)
        self.lora_v_A = nn.Linear(base.embed_dim, rank, bias=False, **factory)
        self.lora_v_B = nn.Linear(rank, base.embed_dim, bias=False, **factory)
        self.adapter_dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        self.register_buffer("_identity_projection", torch.eye(base.embed_dim, **factory), persistent=False)
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.zeros_(self.lora_v_B.weight)

    def _project(self, inputs, weight, bias, lora_A, lora_B):
        residual = lora_B(lora_A(self.adapter_dropout(inputs)))
        return F.linear(inputs, weight, bias) + residual * self.scaling

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        batched = query.dim() == 3
        if self.batch_first and batched:
            query, key, value = (tensor.transpose(0, 1) for tensor in (query, key, value))
        q_weight, k_weight, v_weight = self.base.in_proj_weight.chunk(3, dim=0)
        if self.base.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.base.in_proj_bias.chunk(3, dim=0)
        projected_query = self._project(query, q_weight, q_bias, self.lora_q_A, self.lora_q_B)
        projected_key = F.linear(key, k_weight, k_bias)
        projected_value = self._project(value, v_weight, v_bias, self.lora_v_A, self.lora_v_B)
        identity_projection = cast(torch.Tensor, self._identity_projection)
        output, weights = F.multi_head_attention_forward(
            projected_query, projected_key, projected_value,
            self.base.embed_dim, self.base.num_heads,
            in_proj_weight=None, in_proj_bias=None,
            bias_k=self.base.bias_k, bias_v=self.base.bias_v,
            add_zero_attn=self.base.add_zero_attn, dropout_p=self.base.dropout,
            out_proj_weight=self.base.out_proj.weight, out_proj_bias=self.base.out_proj.bias,
            training=self.training, key_padding_mask=key_padding_mask,
            need_weights=need_weights, attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=identity_projection,
            k_proj_weight=identity_projection,
            v_proj_weight=identity_projection,
            average_attn_weights=average_attn_weights, is_causal=is_causal,
        )
        if self.batch_first and batched:
            output = output.transpose(0, 1)
        return output, weights


@dataclass(frozen=True)
class FADAAdaptedPolicy:
    policy: FADAPlannerIDMPolicy
    lora_config: FADALoRAConfig


def fada_adapter_named_parameters(policy: FADAPlannerIDMPolicy) -> Iterator[tuple[str, nn.Parameter]]:
    suffixes = tuple(f"lora_{projection}_{matrix}.weight" for projection in ("q", "v") for matrix in ("A", "B"))
    for name, parameter in policy.idm.named_parameters(prefix="idm"):
        if name.endswith(suffixes):
            yield name, parameter


def assert_fada_adaptation_parameter_ownership(policy: FADAPlannerIDMPolicy, config: FADALoRAConfig) -> None:
    expected = {
        f"idm.{target}.lora_{projection}_{matrix}.weight"
        for target in config.target_modules
        for projection in ("q", "v") for matrix in ("A", "B")
    }
    adapters = dict(fada_adapter_named_parameters(policy))
    trainable = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    if not expected or set(adapters) != expected or trainable != expected:
        raise ValueError("FADA adaptation must expose exactly the IDM Q/V LoRA parameters as trainable")


def inject_fada_idm_lora(policy: FADAPlannerIDMPolicy, config: FADALoRAConfig) -> FADAAdaptedPolicy:
    if any(isinstance(module, FADALoRAQVMultiheadAttention) for module in policy.idm.modules()):
        raise ValueError("FADA policy already contains LoRA adapters")
    targets = tuple(name for name, module in policy.idm.named_modules()
                    if name and isinstance(module, nn.MultiheadAttention))
    if not targets:
        raise ValueError("FADA IDM exposes no MultiheadAttention LoRA targets")
    if config.target_modules and config.target_modules != targets:
        raise ValueError("LoRA target module names do not match the complete IDM")
    resolved = replace(config, target_modules=targets)
    policy.requires_grad_(False)
    for name in targets:
        base = policy.idm.get_submodule(name)
        if not isinstance(base, nn.MultiheadAttention):
            raise ValueError(f"LoRA target is not MultiheadAttention: {name}")
        wrapped = FADALoRAQVMultiheadAttention(
            base, rank=resolved.rank, alpha=resolved.alpha, dropout=resolved.dropout,
        ).train(base.training)
        parent_name, _, child_name = name.rpartition(".")
        parent = policy.idm.get_submodule(parent_name) if parent_name else policy.idm
        setattr(parent, child_name, wrapped)
    assert_fada_adaptation_parameter_ownership(policy, resolved)
    return FADAAdaptedPolicy(policy, resolved)
