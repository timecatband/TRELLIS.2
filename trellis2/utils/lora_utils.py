from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn


DEFAULT_LORA_TARGETS = [
    "self_attn.to_qkv",
    "self_attn.to_out",
    "cross_attn.to_q",
    "cross_attn.to_kv",
    "cross_attn.to_out",
    "mlp.mlp.0",
    "mlp.mlp.2",
]


@dataclass
class LoraApplySummary:
    wrapped_modules: int
    trainable_params: int
    total_params: int


class LoRALinear(nn.Module):
    """
    Low-rank adapter wrapper for nn.Linear and SparseLinear-compatible modules.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.lora_down.to(device=base_layer.weight.device)
        self.lora_up.to(device=base_layer.weight.device)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_layer.bias

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        down = self.lora_down(self.lora_dropout(x))
        return self.lora_up(down) * self.scaling

    def forward(self, input: Any) -> Any:
        if hasattr(input, "feats") and hasattr(input, "replace"):
            base_out = self.base_layer(input)
            delta = self._delta(input.feats.to(self.lora_down.weight.dtype)).to(base_out.feats.dtype)
            return base_out.replace(base_out.feats + delta)

        base_out = self.base_layer(input)
        delta = self._delta(input.to(self.lora_down.weight.dtype)).to(base_out.dtype)
        return base_out + delta


def _matches_target(module_name: str, target_modules: Iterable[str]) -> bool:
    for target in target_modules:
        if module_name == target or module_name.endswith(f".{target}") or target in module_name:
            return True
        if fnmatch.fnmatch(module_name, target):
            return True
    return False


def _iter_parent_modules(module: nn.Module, prefix: str = ""):
    for child_name, child in module.named_children():
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        yield module, child_name, full_name, child
        yield from _iter_parent_modules(child, full_name)


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_down.weight.requires_grad = True
            module.lora_up.weight.requires_grad = True


def apply_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[Iterable[str]] = None,
    freeze_base: bool = True,
) -> LoraApplySummary:
    """
    Replace matching Linear modules with LoRA adapters.
    """
    target_modules = list(target_modules or DEFAULT_LORA_TARGETS)
    wrapped = 0
    for parent, child_name, full_name, child in list(_iter_parent_modules(model)):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and _matches_target(full_name, target_modules):
            setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            wrapped += 1

    if freeze_base:
        mark_only_lora_as_trainable(model)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return LoraApplySummary(wrapped, trainable_params, total_params)


def apply_lora_from_config(model: nn.Module, config: Optional[Mapping[str, Any]]) -> LoraApplySummary:
    config = dict(config or {})
    if not config.get("enabled", True):
        for param in model.parameters():
            param.requires_grad = False
        total_params = sum(param.numel() for param in model.parameters())
        return LoraApplySummary(0, 0, total_params)
    return apply_lora(
        model,
        rank=int(config.get("rank", 16)),
        alpha=float(config.get("alpha", 16.0)),
        dropout=float(config.get("dropout", 0.05)),
        target_modules=config.get("target_modules", DEFAULT_LORA_TARGETS),
        freeze_base=bool(config.get("freeze_base", True)),
    )


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_down." in name or ".lora_up." in name
    }


def save_lora_checkpoint(
    models: Mapping[str, nn.Module],
    path: str,
    *,
    step: int,
    lora_config: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "step": int(step),
        "lora_config": dict(lora_config or {}),
        "models": {name: lora_state_dict(model) for name, model in models.items()},
    }
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, path)


def load_lora_checkpoint(
    models: Mapping[str, nn.Module],
    path: str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "models" in payload:
        model_states = payload["models"]
    else:
        model_states = {"denoiser": payload}

    missing: Dict[str, Any] = {}
    for name, state in model_states.items():
        if name not in models:
            missing[name] = "model not present"
            continue
        result = models[name].load_state_dict(state, strict=strict)
        missing[name] = {
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }
    return {
        "step": payload.get("step"),
        "lora_config": payload.get("lora_config", {}),
        "load_result": missing,
    }


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total
