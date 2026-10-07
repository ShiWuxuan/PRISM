# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified 2026: extracted the PRISM implementation and removed framework dependencies.

from __future__ import annotations

import json
import os
from typing import Iterable

import torch

from .merge import LORA_A_SUFFIXES, LORA_B_SUFFIXES, _match_suffix

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
SAFE_ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
_ORDER_INSENSITIVE_KEYS = {"target_modules", "modules_to_save", "layers_to_transform", "layers_pattern"}


def find_adapter_weights(adapter_dir: str) -> tuple[str, str]:
    safe_path = os.path.join(adapter_dir, SAFE_ADAPTER_WEIGHTS_NAME)
    bin_path = os.path.join(adapter_dir, ADAPTER_WEIGHTS_NAME)
    if os.path.isfile(safe_path):
        return safe_path, "safetensors"
    if os.path.isfile(bin_path):
        return bin_path, "bin"
    raise FileNotFoundError(f"Missing adapter weights in {adapter_dir}.")


def load_adapter_config(adapter_dir: str) -> dict:
    config_path = os.path.join(adapter_dir, ADAPTER_CONFIG_NAME)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing {ADAPTER_CONFIG_NAME} in {adapter_dir}.")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Adapter config must be a JSON object: {config_path}")
    return config


def save_adapter_config(output_dir: str, config: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, ADAPTER_CONFIG_NAME)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _normalize_config(config: dict) -> dict:
    normalized = json.loads(json.dumps(config))
    for key in _ORDER_INSENSITIVE_KEYS:
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = sorted(value)
    return normalized


def ensure_compatible_configs(configs: Iterable[dict], ignore_keys: Iterable[str] = ()) -> dict:
    configs = list(configs)
    if not configs:
        raise ValueError("No adapter configs provided.")
    ignore_keys = set(ignore_keys)
    base_filtered = _filtered_config(configs[0], ignore_keys)
    for index, config in enumerate(configs[1:], start=1):
        current_filtered = _filtered_config(config, ignore_keys)
        if base_filtered != current_filtered:
            diff_keys = _diff_keys(base_filtered, current_filtered)
            raise ValueError(f"Adapter config mismatch at index {index}: {', '.join(diff_keys)}")
    return configs[0]


def _filtered_config(config: dict, ignore_keys: set[str]) -> dict:
    normalized = _normalize_config(config)
    for key in ignore_keys:
        normalized.pop(key, None)
    return normalized


def _diff_keys(left: dict, right: dict) -> list[str]:
    keys = set(left.keys()) | set(right.keys())
    diff = [key for key in sorted(keys) if left.get(key) != right.get(key)]
    return diff


def load_adapter_state(adapter_dir: str) -> tuple[dict[str, torch.Tensor], str]:
    weights_path, fmt = find_adapter_weights(adapter_dir)
    if fmt == "safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unexpected adapter weights format in {weights_path}.")
    return state_dict, fmt


def save_adapter_state(output_dir: str, state_dict: dict[str, torch.Tensor], save_safetensors: bool) -> str:
    os.makedirs(output_dir, exist_ok=True)
    packed_state = {key: tensor.contiguous() for key, tensor in state_dict.items()}
    if save_safetensors:
        from safetensors.torch import save_file

        weights_path = os.path.join(output_dir, SAFE_ADAPTER_WEIGHTS_NAME)
        save_file(packed_state, weights_path)
    else:
        weights_path = os.path.join(output_dir, ADAPTER_WEIGHTS_NAME)
        torch.save(packed_state, weights_path)
    return weights_path


def infer_lora_rank(state_dict: dict[str, torch.Tensor]) -> int | None:
    ranks: set[int] = set()
    for key, tensor in state_dict.items():
        if _match_suffix(key, LORA_A_SUFFIXES):
            ranks.add(tensor.shape[0])
        elif _match_suffix(key, LORA_B_SUFFIXES):
            ranks.add(tensor.shape[1])
    if not ranks:
        return None
    if len(ranks) > 1:
        raise ValueError(f"Inconsistent LoRA ranks detected: {sorted(ranks)}")
    return next(iter(ranks))


def update_lora_config_for_rank(config: dict, state_dict: dict[str, torch.Tensor]) -> dict:
    new_rank = infer_lora_rank(state_dict)
    if new_rank is None:
        return config
    old_rank = config.get("r")
    if old_rank != new_rank and (
        config.get("use_rslora") or config.get("use_dora") or config.get("rank_pattern") or config.get("alpha_pattern")
    ):
        raise ValueError("Rank changes require standard LoRA without per-layer rank/alpha patterns.")
    if old_rank is None or old_rank == new_rank:
        return config
    updated = dict(config)
    updated["r"] = new_rank
    if "lora_alpha" in updated and old_rank:
        updated["lora_alpha"] = float(updated["lora_alpha"]) * float(new_rank) / float(old_rank)
        if isinstance(config.get("lora_alpha"), int):
            updated["lora_alpha"] = int(round(updated["lora_alpha"]))
    return updated
