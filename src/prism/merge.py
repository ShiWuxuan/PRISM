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

from typing import Iterable

import torch

LORA_A_SUFFIXES = ("lora_A.weight", "lora_embedding_A.weight")
LORA_B_SUFFIXES = ("lora_B.weight", "lora_embedding_B.weight")


def _match_suffix(key: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        if key.endswith(suffix):
            return suffix
    return None


def _is_lora_key(key: str) -> bool:
    return _match_suffix(key, LORA_A_SUFFIXES + LORA_B_SUFFIXES) is not None


def _collect_lora_pairs(keys: Iterable[str]) -> dict[str, dict[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    for key in keys:
        suffix = _match_suffix(key, LORA_A_SUFFIXES)
        if suffix:
            prefix = key[: -len(suffix)]
            pairs.setdefault(prefix, {})["a_key"] = key
            continue
        suffix = _match_suffix(key, LORA_B_SUFFIXES)
        if suffix:
            prefix = key[: -len(suffix)]
            pairs.setdefault(prefix, {})["b_key"] = key
    for prefix, pair in pairs.items():
        if "a_key" not in pair or "b_key" not in pair:
            raise ValueError(f"Missing LoRA A/B weights for prefix: {prefix}")
    return pairs


def _resolve_device(device: str | None) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _average_non_lora(state_dicts: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    ref = state_dicts[0][key]
    if not ref.is_floating_point():
        for other_state in state_dicts[1:]:
            if not torch.equal(ref, other_state[key]):
                raise ValueError(f"Non-floating tensor {key} differs across adapters.")
        return ref.clone()
    stacked = torch.stack([state[key].to(torch.float32) for state in state_dicts], dim=0)
    return stacked.mean(dim=0).to(dtype=ref.dtype)


def _validate_state_dicts(state_dicts: Iterable[dict[str, torch.Tensor]]) -> list[str]:
    state_dicts = list(state_dicts)
    if not state_dicts:
        raise ValueError("No adapter weights provided.")
    for state in state_dicts:
        if not isinstance(state, dict) or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise ValueError("Each adapter state must map string keys to tensors.")
    base_keys = set(state_dicts[0].keys())
    for index, state in enumerate(state_dicts[1:], start=1):
        keys = set(state.keys())
        if keys != base_keys:
            missing = sorted(base_keys - keys)
            extra = sorted(keys - base_keys)
            raise ValueError(
                f"Adapter weights keys mismatch at index {index}; missing={missing or '[]'}, extra={extra or '[]'}."
            )
    for key in base_keys:
        shape = state_dicts[0][key].shape
        for index, state in enumerate(state_dicts[1:], start=1):
            if state[key].shape != shape:
                raise ValueError(
                    f"Adapter weights shape mismatch for {key} at index {index}: {state[key].shape} vs {shape}."
                )
    return sorted(base_keys)


def _apply_epo_orthogonalization(
    U_cat: torch.Tensor, V_cat: torch.Tensor, S_cat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    U_cat = U_cat.to(torch.float32)
    V_cat = V_cat.to(torch.float32)
    S_cat = S_cat.to(torch.float32)

    Q_u, R_u = torch.linalg.qr(U_cat, mode="reduced")
    Q_v, R_v = torch.linalg.qr(V_cat, mode="reduced")

    scale_u = torch.diag(R_u)
    scale_v = torch.diag(R_v)

    sign_u = torch.sign(scale_u)
    sign_v = torch.sign(scale_v)

    U_new = Q_u * sign_u.unsqueeze(0)
    V_new = Q_v * sign_v.unsqueeze(0)
    S_new = S_cat * torch.abs(scale_u) * torch.abs(scale_v)
    return U_new, V_new, S_new


def _flattened_top_singular_indices(sv_matrix: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat = sv_matrix.reshape(-1)
    top_k = min(top_k, flat.numel())
    if top_k <= 0:
        empty = torch.empty(0, dtype=torch.long, device=sv_matrix.device)
        return empty, empty
    top_indices = torch.argsort(flat, descending=True, stable=True)[:top_k]
    width = sv_matrix.shape[1]
    return top_indices // width, top_indices % width


def _compute_lora_svd_cache(
    a_list: list[torch.Tensor], b_list: list[torch.Tensor], target_rank: int, device: str
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor, int]:
    svd_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    singular_values: list[torch.Tensor] = []
    max_rank = min(target_rank, min(a_list[0].shape[0], b_list[0].shape[0], a_list[0].shape[1]))

    for A_cpu, B_cpu in zip(a_list, b_list):
        A = A_cpu.to(device=device, dtype=torch.float32)
        B = B_cpu.to(device=device, dtype=torch.float32)
        delta_w = B @ A
        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
        svd_cache.append((U, S, Vh))
        singular_values.append(S[:max_rank])

    return svd_cache, torch.stack(singular_values, dim=0), max_rank


def prism_fusion(
    state_dicts: Iterable[dict[str, torch.Tensor]],
    target_rank: int = 16,
    enable_epo: bool = True,
    device: str | None = "cpu",
) -> dict[str, torch.Tensor]:
    """Merge compatible standard LoRA state dicts using PRISM.

    Candidates are selected in global descending singular-value order. EPO
    orthogonalizes the selected factors without changing that ordering.
    Computation uses float32; output factors keep the input dtype on CPU.
    """
    if isinstance(target_rank, bool) or not isinstance(target_rank, int) or target_rank < 1:
        raise ValueError("target_rank must be a positive integer.")
    state_dicts = list(state_dicts)
    if not state_dicts:
        raise ValueError("No adapter weights provided.")
    keys = _validate_state_dicts(state_dicts)
    lora_pairs = _collect_lora_pairs(keys)
    if not lora_pairs:
        raise ValueError("No LoRA A/B weight pairs found.")
    for pair in lora_pairs.values():
        a, b = state_dicts[0][pair["a_key"]], state_dicts[0][pair["b_key"]]
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0] or min(*a.shape, *b.shape) < 1:
            raise ValueError("LoRA factors must be nonempty compatible 2D matrices.")
        if any(not state[key].is_floating_point() for state in state_dicts for key in pair.values()):
            raise ValueError("LoRA factors must be floating-point tensors.")
        candidates = len(state_dicts) * min(target_rank, a.shape[0], b.shape[0], a.shape[1])
        if enable_epo and min(target_rank, candidates) > min(a.shape[1], b.shape[0]):
            raise ValueError("EPO selected rank exceeds a layer dimension; lower target_rank or disable EPO.")
    merged = {key: _average_non_lora(state_dicts, key) for key in keys if not _is_lora_key(key)}
    device = _resolve_device(device)

    for prefix, pair in lora_pairs.items():
        a_key = pair["a_key"]
        b_key = pair["b_key"]
        a_list = [state[a_key] for state in state_dicts]
        b_list = [state[b_key] for state in state_dicts]
        svd_cache, singular_matrix, _ = _compute_lora_svd_cache(a_list, b_list, target_rank, device)
        top_tasks, top_ranks = _flattened_top_singular_indices(singular_matrix, target_rank)

        u_list = []
        s_list = []
        v_list = []
        for task_idx, rank_idx in zip(top_tasks.tolist(), top_ranks.tolist()):
            U, S, Vh = svd_cache[task_idx]
            u_vec = U[:, rank_idx : rank_idx + 1]
            s_val = S[rank_idx : rank_idx + 1]
            v_vec = Vh[rank_idx : rank_idx + 1, :].T
            u_list.append(u_vec)
            s_list.append(s_val)
            v_list.append(v_vec)

        if u_list:
            U_cat = torch.cat(u_list, dim=1)
            V_cat = torch.cat(v_list, dim=1)
            S_cat = torch.cat(s_list, dim=0)
            if enable_epo:
                U_cat, V_cat, S_cat = _apply_epo_orthogonalization(U_cat, V_cat, S_cat)
            sqrt_s = torch.sqrt(S_cat)
            B_fuse = U_cat * sqrt_s.unsqueeze(0)
            A_fuse = (V_cat * sqrt_s.unsqueeze(0)).T
        else:
            d_in = a_list[0].shape[1]
            d_out = b_list[0].shape[0]
            A_fuse = torch.zeros(0, d_in, device=device)
            B_fuse = torch.zeros(d_out, 0, device=device)

        if A_fuse.shape[0] < target_rank:
            pad_r = target_rank - A_fuse.shape[0]
            d_in = A_fuse.shape[1]
            d_out = B_fuse.shape[0]
            A_pad = torch.zeros(pad_r, d_in, device=device)
            B_pad = torch.zeros(d_out, pad_r, device=device)
            A_fuse = torch.cat([A_fuse, A_pad], dim=0)
            B_fuse = torch.cat([B_fuse, B_pad], dim=1)

        orig_dtype = a_list[0].dtype
        merged[a_key] = A_fuse.cpu().to(orig_dtype)
        merged[b_key] = B_fuse.cpu().to(orig_dtype)
    return merged
