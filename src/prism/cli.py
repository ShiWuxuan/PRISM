# Copyright 2025 the LlamaFactory team.
# Copyright 2026 Wuxuan Shi.
# SPDX-License-Identifier: Apache-2.0
# Modified: extracted a PRISM-only CLI, independent of the training framework.

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from .adapter_io import (
    ADAPTER_CONFIG_NAME,
    ensure_compatible_configs,
    find_adapter_weights,
    load_adapter_config,
    load_adapter_state,
    save_adapter_config,
    save_adapter_state,
    update_lora_config_for_rank,
)
from .merge import prism_fusion

AUX_FILES = (
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def discover_adapters(input_root: str) -> list[str]:
    adapters = []
    for candidate in sorted(Path(input_root).iterdir()):
        if not candidate.is_dir() or not (candidate / ADAPTER_CONFIG_NAME).is_file():
            continue
        try:
            find_adapter_weights(str(candidate))
        except FileNotFoundError:
            continue
        adapters.append(str(candidate))
    return adapters


def fuse_adapters(
    inputs: list[str],
    output: str,
    *,
    target_rank: int = 16,
    enable_epo: bool = True,
    device: str = "cpu",
    overwrite: bool = False,
) -> None:
    """Validate adapters, merge them, and save a standard PEFT adapter."""
    inputs = list(dict.fromkeys(str(Path(path).resolve()) for path in inputs))
    if not inputs:
        raise ValueError("No adapter directories provided.")
    destination = Path(output).resolve()
    for path in inputs:
        source = Path(path)
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("Output must be separate from input adapter directories.")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("Output must be a directory.")
        if any(destination.iterdir()) and not overwrite:
            raise ValueError("Output directory is not empty. Use --overwrite to replace generated files.")

    configs = [load_adapter_config(path) for path in inputs]
    for config in configs:
        if config.get("peft_type") != "LORA":
            raise ValueError("PRISM expects standard PEFT LoRA adapters.")
        if any(config.get(key) for key in ("use_dora", "use_rslora", "rank_pattern", "alpha_pattern")):
            raise ValueError("PRISM expects standard LoRA without DoRA, rank stabilization, or per-layer patterns.")
    config = ensure_compatible_configs(configs)
    states = [load_adapter_state(path)[0] for path in inputs]
    merged = prism_fusion(states, target_rank=target_rank, enable_epo=enable_epo, device=device)
    config = update_lora_config_for_rank(config, merged)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prism-", dir=destination.parent) as temporary:
        save_adapter_config(temporary, config)
        save_adapter_state(temporary, merged, save_safetensors=True)
        for filename in AUX_FILES:
            source = Path(inputs[0]) / filename
            if source.is_file():
                shutil.copyfile(source, Path(temporary) / filename)
        destination.mkdir(exist_ok=True)
        for generated in Path(temporary).iterdir():
            os.replace(generated, destination / generated.name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge compatible LoRA adapters using PRISM.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inputs", nargs="+", help="Ordered adapter directories.")
    inputs.add_argument("--input-root", help="Discover immediate adapter subdirectories in name order.")
    parser.add_argument("--output", required=True, help="Directory for the merged adapter.")
    parser.add_argument("--target-rank", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or auto.")
    parser.add_argument("--disable-epo", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in an existing output.")
    args = parser.parse_args(argv)
    try:
        adapter_dirs = args.inputs if args.inputs is not None else discover_adapters(args.input_root)
        fuse_adapters(
            adapter_dirs,
            args.output,
            target_rank=args.target_rank,
            enable_epo=not args.disable_epo,
            device=args.device,
            overwrite=args.overwrite,
        )
    except (ValueError, FileNotFoundError, NotADirectoryError) as error:
        parser.error(str(error))
    print(f"Merged {len(set(adapter_dirs))} adapters into {args.output} using PRISM.")
