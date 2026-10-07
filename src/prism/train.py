# Copyright 2026 Wuxuan Shi.
# SPDX-License-Identifier: Apache-2.0

"""Single-task SFT LoRA entry point using a pinned LLaMA-Factory backend."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

BACKEND_COMMIT = "d7d734d54c7c9a1e3c9660e9689e89a4abb65a9d"
DEFAULT_TARGETS = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"


def build_training_args(args: argparse.Namespace) -> dict:
    if not args.dataset or "," in args.dataset:
        raise ValueError("Supply exactly one registered dataset per task.")
    if args.lora_rank < 1 or args.lora_alpha < 1 or not 0 <= args.lora_dropout < 1:
        raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0, 1).")
    config = {
        "model_name_or_path": args.model_name_or_path,
        "template": args.template,
        "dataset": args.dataset,
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target": args.lora_target,
        "report_to": "none",
        "seed": 42,
    }
    protected = set(config) - {"report_to", "seed"}
    for assignment in args.settings:
        key, separator, value = assignment.partition("=")
        if not separator or not key:
            raise ValueError("--set expects KEY=VALUE.")
        if key in protected:
            raise ValueError(f"{key} is fixed or has a dedicated CLI argument.")
        if key == "adapter_name_or_path":
            raise ValueError("Each task must start from its specified base model, not a previous task adapter.")
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value
    return config


def require_training_backend() -> None:
    try:
        distribution = importlib.metadata.distribution("llamafactory")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Install the training extra first: pip install -e '.[train]'") from error
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != BACKEND_COMMIT:
        raise RuntimeError(
            "Training requires the pinned LLaMA-Factory commit. Install with pip install -e '.[train]'."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train one task-specific LoRA using pinned LLaMA-Factory.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--dataset", required=True, help="One name registered in dataset_info.json.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target", default=DEFAULT_TARGETS)
    parser.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional LLaMA-Factory argument; JSON values are parsed, other values are strings.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved arguments without training.")
    args = parser.parse_args(argv)
    try:
        config = build_training_args(args)
        if args.dry_run:
            print(json.dumps(config, indent=2))
            return
        if not Path(args.dataset_dir, "dataset_info.json").is_file():
            raise ValueError("dataset-dir must contain dataset_info.json.")
        require_training_backend()
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    # Import the optional backend only after argument validation.
    from llamafactory.train.tuner import run_exp

    run_exp(config)
