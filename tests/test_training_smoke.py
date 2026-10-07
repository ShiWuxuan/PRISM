"""An offline CPU training-to-fusion integration test; no external weights."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_two_task_training_and_fusion(tmp_path):
    import torch
    from peft import PeftModel
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from prism.adapter_io import load_adapter_config, load_adapter_state
    from prism.cli import fuse_adapters

    model_dir = tmp_path / "base"
    torch.manual_seed(5)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    LlamaForCausalLM(config).save_pretrained(model_dir)
    vocabulary = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "hello": 4,
        "red": 5,
        "blue": 6,
        "User": 7,
        "Assistant": 8,
        ":": 9,
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>"
    )
    tokenizer.save_pretrained(model_dir)
    adapters = []
    for index, answer in enumerate(["red", "blue"]):
        data_dir = tmp_path / f"data_{index}"
        data_dir.mkdir()
        examples = [{"conversations": [{"from": "human", "value": "hello"}, {"from": "gpt", "value": answer}]}] * 4
        (data_dir / "train.json").write_text(json.dumps(examples))
        (data_dir / "dataset_info.json").write_text(
            json.dumps(
                {
                    "task": {
                        "file_name": "train.json",
                        "formatting": "sharegpt",
                        "columns": {"messages": "conversations"},
                    },
                }
            )
        )
        output = tmp_path / f"task_{index}"
        command = [
            sys.executable,
            "-c",
            "from prism.train import main; main()",
            "--model-name-or-path",
            str(model_dir),
            "--template",
            "default",
            "--dataset",
            "task",
            "--dataset-dir",
            str(data_dir),
            "--output",
            str(output),
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
            "--lora-dropout",
            "0",
        ]
        for setting in [
            "use_cpu=true",
            "bf16=false",
            "fp16=false",
            "max_steps=2",
            "cutoff_len=64",
            "per_device_train_batch_size=1",
            "gradient_accumulation_steps=1",
            "learning_rate=0.01",
            "save_strategy=no",
            "plot_loss=false",
            "preprocessing_num_workers=1",
            "dataloader_num_workers=0",
            "disable_tqdm=true",
            "logging_steps=1",
            f"seed={42 + index}",
        ]:
            command.extend(["--set", setting])
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            CUDA_VISIBLE_DEVICES="",
        )
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        assert load_adapter_config(str(output))["r"] == 2
        assert (output / "adapter_model.safetensors").is_file()
        adapters.append(str(output))
    fused_dir = tmp_path / "merged"
    fuse_adapters(adapters, str(fused_dir), target_rank=3)
    fused_config = load_adapter_config(str(fused_dir))
    assert fused_config["r"] == 3 and fused_config["lora_alpha"] == 6
    assert all(torch.isfinite(tensor).all() for tensor in load_adapter_state(str(fused_dir))[0].values())
    base = AutoModelForCausalLM.from_pretrained(model_dir)
    fused = PeftModel.from_pretrained(base, fused_dir)
    with torch.no_grad():
        logits = fused(input_ids=torch.tensor([[1, 4, 2]])).logits
    assert torch.isfinite(logits).all()
