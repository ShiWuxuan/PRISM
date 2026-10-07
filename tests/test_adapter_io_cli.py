import json

import pytest
import torch

from prism.adapter_io import (
    ensure_compatible_configs,
    load_adapter_config,
    load_adapter_state,
    save_adapter_config,
    save_adapter_state,
)
from prism.cli import discover_adapters, fuse_adapters


def make_adapter(path, seed, safe=True):
    generator = torch.Generator().manual_seed(seed)
    state = {
        "layer.lora_A.weight": torch.randn(2, 8, generator=generator),
        "layer.lora_B.weight": torch.randn(8, 2, generator=generator),
    }
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "test/base",
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["q_proj", "v_proj"],
    }
    save_adapter_config(str(path), config)
    save_adapter_state(str(path), state, save_safetensors=safe)
    return state


@pytest.mark.parametrize("safe", [True, False])
def test_roundtrip(tmp_path, safe):
    expected = make_adapter(tmp_path / "adapter", 1, safe=safe)
    restored, fmt = load_adapter_state(str(tmp_path / "adapter"))
    assert fmt == ("safetensors" if safe else "bin")
    for key in expected:
        torch.testing.assert_close(restored[key], expected[key])


def test_cli_discovery_rank_alpha_and_aux_files(tmp_path):
    root = tmp_path / "inputs"
    make_adapter(root / "b", 1)
    make_adapter(root / "a", 2, safe=False)
    (root / "a" / "tokenizer_config.json").write_text('{"test": true}')
    (root / "unrelated").mkdir()
    adapters = discover_adapters(str(root))
    assert [path.rsplit("/", 1)[-1] for path in adapters] == ["a", "b"]
    output = tmp_path / "merged"
    fuse_adapters(adapters, str(output), target_rank=3)
    config = load_adapter_config(str(output))
    assert config["r"] == 3 and config["lora_alpha"] == 6
    assert (output / "adapter_model.safetensors").is_file()
    assert json.loads((output / "tokenizer_config.json").read_text()) == {"test": True}
    assert load_adapter_state(str(output))[0]["layer.lora_A.weight"].shape == (3, 8)


def test_output_protection_and_overwrite(tmp_path):
    inputs = [tmp_path / "a", tmp_path / "b"]
    for index, path in enumerate(inputs):
        make_adapter(path, index)
    output = tmp_path / "output"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        fuse_adapters(list(map(str, inputs)), str(output))
    fuse_adapters(list(map(str, inputs)), str(output), overwrite=True, target_rank=2)
    assert unrelated.read_text() == "keep"
    with pytest.raises(ValueError, match="separate"):
        fuse_adapters(list(map(str, inputs)), str(inputs[0]), overwrite=True)


def test_config_mismatch_does_not_write_output(tmp_path):
    inputs = [tmp_path / "a", tmp_path / "b"]
    for index, path in enumerate(inputs):
        make_adapter(path, index)
    config = load_adapter_config(str(inputs[1]))
    config["lora_alpha"] = 8
    save_adapter_config(str(inputs[1]), config)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="config mismatch"):
        fuse_adapters(list(map(str, inputs)), str(output))
    assert not output.exists()


def test_order_insensitive_config_fields():
    left = {"target_modules": ["q_proj", "v_proj"]}
    right = {"target_modules": ["v_proj", "q_proj"]}
    assert ensure_compatible_configs([left, right]) == left


def test_malformed_config_is_rejected(tmp_path):
    (tmp_path / "adapter_config.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_adapter_config(str(tmp_path))
