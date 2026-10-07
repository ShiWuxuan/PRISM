import pytest
import torch

from prism import prism_fusion


def states(seed=17, count=3, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "layer.lora_A.weight": torch.randn(4, 20, generator=generator).to(dtype),
            "layer.lora_B.weight": torch.randn(24, 4, generator=generator).to(dtype),
            "projector.weight": torch.randn(2, 3, generator=generator).to(dtype),
            "counter": torch.tensor(3),
        }
        for _ in range(count)
    ]


@pytest.mark.parametrize("epo", [True, False])
@pytest.mark.parametrize("rank", [2, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rank_dtype_finiteness_and_shared_parameters(epo, rank, dtype):
    inputs = states(dtype=dtype)
    result = prism_fusion(inputs, target_rank=rank, enable_epo=epo)
    assert result["layer.lora_A.weight"].shape == (rank, 20)
    assert result["layer.lora_B.weight"].shape == (24, rank)
    for key, value in result.items():
        assert value.dtype == inputs[0][key].dtype
        assert torch.isfinite(value).all()
    expected_shared = torch.stack([s["projector.weight"].float() for s in inputs]).mean(0).to(dtype)
    torch.testing.assert_close(result["projector.weight"], expected_shared)
    assert result["counter"].item() == 3
    if rank == 16:
        assert torch.count_nonzero(result["layer.lora_A.weight"][12:]) == 0
        assert torch.count_nonzero(result["layer.lora_B.weight"][:, 12:]) == 0


def test_selects_largest_direction_across_tasks():
    first = {
        "layer.lora_A.weight": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        "layer.lora_B.weight": torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
    }
    second = {
        "layer.lora_A.weight": torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        "layer.lora_B.weight": torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 0.5]]),
    }
    merged = prism_fusion([first, second], target_rank=1)
    delta = merged["layer.lora_B.weight"] @ merged["layer.lora_A.weight"]
    expected = torch.zeros(4, 4)
    expected[2, 2] = 3
    torch.testing.assert_close(delta, expected)


def test_zero_singular_values_remain_finite():
    inputs = states(count=2)
    for state in inputs:
        state["layer.lora_A.weight"].zero_()
        state["layer.lora_B.weight"].zero_()
    result = prism_fusion(inputs, target_rank=16)
    assert torch.count_nonzero(result["layer.lora_A.weight"]) == 0
    assert torch.count_nonzero(result["layer.lora_B.weight"]) == 0


@pytest.mark.parametrize("rank", [0, -1, True, 1.5])
def test_invalid_rank(rank):
    with pytest.raises(ValueError, match="positive integer"):
        prism_fusion(states(), target_rank=rank)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="No adapter"):
        prism_fusion([])
    with pytest.raises(ValueError, match="No LoRA"):
        prism_fusion([{"shared": torch.ones(2)}])
    inputs = states(count=2)
    inputs[1].pop("counter")
    with pytest.raises(ValueError, match="keys mismatch"):
        prism_fusion(inputs)
    inputs = states(count=2)
    inputs[1]["layer.lora_A.weight"] = torch.ones(3, 20)
    with pytest.raises(ValueError, match="shape mismatch"):
        prism_fusion(inputs)
    inputs = states(count=2)
    inputs[1]["counter"] = torch.tensor(4)
    with pytest.raises(ValueError, match="Non-floating"):
        prism_fusion(inputs)
    broken = states(count=1)[0]
    broken.pop("layer.lora_B.weight")
    with pytest.raises(ValueError, match="Missing LoRA"):
        prism_fusion([broken])


def test_epo_dimension_error_is_explicit():
    inputs = [{"l.lora_A.weight": torch.ones(2, 2), "l.lora_B.weight": torch.ones(2, 2)}] * 2
    with pytest.raises(ValueError, match="layer dimension"):
        prism_fusion(inputs, target_rank=4)
    result = prism_fusion(inputs, target_rank=4, enable_epo=False)
    assert result["l.lora_A.weight"].shape == (4, 2)
