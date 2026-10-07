# PRISM

Code for **PRISM: Spectral Pruning and Reconstruction for Parameter-Efficient
Model Merging of MLLMs**.

PRISM merges independently trained, compatible LoRA adapters into one adapter.
This repository provides the PRISM fusion implementation and a single-task SFT
LoRA training entry point backed by a pinned version of
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

Fusion can run without the training framework or the base model weights.

## Installation

Python 3.11 or newer is required. Use a separate environment for this project.

```bash
git clone https://github.com/ShiWuxuan/PRISM.git
cd PRISM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For training, install the optional backend:

```bash
python -m pip install -e '.[train]'
```

The backend is pinned to upstream commit
`d7d734d54c7c9a1e3c9660e9689e89a4abb65a9d`. Its declared dependency ranges
apply to training, including Transformers up to 4.57.1 and PEFT up to 0.17.1.
The training entry point checks the installed backend's commit provenance.

The offline training-to-fusion smoke test was verified with Python 3.11,
PyTorch 2.6.0, Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.11.0,
Datasets 4.0.0 and safetensors 0.7.0.

## Merge adapters

Each input directory must contain `adapter_config.json` and either
`adapter_model.safetensors` or `adapter_model.bin`:

```text
adapters/
  task_a/
    adapter_config.json
    adapter_model.safetensors
  task_b/
    adapter_config.json
    adapter_model.safetensors
```

```bash
prism-merge \
  --inputs adapters/task_a adapters/task_b \
  --output outputs/prism \
  --target-rank 16 \
  --device cpu
```

`python scripts/fuse_lora.py` accepts the same arguments. Alternatively, pass
`--input-root adapters` to discover immediate subdirectories in name order.
Explicit `--inputs` makes the input order reproducible, including singular-value ties.

The output contains `adapter_config.json` and `adapter_model.safetensors`.
Tokenizer and processor files present in the first input are also copied.
Load the output with the same base model using PEFT:

```python
from peft import PeftModel

merged_model = PeftModel.from_pretrained(base_model, "outputs/prism")
```

Supported inputs are standard PEFT LoRA adapters from the **same base model**,
with matching configurations, weight keys and input shapes. Configurations
must match, including `base_model_name_or_path`; use the same model identifier
when training each task. DoRA, rank-stabilized LoRA and per-layer rank/alpha
patterns are outside this release's adapter interface.

The default output rank is 16; EPO is enabled and computation runs on CPU.
Use `--disable-epo` for the EPO ablation or `--device cuda:0` for GPU computation.
The factors are computed in float32 and saved on CPU in their original dtype.
If output rank changes, `lora_alpha` is updated to preserve `lora_alpha / r`.
Non-LoRA floating parameters are averaged; non-floating parameters must agree.

When EPO is enabled, the number of selected directions cannot exceed a layer's
input or output dimension. A large requested output rank can still be padded
when fewer candidate directions are available.

The output must be separate from the input directories. Existing nonempty
outputs require `--overwrite`, which replaces generated files while retaining
unrelated files. An invalid merge is rejected before output files are written.

## Python API

```python
from prism import prism_fusion
from prism.adapter_io import load_adapter_state

states = [load_adapter_state(path)[0] for path in input_adapter_directories]
merged_state = prism_fusion(states, target_rank=16, enable_epo=True, device="cpu")
```

For each LoRA layer, PRISM computes the SVD of each task's `B @ A`, selects
directions in global descending singular-value order, optionally applies EPO,
and reconstructs the fused A/B factors. The Python API merges tensors;
the CLI additionally checks configurations and updates the saved adapter metadata.

## Dataset

The data used in our experiments comes from
[MLLM-CL on Hugging Face](https://huggingface.co/datasets/MLLM-CL/MLLM-CL).
Download the relevant task data from that repository and prepare it in the
format described below for single-task LoRA training.

## Train one task-specific LoRA

Prepare a data directory containing `dataset_info.json`, your training JSON,
and any referenced images. For example, register a ShareGPT dataset:

```json
{
  "OCR_data": {
    "file_name": "train_sharegpt.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "images"},
    "tags": {
      "role_tag": "from", "content_tag": "value",
      "user_tag": "human", "assistant_tag": "gpt"
    }
  }
}
```

A record in `train_sharegpt.json` can have this structure:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "<image>Read the text in this image."},
      {"from": "gpt", "value": "Example"}
    ],
    "images": ["images/example.png"]
  }
]
```

Create the referenced image yourself. Image paths are relative to the data
directory. Dataset and training configuration files are supplied locally.

An example single-task Qwen2.5-VL-3B invocation is:

```bash
CUDA_VISIBLE_DEVICES=0 prism-train \
  --model-name-or-path Qwen/Qwen2.5-VL-3B-Instruct \
  --template qwen2_vl \
  --dataset OCR_data \
  --dataset-dir ./data/ocr \
  --output ./outputs/task_ocr \
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
  --set additional_target=merger \
  --set freeze_vision_tower=true \
  --set image_max_pixels=589824 \
  --set cutoff_len=2048 \
  --set per_device_train_batch_size=8 \
  --set gradient_accumulation_steps=2 \
  --set learning_rate=0.00001 \
  --set num_train_epochs=3 \
  --set bf16=true
```

`python scripts/train_lora.py` accepts the same arguments. Each invocation
trains one registered dataset from the specified base model using SFT and
LoRA. Run it separately for each task and then merge the resulting adapters.
The example illustrates the workflow; it does not assert a complete reproduction
of the paper's experiments.

LoRA defaults are rank 16, alpha 32, dropout 0.05 and targets
`q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj`.
General training options use repeated `--set KEY=VALUE` arguments. JSON booleans,
numbers, lists and `null` are parsed; other values remain strings. SFT/LoRA mode
and explicit CLI arguments cannot be overridden with `--set`.
`--set resume_from_checkpoint=PATH` resumes a checkpoint for the same task.

Add `--dry-run` to print the resolved arguments without loading a model or
requiring the optional training backend. Other model families can use their
corresponding LLaMA-Factory template and model-specific options.
For distributed training, invoke the training script with `torchrun`.

## Tests

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m build
```

The core tests use synthetic tensors and temporary adapter files. To exercise
the training-to-fusion workflow with a locally generated tiny model and tokenizer:

```bash
python -m pip install -e '.[train,dev]'
python -m pytest -m integration
```

The integration test runs two steps for each of two synthetic tasks on CPU;
it downloads no model or dataset. Its temporary weights are not release assets.

## Citation and license

If you use PRISM in your research, please cite our paper:

```bibtex
@article{shi2026prism,
  title={PRISM: Spectral Pruning and Reconstruction for Parameter-Efficient Model Merging of MLLMs},
  author={Shi, Wuxuan and Chen, Haotian and Li, He Lia and Yang, Qiang and Ye, Mang},
  journal={Advances in Neural Information Processing Systems},
  year={2026}
}
```

Software citation metadata is provided in `CITATION.cff`.
This code is distributed under Apache-2.0; see `LICENSE` and `NOTICE` for
copyright and upstream attribution. The optional training backend is maintained
by the LLaMA-Factory contributors.
