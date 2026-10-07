import json
import subprocess
import sys

import pytest

from prism.train import BACKEND_COMMIT, main, require_training_backend

BASE_ARGS = [
    "--model-name-or-path",
    "test/base",
    "--template",
    "default",
    "--dataset",
    "task",
    "--dataset-dir",
    "data",
    "--output",
    "outputs/task",
]


def test_dry_run_defaults_and_typed_settings(capsys):
    main(BASE_ARGS + ["--dry-run", "--set", "bf16=true", "--set", "max_steps=2", "--set", "seed=7"])
    config = json.loads(capsys.readouterr().out)
    assert config["stage"] == "sft" and config["finetuning_type"] == "lora"
    assert (config["lora_rank"], config["lora_alpha"], config["lora_dropout"]) == (16, 32, 0.05)
    assert config["bf16"] is True and config["max_steps"] == 2 and config["seed"] == 7


@pytest.mark.parametrize("setting", ["stage=dpo", "finetuning_type=full", "adapter_name_or_path=old-task"])
def test_rejects_mode_and_previous_adapter_overrides(setting):
    with pytest.raises(SystemExit) as error:
        main(BASE_ARGS + ["--dry-run", "--set", setting])
    assert error.value.code == 2


def test_one_dataset_per_task():
    args = BASE_ARGS.copy()
    args[args.index("--dataset") + 1] = "task_a,task_b"
    with pytest.raises(SystemExit):
        main(args + ["--dry-run"])


def test_core_import_does_not_load_training_backend():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, prism; assert not {'llamafactory', 'transformers', 'peft'} & set(sys.modules)",
        ],
        check=True,
    )


def test_repeated_general_setting_uses_last_value(capsys):
    main(BASE_ARGS + ["--dry-run", "--set", "max_steps=1", "--set", "max_steps=2"])
    assert json.loads(capsys.readouterr().out)["max_steps"] == 2


@pytest.mark.parametrize("commit", [None, "different-commit", BACKEND_COMMIT])
def test_backend_commit_provenance(monkeypatch, commit):
    import prism.train as training

    class Distribution:
        def read_text(self, name):
            return json.dumps({"vcs_info": {"commit_id": commit}})

    monkeypatch.setattr(training.importlib.metadata, "distribution", lambda name: Distribution())
    if commit == BACKEND_COMMIT:
        require_training_backend()
    else:
        with pytest.raises(RuntimeError, match="pinned"):
            require_training_backend()
