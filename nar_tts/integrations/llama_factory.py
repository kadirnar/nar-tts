"""Validated launcher for LLaMA-Factory post-pretraining stages."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from nar_tts.training.grpo_config import available_cpu_count, dataloader_worker_count


def load_llama_factory_config(path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("LLaMA-Factory config must contain a YAML mapping")
    return validate_llama_factory_config(config)


def validate_llama_factory_config(config: dict) -> dict:
    """Reject incomplete configs and stages Nar cannot truthfully provide."""
    for name in ("model_name_or_path", "stage", "dataset", "output_dir"):
        if config.get(name) in (None, ""):
            raise ValueError(f"LLaMA-Factory config must set {name}")
    if str(config["stage"]).casefold() == "grpo":
        raise ValueError(
            "LLaMA-Factory does not provide native GRPO; use Nar's GRPO trainer "
            "with the Transformers, vLLM, or SGLang rollout backend"
        )
    return config


def resolve_runtime_values(config: dict) -> dict:
    resolved = dict(config)
    if resolved.get("preprocessing_num_workers") == "auto":
        resolved["preprocessing_num_workers"] = available_cpu_count()
    if resolved.get("dataloader_num_workers") == "auto":
        resolved["dataloader_num_workers"] = dataloader_worker_count(
            "auto", world_size=int(os.environ.get("WORLD_SIZE", "1"))
        )
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print resolved automatic settings without training",
    )
    args = parser.parse_args(argv)
    config = resolve_runtime_values(load_llama_factory_config(args.config))
    executable = shutil.which("llamafactory-cli")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "config": str(Path(args.config).expanduser().resolve()),
                    "stage": config["stage"],
                    "preprocessing_num_workers": config.get(
                        "preprocessing_num_workers"
                    ),
                    "dataloader_num_workers": config.get(
                        "dataloader_num_workers"
                    ),
                    "llamafactory_cli": executable,
                },
                indent=2,
            )
        )
        return
    if executable is None:
        raise RuntimeError(
            "llamafactory-cli is not installed. Create the separate "
            "LLaMA-Factory environment described in docs/grpo.md."
        )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", encoding="utf-8"
    ) as resolved_file:
        yaml.safe_dump(config, resolved_file, sort_keys=False)
        resolved_file.flush()
        subprocess.run(
            [executable, "train", resolved_file.name],
            check=True,
        )


if __name__ == "__main__":
    main()
