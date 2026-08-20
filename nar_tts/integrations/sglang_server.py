"""Launch the SGLang rollout server entirely from a Nar GRPO YAML config."""

import argparse
import json
import os
import sys
from urllib.parse import urlparse

from nar_tts.training.grpo_config import load_grpo_config, validate_grpo_config


def server_command(config: dict) -> list[str]:
    derived = validate_grpo_config(
        config,
        world_size=int(config.get("runtime", {}).get("expected_world_size", 1)),
    )
    if derived["rollout_backend"] != "sglang":
        raise ValueError("the selected config does not use rollout.backend: sglang")
    settings = config["rollout"]["sglang"]
    parsed = urlparse(str(settings["base_url"]))
    host = str(settings.get("host", parsed.hostname or "127.0.0.1"))
    port = int(settings.get("port", parsed.port or 30000))
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(settings.get("model_path", config["model"]["checkpoint"])),
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        str(int(settings.get("tensor_parallel_size", 1))),
        "--enable-lora",
        "--enable-custom-logit-processor",
        "--skip-tokenizer-init",
        "--max-loras-per-batch",
        "2",
        "--max-lora-rank",
        str(int(settings.get("max_lora_rank", config.get("peft", {}).get("rank", 16)))),
        "--lora-target-modules",
        str(settings.get("lora_target_modules", "all")),
    ]
    extra_args = settings.get("server_args", [])
    if not isinstance(extra_args, list) or not all(
        isinstance(value, str) for value in extra_args
    ):
        raise TypeError("rollout.sglang.server_args must be a list of strings")
    return [*command, *extra_args]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="validate and print the argument vector instead of starting the server",
    )
    args = parser.parse_args(argv)
    command = server_command(load_grpo_config(args.config))
    if args.print_command:
        print(json.dumps(command, indent=2))
        return
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
