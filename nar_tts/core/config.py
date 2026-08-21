"""Small shared helpers for model-neutral Nar configuration files."""

from __future__ import annotations

import os
from pathlib import Path


def load_yaml_config(path, label: str) -> dict:
    import yaml

    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"{label} config must contain a YAML mapping")
    config["_config_path"] = os.fspath(path)
    return config


def user_token_ids(config: dict) -> tuple[int, int]:
    """Return explicit text-EOS and padding IDs from a config."""
    tokens = config.get("tokens")
    if not isinstance(tokens, dict):
        raise TypeError("config must contain a tokens section")
    values = []
    for name in ("text_eos_token_id", "pad_token_id"):
        value = tokens.get(name)
        if value is None or isinstance(value, bool):
            raise ValueError(f"tokens.{name} must be written by the user")
        try:
            value = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"tokens.{name} must be a non-negative integer") from error
        if value < 0:
            raise ValueError(f"tokens.{name} must be a non-negative integer")
        values.append(value)
    return values[0], values[1]


def apply_user_token_ids(tokenizer, config: dict) -> tuple[int, int]:
    """Apply and validate the user-selected IDs on any HF tokenizer."""
    from nar_tts.core.tokens import EOS_SPEECH

    text_eos_token_id, pad_token_id = user_token_ids(config)
    vocabulary_size = len(tokenizer)
    for name, value in (
        ("text_eos_token_id", text_eos_token_id),
        ("pad_token_id", pad_token_id),
    ):
        if value >= vocabulary_size:
            raise ValueError(
                f"tokens.{name}={value} is outside tokenizer size {vocabulary_size}"
            )

    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    saved_text_eos = init_kwargs.get("nar_text_eos_token_id")
    custom_zero = tokenizer.get_vocab().get("<custom_token_0>")
    speech_eos = None if custom_zero is None else int(custom_zero) + EOS_SPEECH
    # A legacy GRPO tokenizer replaces eos_token_id with EOS_SPEECH. In that
    # case the explicit config is the only reliable source for the text EOS.
    current_text_eos = tokenizer.eos_token_id
    if saved_text_eos is not None:
        current_text_eos = int(saved_text_eos)
    elif speech_eos is not None and current_text_eos == speech_eos:
        current_text_eos = None
    if current_text_eos is not None and text_eos_token_id != current_text_eos:
        raise ValueError(
            "tokens.text_eos_token_id does not match the selected tokenizer"
        )
    if tokenizer.pad_token_id not in (None, pad_token_id):
        raise ValueError("tokens.pad_token_id does not match the selected tokenizer")

    init_kwargs["nar_text_eos_token_id"] = text_eos_token_id
    tokenizer.init_kwargs = init_kwargs
    tokenizer.eos_token_id = text_eos_token_id
    tokenizer.pad_token_id = pad_token_id
    if tokenizer.eos_token_id != text_eos_token_id:
        raise ValueError("tokenizer rejected tokens.text_eos_token_id")
    if tokenizer.pad_token_id != pad_token_id:
        raise ValueError("tokenizer rejected tokens.pad_token_id")
    return text_eos_token_id, pad_token_id


def configure_reporting(config: dict | None):
    """Configure optional W&B environment fields and return ``report_to``."""
    config = dict(config or {})
    if not config.get("enabled", True):
        return "none"
    report_to = config.get("report_to", "wandb")
    targets = [report_to] if isinstance(report_to, str) else list(report_to or ())
    if "wandb" in targets:
        environment = {
            "WANDB_PROJECT": config.get("project"),
            "WANDB_ENTITY": config.get("entity"),
            "WANDB_RUN_GROUP": config.get("group"),
            "WANDB_MODE": config.get("mode"),
        }
        tags = config.get("tags")
        if tags:
            environment["WANDB_TAGS"] = ",".join(str(tag) for tag in tags)
        for name, value in environment.items():
            if value not in (None, ""):
                os.environ.setdefault(name, str(value))
    if not targets:
        return "none"
    return targets[0] if len(targets) == 1 else targets
