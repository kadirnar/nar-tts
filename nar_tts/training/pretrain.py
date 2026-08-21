"""Model-neutral speech-token pretraining for Nar TTS."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nar_tts.core.config import (
    apply_user_token_ids,
    configure_reporting,
    load_yaml_config,
    user_token_ids,
)
from nar_tts.core.data import GradualRatioDataset, make_collator
from nar_tts.core.tokens import TokenLayout
from nar_tts.core.trainer import RatioTrainer

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "train" / "pretrain.yaml"
)


def _load_config(path=DEFAULT_CONFIG):
    return load_yaml_config(path, "pretraining")


def _dtype(name):
    value = getattr(torch, str(name), None)
    if not isinstance(value, torch.dtype):
        raise TypeError(f"unknown model dtype: {name!r}")
    return value


def _load_dataset(section: dict, *, shuffle: bool, seed: int):
    from datasets import load_dataset

    if not isinstance(section, dict) or not section.get("path"):
        raise ValueError("dataset.text and dataset.speech need a path")
    kwargs = {"split": section.get("split", "train")}
    for name in ("data_files", "revision", "token"):
        if section.get(name) is not None:
            kwargs[name] = section[name]
    dataset = load_dataset(section["path"], section.get("name"), **kwargs)
    if "input_ids" not in dataset.column_names:
        raise ValueError("pretraining datasets must contain an input_ids column")
    return dataset.shuffle(seed=seed) if shuffle else dataset


def _mix(config: dict) -> tuple[int, int]:
    mix = config.get("mix", {})
    initial = int(mix.get("initial_text_batches_per_speech", 1))
    final = int(mix.get("final_text_batches_per_speech", 0))
    if initial < 1 or final < 0 or final > initial:
        raise ValueError(
            "mix must satisfy initial_text_batches_per_speech >= 1 and "
            "0 <= final <= initial"
        )
    return initial, final


def _model_and_tokenizer(config: dict, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    checkpoint = model_config.get("checkpoint")
    if not checkpoint:
        raise ValueError("model.checkpoint must be set")
    tokenizer_name = model_config.get("tokenizer") or checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_config.get("tokenizer_revision"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    text_eos_token_id, pad_token_id = apply_user_token_ids(tokenizer, config)
    layout = TokenLayout.from_tokenizer(tokenizer)

    if model_config.get("use_liger_model", True):
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = AutoModelForCausalLM
    kwargs = {
        "dtype": _dtype(model_config.get("dtype", "bfloat16")),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
    }
    for name in ("revision", "attn_implementation"):
        if model_config.get(name) is not None:
            kwargs[name] = model_config[name]
    model = model_class.from_pretrained(checkpoint, **kwargs)

    if "<custom_token_0>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(layout.added_token_strings())
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = text_eos_token_id
    model.config.pad_token_id = pad_token_id
    return model.to(device), tokenizer, layout


def train(config: dict):
    user_token_ids(config)

    from accelerate import Accelerator
    from transformers import TrainingArguments, set_seed

    training = config.get("training", {})
    dataset_config = config.get("dataset", {})
    runtime = config.get("runtime", {})
    initial_ratio, final_ratio = _mix(config)
    set_seed(int(training.get("seed", 42)))
    accelerator = Accelerator()
    model, tokenizer, _ = _model_and_tokenizer(config, accelerator.device)

    shuffle = bool(dataset_config.get("shuffle", True))
    seed = int(dataset_config.get("seed", 42))
    text_dataset = _load_dataset(dataset_config.get("text"), shuffle=shuffle, seed=seed)
    speech_dataset = _load_dataset(
        dataset_config.get("speech"), shuffle=shuffle, seed=seed + 1
    )
    batch_size = int(training.get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("training.batch_size must be positive")
    global_microbatch = batch_size * accelerator.num_processes
    train_dataset = GradualRatioDataset(
        text_dataset,
        speech_dataset,
        global_microbatch,
        initial_ratio=initial_ratio,
        final_ratio=final_ratio,
        total_steps=0,
    )
    if not len(train_dataset):
        raise ValueError("pretraining datasets are too small for one global batch")

    logging = config.get("logging", {})
    workers = int(runtime.get("dataloader_workers", 0))
    if workers < 0:
        raise ValueError("runtime.dataloader_workers cannot be negative")
    args = TrainingArguments(
        output_dir=training.get("output_dir", "checkpoints/pretrain"),
        num_train_epochs=float(training.get("epochs", 1)),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training.get("learning_rate", 5e-5)),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        save_steps=int(training.get("save_steps", 1000)),
        save_total_limit=int(training.get("save_total_limit", 3)),
        logging_steps=int(training.get("logging_steps", 1)),
        bf16=bool(training.get("bf16", True)),
        fp16=bool(training.get("fp16", False)),
        tf32=training.get("tf32", True),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", False)),
        report_to=configure_reporting(logging),
        run_name=logging.get("run_name"),
        remove_unused_columns=True,
        dataloader_num_workers=workers,
        dataloader_pin_memory=bool(runtime.get("pin_memory", True)),
        seed=int(training.get("seed", 42)),
        average_tokens_across_devices=False,
    )
    _, pad_token_id = apply_user_token_ids(tokenizer, config)
    trainer = RatioTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=make_collator(pad_token_id),
        processing_class=tokenizer,
        initial_ratio=initial_ratio,
        final_ratio=final_ratio,
    )
    if accelerator.is_local_main_process:
        print(
            f"Pretraining {len(train_dataset)} rows: text ratio "
            f"{initial_ratio}:1 -> {final_ratio}:1"
        )
    trainer.train(resume_from_checkpoint=training.get("resume_from_checkpoint") or None)
    trainer.save_model(args.output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    train(_load_config(args.config))


if __name__ == "__main__":
    main()
