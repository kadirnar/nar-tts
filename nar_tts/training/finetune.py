"""Config-driven supervised post-pretraining for Nar TTS."""

import argparse
import os
from pathlib import Path

import torch
import yaml

from nar_tts.core.data import make_collator
from nar_tts.training.grpo_config import dataloader_worker_count

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "finetune.yaml"


def _load_config(path):
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("fine-tuning config must contain a YAML mapping")
    return config


def _dtype(name):
    value = getattr(torch, str(name), None)
    if not isinstance(value, torch.dtype):
        raise TypeError(f"unknown model dtype: {name!r}")
    return value


def _load_dataset(config):
    from datasets import IterableDataset, load_dataset

    kwargs = {
        "split": config.get("split", "train"),
        "streaming": bool(config.get("streaming", False)),
    }
    for name in ("data_files", "revision", "token"):
        if config.get(name) is not None:
            kwargs[name] = config[name]
    dataset = load_dataset(config["path"], config.get("name"), **kwargs)
    if config.get("shuffle", True):
        if isinstance(dataset, IterableDataset):
            dataset = dataset.shuffle(
                seed=int(config.get("seed", 42)),
                buffer_size=int(config.get("shuffle_buffer", 10_000)),
            )
        else:
            dataset = dataset.shuffle(seed=int(config.get("seed", 42)))
    max_samples = config.get("max_samples")
    if max_samples is not None:
        if isinstance(dataset, IterableDataset):
            dataset = dataset.take(int(max_samples))
        else:
            dataset = dataset.select(range(min(int(max_samples), len(dataset))))
    return dataset


def _unsloth_target_modules(value):
    if value != "all-linear":
        return value
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def _padding_token_id(tokenizer):
    """Preserve a checkpoint's explicit pad token across post-training stages."""
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define a pad_token_id or eos_token_id")
    return tokenizer.eos_token_id


def _load_model_and_tokenizer(model_config, training, accelerator):
    loader = model_config.get("loader", "transformers")
    if loader not in {"transformers", "unsloth"}:
        raise ValueError("model.loader must be transformers or unsloth")
    if loader == "unsloth":
        try:
            from unsloth import FastLanguageModel
        except ImportError as error:
            raise ImportError(
                "model.loader is unsloth, but Unsloth is not installed; use the "
                "separate environment documented in docs/grpo.md"
            ) from error
        settings = model_config.get("unsloth", {})
        peft = model_config.get("peft", {})
        load_in_4bit = bool(settings.get("load_in_4bit", False))
        load_in_8bit = bool(settings.get("load_in_8bit", False))
        kwargs = {
            "model_name": model_config["checkpoint"],
            "tokenizer_name": model_config.get(
                "tokenizer", model_config["checkpoint"]
            ),
            "max_seq_length": int(settings.get("max_seq_length", 8192)),
            "dtype": _dtype(model_config.get("dtype", "bfloat16")),
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
            "load_in_16bit": not load_in_4bit and not load_in_8bit,
            "full_finetuning": not peft.get("enabled", True),
            "fast_inference": False,
            "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        }
        if model_config.get("revision") is not None:
            kwargs["revision"] = model_config["revision"]
        tokenizer_revision = model_config.get("tokenizer_revision")
        if tokenizer_revision is not None and tokenizer_revision != model_config.get(
            "revision"
        ):
            raise ValueError(
                "Unsloth derives the tokenizer revision from model.revision and does "
                "not support a separate model.tokenizer_revision"
            )
        if settings.get("device_map") is not None:
            kwargs["device_map"] = settings["device_map"]
        model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
        if peft.get("enabled", True):
            checkpointing = False
            if training.get("gradient_checkpointing", True):
                checkpointing = settings.get(
                    "gradient_checkpointing", "unsloth"
                )
            model = FastLanguageModel.get_peft_model(
                model,
                r=int(peft.get("rank", 16)),
                target_modules=_unsloth_target_modules(
                    peft.get("target_modules", "all-linear")
                ),
                lora_alpha=int(peft.get("alpha", 32)),
                lora_dropout=float(peft.get("dropout", 0.0)),
                bias=peft.get("bias", "none"),
                use_gradient_checkpointing=checkpointing,
                random_state=int(training.get("seed", 42)),
                use_rslora=bool(peft.get("use_rslora", True)),
            )
        FastLanguageModel.for_training(model)
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer"], revision=model_config.get("tokenizer_revision")
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("the base tokenizer must define eos_token_id")
    if model_config.get("use_liger_model", True):
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = AutoModelForCausalLM
    model_kwargs = {"dtype": _dtype(model_config.get("dtype", "bfloat16"))}
    for name in ("revision", "attn_implementation"):
        if model_config.get(name) is not None:
            model_kwargs[name] = model_config[name]
    model = model_class.from_pretrained(
        model_config["checkpoint"], **model_kwargs
    ).to(accelerator.device)
    return model, tokenizer


def train(config):
    model_config = config["model"]
    dataset_config = config["dataset"]
    training = config["training"]
    logging = config.get("logging", {})
    runtime = config.get("runtime", {})
    if (
        dataset_config.get("streaming", False)
        and int(training.get("max_steps", -1)) <= 0
    ):
        raise ValueError("streaming supervised fine-tuning requires training.max_steps")
    if logging.get("project"):
        os.environ.setdefault("WANDB_PROJECT", str(logging["project"]))

    # Unsloth must patch the ML stack before Transformers, TRL, or PEFT is
    # imported. Keep every import of those libraries below this point.
    if model_config.get("loader", "transformers") == "unsloth":
        try:
            import unsloth  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "model.loader is unsloth, but Unsloth is not installed; use the "
                "separate environment documented in docs/grpo.md"
            ) from error

    from accelerate import Accelerator
    from datasets import IterableDataset
    from transformers import TrainingArguments

    from nar_tts.core.trainer import FSDPTrainer

    accelerator = Accelerator()
    model, tokenizer = _load_model_and_tokenizer(
        model_config, training, accelerator
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("the base tokenizer must define eos_token_id")
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError(
            "supervised post-training requires the checkpoint's expanded Nar tokenizer"
        )
    train_dataset = _load_dataset(dataset_config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    workers = dataloader_worker_count(
        runtime.get("dataloader_workers", "auto"), world_size=world_size
    )

    args = TrainingArguments(
        output_dir=training["output_dir"],
        num_train_epochs=float(training.get("epochs", 10)),
        max_steps=int(training.get("max_steps", -1)),
        per_device_train_batch_size=int(training.get("batch_size", 8)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training.get("learning_rate", 1e-5)),
        warmup_steps=float(
            training.get("warmup_steps", training.get("warmup_ratio", 0.03))
        ),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        weight_decay=float(training.get("weight_decay", 0.0)),
        save_strategy=training.get("save_strategy", "steps"),
        save_steps=int(training.get("save_steps", 100)),
        save_total_limit=int(training.get("save_total_limit", 3)),
        logging_steps=int(training.get("logging_steps", 1)),
        bf16=bool(training.get("bf16", True)),
        fp16=bool(training.get("fp16", False)),
        tf32=training.get("tf32", True),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", False)),
        report_to=logging.get("report_to", "wandb"),
        run_name=logging.get("run_name"),
        remove_unused_columns=True,
        dataloader_num_workers=workers,
        dataloader_pin_memory=bool(runtime.get("pin_memory", True)),
        dataloader_persistent_workers=workers > 0,
        seed=int(training.get("seed", 42)),
        average_tokens_across_devices=False,
    )
    trainer = FSDPTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=make_collator(_padding_token_id(tokenizer)),
        processing_class=tokenizer,
    )
    if accelerator.is_local_main_process:
        size = (
            "streaming"
            if isinstance(train_dataset, IterableDataset)
            else len(train_dataset)
        )
        print(
            f"Supervised post-training on {size} rows from {model_config['checkpoint']}"
        )
    trainer.train(resume_from_checkpoint=training.get("resume_from_checkpoint") or None)
    trainer.save_model(training["output_dir"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    train(_load_config(args.config))


if __name__ == "__main__":
    main()
