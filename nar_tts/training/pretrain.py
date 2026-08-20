import os
from pathlib import Path

import torch
import wandb
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer, TrainingArguments

from nar_tts.core.data import GradualRatioDataset, make_collator
from nar_tts.core.tokens import TokenLayout
from nar_tts.core.trainer import RatioTrainer

# Pretrain a Qwen3 + Mimi TTS model with a decaying text/speech mixture: a text-QA
# stream keeps the LM's language ability while a speech stream teaches TTS, the mix
# decaying from text-heavy toward pure speech. All knobs live in the YAML below.
# Launch:  accelerate launch --config_file nar_tts/configs/launch/fsdp.yaml \
#                            nar_tts/training/pretrain.py
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pretrain.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# "start:end" text-to-speech ratio, linearly decayed across training.
initial_ratio, final_ratio = (int(x) for x in cfg["ratio"].split(":"))

wandb.init(project=cfg["project_name"], name=cfg["run_name"])
accelerator = Accelerator()                 # initializes the distributed env

tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
layout = TokenLayout.from_tokenizer(tokenizer)

model = AutoLigerKernelForCausalLM.from_pretrained(
    cfg["model_name"], attn_implementation="kernels-community/flash-attn2",
    torch_dtype=torch.bfloat16)
model = model.to(accelerator.device)        # init on one GPU before FSDP wraps it

# Register the Mimi audio tokens and grow the embedding table to match.
tokenizer.add_tokens(layout.added_token_strings())
model.resize_token_embeddings(len(tokenizer))

# Shards load language-blocked and the sampler does NOT shuffle, so interleave
# EN/JA here — otherwise the model sees one whole language then the other and
# catastrophically forgets the first.
text_ds = load_dataset(
    "parquet", data_files=os.path.join(cfg["text_QA_dataset"], "*.parquet"),
    split="train").shuffle(seed=42)
speech_ds = load_dataset(
    "parquet", data_files=os.path.join(cfg["TTS_dataset"], "*.parquet"),
    split="train").shuffle(seed=43)

batch_total = cfg["batch_size"] * cfg["number_processes"]
total_steps = int((len(text_ds) // batch_total) * cfg["epochs"])

train_dataset = GradualRatioDataset(
    text_ds, speech_ds, batch_total,
    initial_ratio=initial_ratio, final_ratio=final_ratio, total_steps=total_steps)

training_args = TrainingArguments(
    num_train_epochs=cfg["epochs"],
    per_device_train_batch_size=cfg["batch_size"],
    learning_rate=cfg["learning_rate"],
    lr_scheduler_type="cosine",
    save_steps=cfg["save_steps"],
    output_dir=f"./{cfg['save_folder']}",
    logging_steps=1,
    bf16=True,
    report_to="wandb",
    remove_unused_columns=True,
    average_tokens_across_devices=False)

trainer = RatioTrainer(
    model=model, args=training_args, train_dataset=train_dataset,
    data_collator=make_collator(cfg["pad_token"]),
    processing_class=tokenizer,
    initial_ratio=initial_ratio, final_ratio=final_ratio)

print(f"Pretraining: ratio {initial_ratio}:1 -> {final_ratio}:1, {total_steps} total steps")
trainer.train()                             # always from scratch (no resume)
