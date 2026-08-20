import os

import torch
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import TrainingArguments

from nar_tts.core.data import make_collator
from nar_tts.core.trainer import FSDPTrainer

# Config (hardcoded; no CLI args). Continues from a pretrained checkpoint (whose
# vocab is already resized) on a small single-voice set encoded by
# preprocessing/encode_finetune.py. Pure TTS: no text mixing, no ratio schedule.
# Launch:  accelerate launch --config_file nar_tts/configs/accelerate_config.yaml \
#                            -m nar_tts.training.finetune
PRETRAINED_CKPT = "checkpoints/checkpoint-171622"
FT_DATASET = "/scratch/kadirnar/elevenlabs-en-ft-mimi32/data"
OUTPUT_DIR = "checkpoints_ft_elevenlabs_en"
PROJECT_NAME = "qwen3-ja-en-tts_finetune"
RUN_NAME = "ft-elevenlabs-en-iP95p4"

EPOCHS = 10                     # 2000 clips is tiny; more passes for voice adaptation
BATCH_SIZE = 8                  # per device
LEARNING_RATE = 1.0e-5          # low: adapt the voice, don't wreck pretrained knowledge
WARMUP_RATIO = 0.03
SAVE_STEPS = 100
PAD_TOKEN = 151643              # Qwen3 <|endoftext|> (collator pad value)

wandb.init(project=PROJECT_NAME, name=RUN_NAME)
accelerator = Accelerator()

# Load FROM the pretrained checkpoint: its config already carries the resized
# vocab (Qwen3 + Mimi tokens), so there is no add_tokens / resize step here.
model = AutoLigerKernelForCausalLM.from_pretrained(
    PRETRAINED_CKPT, attn_implementation="kernels-community/flash-attn2",
    torch_dtype=torch.bfloat16).to(accelerator.device)

train_dataset = load_dataset(
    "parquet", data_files=os.path.join(FT_DATASET, "*.parquet"),
    split="train").shuffle(seed=42)
if accelerator.is_local_main_process:
    print(f"Finetuning on {len(train_dataset)} clips from {PRETRAINED_CKPT}")

training_args = TrainingArguments(
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    output_dir=OUTPUT_DIR,
    logging_steps=1,
    bf16=True,
    report_to="wandb",
    remove_unused_columns=True,
    average_tokens_across_devices=False)

trainer = FSDPTrainer(
    model=model, args=training_args, train_dataset=train_dataset,
    data_collator=make_collator(PAD_TOKEN))

# Finetune (continues from PRETRAINED_CKPT weights; not a resume of its optimizer).
trainer.train()
trainer.save_model(OUTPUT_DIR)
