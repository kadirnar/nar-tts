# Training sequence and data formats

## Initial setup

Training configuration files live under `nar_tts/configs/train/`. Optional
inference overrides live under `nar_tts/configs/inference/`. There are no
model-specific configuration files.

Display the tokenizer values:

```bash
nar-tts inspect-tokenizer --model /path/to/base-model
```

Copy the reported `text_eos_token_id` and `suggested_pad_token_id` values into
the `tokens` section of `pretrain.yaml`, `finetune.yaml`, and `grpo.yaml`. Also
set every model and data path marked `REQUIRED` in those files.

For SFT, `model.loader` may be either `transformers` or `unsloth`. Setting
`peft.enabled: true` enables LoRA with either loader; full training is the
default.

## Training sequence

1. Convert raw audio to Mimi tokens:

   ```bash
   nar-tts preprocess --config nar_tts/configs/train/preprocess.yaml
   ```

2. Train the base model with text and TTS data:

   ```bash
   accelerate launch --config_file nar_tts/configs/train/launch/fsdp.yaml \
     nar_tts/training/pretrain.py --config nar_tts/configs/train/pretrain.yaml
   ```

3. Optionally run high-quality SFT:

   ```bash
   accelerate launch --config_file nar_tts/configs/train/launch/fsdp.yaml \
     nar_tts/training/finetune.py --config nar_tts/configs/train/finetune.yaml
   ```

4. Optionally run quality-focused GRPO training:

   ```bash
   torchrun --standalone --nproc-per-node=8 \
     nar_tts/training/grpo.py --config nar_tts/configs/train/grpo.yaml
   ```

At each stage, use the tokenizer saved with the previous stage's checkpoint.

## Parquet formats

The preprocessing input requires `audio` and `text` columns. `audio` may be a
Hugging Face Audio object or a file path.

| Stage | Required field | Optional fields |
|---|---|---|
| Text pretraining | `input_ids: list[int]` | None |
| TTS pretraining/SFT | `input_ids: list[int]` | `speaker`, `language`, `emotion`, `events` |
| GRPO `tts_tokens` | `input_ids: list[int]` | `language`, `emotion`, `intensity`, `delivery`, `events`, `hard_case` |

`input_ids` is a complete Nar sequence containing text and Mimi speech tokens.
The raw WAV file is not read again during training.

Expressive JSONL input uses the following format:

```json
{"audio":"voice.wav","text":"Hello","speaker":"spk1","language":"en","emotion":"joy","intensity":0.8,"delivery":"speech_laugh","events":[{"type":"laugh","after_word":1}],"license":"owned"}
```

Convert this file to Parquet:

```bash
nar-tts encode-expressive --manifest expressive.jsonl \
  --output expressive.parquet --tokenizer /path/to/tokenizer
```

## W&B

All three training configurations set `logging.enabled: true` and
`report_to: wandb`. Run `wandb login` before the first use. Customize the
`project`, `run_name`, `entity`, `group`, `tags`, and `mode` fields as needed.
To disable W&B, set only `logging.enabled: false`.
