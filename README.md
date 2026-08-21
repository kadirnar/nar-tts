# Nar TTS

Nar TTS combines a user-selected causal language model with Mimi speech tokens.
It supports quality-gated inference, expressive controls, supervised training,
and GRPO without model-specific config files.

## Install

Python 3.10 and a CUDA 12.x environment are required.

```bash
git clone https://github.com/kadirnar/nar-tts.git
cd nar-tts
pip install -r requirements.txt
pip install -e .
```

Install the speaker evaluator for default quality inference and GRPO:

```bash
pip install -e ".[evaluation]"
```

## Inference

Inference defaults are embedded in the code. Set the checkpoint on the command
line; an optional [override file](nar_tts/configs/inference/override.yaml) is
available for persistent changes.

```bash
nar-tts infer \
  --checkpoint checkpoints/latest \
  --text "Bugün güzel bir gün." \
  --reference reference.wav \
  --reference-text "Bu ses klibinin doğru metni." \
  --output output.wav \
  --language tr
```

The default flow generates two candidates, verifies ASR and speaker identity,
and tries up to four candidates only when needed. It also writes a JSON quality
report. Batch input uses `--manifest jobs.jsonl`; long text adds `--long-form`.

Emotion controls require a checkpoint trained with the same control schema:

```bash
nar-tts infer \
  --checkpoint checkpoints/latest \
  --text "Bugün seni çok özledim." \
  --reference reference.wav \
  --reference-text "Bu ses klibinin doğru metni." \
  --output sad.wav \
  --emotion sadness --intensity 0.9 --delivery crying_speech \
  --event '{"type":"sob","after_word":2,"duration":"short"}'
```

## Data and quality

```bash
nar-tts audit-data --manifest raw.jsonl \
  --accepted clean.jsonl --rejected rejected.jsonl

nar-tts codec-check --audio "samples/*.wav" --output codec-report.json

nar-tts encode-expressive --manifest clean.jsonl \
  --output expressive.parquet --tokenizer /path/to/tokenizer

nar-tts evaluate --manifest generations.jsonl \
  --output evaluation.json --listening-manifest listening.jsonl

nar-tts distill --reports "infer_out/*.json" \
  --output verified-sft.jsonl
```

Raw speech preprocessing uses
[`preprocess.yaml`](nar_tts/configs/train/preprocess.yaml):

```bash
nar-tts preprocess --config nar_tts/configs/train/preprocess.yaml
```

## Training

```bash
nar-tts inspect-tokenizer --model /path/to/base-model

accelerate launch --config_file nar_tts/configs/train/launch/fsdp.yaml \
  nar_tts/training/pretrain.py --config nar_tts/configs/train/pretrain.yaml

accelerate launch --config_file nar_tts/configs/train/launch/fsdp.yaml \
  nar_tts/training/finetune.py --config nar_tts/configs/train/finetune.yaml

torchrun --standalone --nproc-per-node=8 \
  nar_tts/training/grpo.py --config nar_tts/configs/train/grpo.yaml
```

Write the selected tokenizer's `text_eos_token_id` and `pad_token_id` into each
training config before starting. All training stages support W&B through their
`logging` section. A separate LLaMA Factory YAML is not needed for Nar inference.

See [training and datasets](docs/training.md), [quality](docs/quality.md),
[emotion](docs/emotion.md), and [GRPO](docs/grpo.md).

## Credits

[Orpheus TTS](https://github.com/canopyai/Orpheus-TTS),
[Qwen3](https://huggingface.co/Qwen), and
[Mimi](https://huggingface.co/kyutai/mimi).
