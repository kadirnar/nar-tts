# Nar TTS

Nar TTS is a Qwen3 + Mimi voice-cloning TTS project. It supports quality-gated
batch inference, expressive controls, supervised training, and GRPO.

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

Set the checkpoint and device once in
[`inference.yaml`](nar_tts/configs/inference.yaml), then run:

```bash
nar-tts infer \
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
  --output expressive.parquet

nar-tts evaluate --manifest generations.jsonl \
  --output evaluation.json --listening-manifest listening.jsonl

nar-tts distill --reports "infer_out/*.json" \
  --output verified-sft.jsonl
```

Existing large preprocessing jobs still use
[`preprocess_pretrain.yaml`](nar_tts/configs/preprocess_pretrain.yaml).

## Training

```bash
accelerate launch --config_file nar_tts/configs/launch/fsdp.yaml \
  nar_tts/training/pretrain.py

accelerate launch --config_file nar_tts/configs/launch/fsdp.yaml \
  nar_tts/training/finetune.py --config nar_tts/configs/finetune.yaml

torchrun --standalone --nproc-per-node=8 \
  nar_tts/training/grpo.py --config nar_tts/configs/grpo.yaml
```

There is one quality-first GRPO config. LLaMA Factory is inference-only; its
config is under `nar_tts/configs/llama_factory/` and no dataset registry is
needed.

See [quality](docs/quality.md), [emotion](docs/emotion.md),
[GRPO](docs/grpo.md), and the [2026 TTS review](docs/tts_2026.md).

## Credits

[Orpheus TTS](https://github.com/canopyai/Orpheus-TTS),
[Qwen3](https://huggingface.co/Qwen), and
[Mimi](https://huggingface.co/kyutai/mimi).
