# GRPO post-training

Nar uses one quality-focused GRPO recipe:
[`configs/train/grpo.yaml`](../nar_tts/configs/train/grpo.yaml). It optimizes Mimi
speech tokens with constrained generation. Each reward is normalized inside
its own prompt group before the weighted sum:

| Reward | Weight | Model |
|---|---:|---|
| Intelligibility | 0.60 | Qwen3-ASR CER + NLL |
| Speaker similarity | 0.15 | WavLM-Large + ECAPA |
| Duration consistency | 0.05 | Reference duration |
| Technical quality | 0.08 | Clipping, level, silence, repetition |
| Prosody | 0.05 | Pitch, energy, voicing, pauses |
| Speaker drift | 0.07 | Overlapping WavLM windows |

The recipe uses one group of eight generations across eight GPUs. Style-only
Emotion and event rewards are implemented but remain at zero until their
classifiers pass held-out synthetic-speech and human evaluation.

## Run

Install the speaker evaluator:

```bash
pip install -e ".[evaluation]"
```

Set the checkpoint, dataset path, and output directory in `grpo.yaml`. Validate
the config without loading models:

```bash
python nar_tts/training/grpo.py \
  --config nar_tts/configs/train/grpo.yaml \
  --validate-only
```

Start training:

```bash
torchrun --standalone --nproc-per-node=8 \
  nar_tts/training/grpo.py \
  --config nar_tts/configs/train/grpo.yaml
```

## Design rules

- A Mimi frame contains 32 ordered codebook tokens.
- Invalid codebooks are masked during sampling and log-probability calculation.
- Speech EOS is legal only between complete frames.
- Each waveform is decoded once and shared by all reward models.
- The saved Nar tokenizer must be used from the checkpoint directory.

## Evaluation

Do not evaluate only with the Qwen3-ASR model used for training. Use an
independent ASR family, speaker-drift checks, multi-dimensional quality metrics,
and listening tests.
