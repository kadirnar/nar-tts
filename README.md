# Nar TTS

Nar TTS is a voice-cloning text-to-speech project built with
[Qwen3](https://huggingface.co/Qwen) and the
[Mimi](https://huggingface.co/kyutai/mimi) audio codec. It generates Mimi audio
tokens from text, then decodes them into speech. The project is based on the
[Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) recipe.

## Installation

Nar TTS requires Python 3.10 and a CUDA 12.x environment.

```bash
git clone https://github.com/kadirnar/nar-tts.git
cd nar-tts
pip install -r requirements.txt
pip install -e .
```

Log in with `hf auth login` if you need access to private Hugging Face models or
datasets. For WavLM-Large + ECAPA speaker scoring, also install:

```bash
pip install -e ".[speaker-quality]"
```

## Inference

```python
from nar_tts.inference.infer import NarTTS

engine = NarTTS(
    checkpoint="checkpoints/checkpoint-171622",
    tokenizer_name="Qwen/Qwen3-0.6B",
    device="cuda:0",
)

engine.clone(
    text="Hello, this is a cloned voice.",
    ref_wav="andrew.wav",
    ref_text="all sorts of things, mainly browsing.",
    output_path="output.wav",
)
```

For batch synthesis, edit `CKPT` and `JOBS` in the inference script, then run:

```bash
python nar_tts/inference/infer.py
```

## Dataset preparation

The preprocessing commands convert audio and text into Parquet files containing
model-ready `input_ids`:

```bash
python nar_tts/preprocessing/encode_textqa.py
python nar_tts/preprocessing/encode_pretrain.py
python nar_tts/preprocessing/encode_finetune.py
```

Configure large pretraining jobs in
[`nar_tts/configs/preprocess_pretrain.yaml`](nar_tts/configs/preprocess_pretrain.yaml).
The pipeline can stream source shards, upload outputs to Hugging Face, and resume
interrupted runs. Use a separate config file with:

```bash
python nar_tts/preprocessing/encode_pretrain.py --config /path/to/preprocess.yaml
```

Check the Mimi encode/decode round trip with:

```bash
python tests/test_mimi.py
```

## Training

```bash
# Pretraining (8-GPU FSDP)
accelerate launch --config_file nar_tts/configs/launch/fsdp.yaml \
  nar_tts/training/pretrain.py

# Supervised fine-tuning (8-GPU FSDP)
accelerate launch --config_file nar_tts/configs/launch/fsdp.yaml \
  nar_tts/training/finetune.py \
  --config nar_tts/configs/finetune.yaml

# GRPO (single GPU)
accelerate launch --config_file nar_tts/configs/launch/single_gpu.yaml \
  nar_tts/training/grpo.py \
  --config nar_tts/configs/grpo_intelligibility.yaml
```

Additional configs cover LoRA, multi-GPU training, and vLLM/SGLang rollouts.
See the [GRPO and post-training guide](docs/grpo.md) for details.

## Acknowledgements

- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS)
- [Kyutai Mimi](https://huggingface.co/kyutai/mimi)
- [Qwen3](https://huggingface.co/Qwen)
