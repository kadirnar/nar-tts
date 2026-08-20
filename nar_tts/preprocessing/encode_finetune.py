import os

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from nar_tts.core.audio import MimiCodec
from nar_tts.core.tokens import TokenLayout
from nar_tts.preprocessing.fast_pipeline import (
    ParallelAudioDecoder,
    encode_audio_parquet,
)

# Single-speaker English set (~2000 clips). One GPU owns Mimi while a bounded
# CPU pool decodes/resamples ahead of it; no full-table Python materialization or
# cross-process waveform copies are required.
SRC_REPO = "Vyvo/ElevenLabs-EN-iP95p4xoKVk53GoZ742B"
SRC_FILE = "data/train-00000-of-00001.parquet"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_ROOT = "/scratch/kadirnar/elevenlabs-en-ft-mimi32"
OUTPUT_DIR = OUTPUT_ROOT + "/data"
TOKEN = os.environ.get("HF_TOKEN")

CPU_COUNT = os.cpu_count() or 1
GPU_ID = int(os.environ.get("NAR_TTS_GPU_ID", "0"))
CPU_WORKERS = int(os.environ.get(
    "NAR_TTS_CPU_WORKERS", str(max(1, CPU_COUNT - 2))))
DECODE_BACKEND = os.environ.get("NAR_TTS_DECODE_BACKEND", "thread")
DECODE_PREFETCH = int(os.environ.get(
    "NAR_TTS_DECODE_PREFETCH", str(max(4, CPU_WORKERS * 4))))
TOKENIZER_THREADS = int(os.environ.get("NAR_TTS_TOKENIZER_THREADS", "2"))
BATCH_SIZE = int(os.environ.get("NAR_TTS_GPU_BATCH_SIZE", "32"))
BUCKET_SIZE = int(os.environ.get("NAR_TTS_BUCKET_SIZE", str(BATCH_SIZE * 8)))
MAX_BATCH_SECONDS = float(os.environ.get("NAR_TTS_MAX_BATCH_SECONDS", "300"))
READ_BATCH_SIZE = int(os.environ.get("NAR_TTS_READ_BATCH_SIZE", "2048"))
ROW_GROUP_SIZE = int(os.environ.get("NAR_TTS_ROW_GROUP_SIZE", "256"))
MIN_AUDIO_SECONDS = 0.2
MAX_AUDIO_SECONDS = 30.0

MIMI_DTYPE = os.environ.get("NAR_TTS_MIMI_DTYPE") or None
MIMI_COMPILE = os.environ.get("NAR_TTS_MIMI_COMPILE", "0") == "1"
MIMI_COMPILE_MODE = os.environ.get("NAR_TTS_MIMI_COMPILE_MODE", "default")
ALLOW_TF32 = os.environ.get("NAR_TTS_ALLOW_TF32", "1") == "1"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "train-00.parquet")
    if os.path.exists(out_path):
        rows = pq.read_metadata(out_path).num_rows
        print(f"output exists, skip: {rows} rows in {out_path}", flush=True)
        return
    if GPU_ID >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {GPU_ID} is unavailable")

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(TOKENIZER_THREADS)
    torch.set_num_threads(1)
    torch.cuda.set_device(GPU_ID)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    layout = TokenLayout.from_tokenizer(tokenizer)
    codec = MimiCodec(
        f"cuda:{GPU_ID}", num_codebooks=layout.num_codebooks,
        dtype=MIMI_DTYPE, compile_model=MIMI_COMPILE,
        compile_mode=MIMI_COMPILE_MODE, allow_tf32=ALLOW_TF32)

    src = hf_hub_download(SRC_REPO, SRC_FILE, repo_type="dataset", token=TOKEN)
    print(f"encoding on cuda:{GPU_ID} | {CPU_WORKERS} decode workers | "
          f"batch={BATCH_SIZE} bucket={BUCKET_SIZE}", flush=True)
    with ParallelAudioDecoder(
            CPU_WORKERS, prefetch=DECODE_PREFETCH,
            backend=DECODE_BACKEND) as decoder:
        stats = encode_audio_parquet(
            src, out_path, tokenizer, layout, codec, decoder=decoder,
            batch_size=BATCH_SIZE, bucket_size=BUCKET_SIZE,
            max_batch_seconds=MAX_BATCH_SECONDS or None,
            read_batch_size=READ_BATCH_SIZE,
            min_seconds=MIN_AUDIO_SECONDS, max_seconds=MAX_AUDIO_SECONDS,
            row_group_size=ROW_GROUP_SIZE)

    rows_per_second = stats.output_rows / max(stats.elapsed_seconds, 1e-9)
    print(f"DONE {stats.output_rows} rows ({stats.dropped_rows} dropped) -> "
          f"{out_path} | {stats.elapsed_seconds:.1f}s | "
          f"{rows_per_second:.1f} rows/s | "
          f"{stats.padding_efficiency * 100:.1f}% useful audio", flush=True)


if __name__ == "__main__":
    main()
