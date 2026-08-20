import multiprocessing as mp
import os
import time

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

from nar_tts.core.tokens import TokenLayout
from nar_tts.preprocessing.fast_pipeline import (
    AsyncParquetWriter,
    batch_tokenize,
)

# CPU-only text-QA stream. Each process keeps one tokenizer alive, invokes its
# native batch path, and overlaps the next record batch with Parquet compression.
SOURCES = [
    ("en", "SynDataLab/DeepSeekFlash-3M-en"),
    ("ja", "SynDataLab/DeepSeekFlash-3M-ja"),
]
TOKENIZER_NAME = "Qwen/Qwen3-4B"
OUTPUT_ROOT = "/scratch/kadirnar/textqa-qwen3"
OUTPUT_DIR = OUTPUT_ROOT + "/data"
LOG_PATH = OUTPUT_ROOT + "/encode.log"

CPU_COUNT = os.cpu_count() or 1
NUM_WORKERS = int(os.environ.get(
    "NAR_TTS_TEXT_WORKERS", str(max(1, min(4, CPU_COUNT // 4)))))
# Zero means divide available CPU threads between the workers selected after the
# shard list is known. One thread per process is reserved for Arrow/write work.
TOKENIZER_THREADS = int(os.environ.get("NAR_TTS_TOKENIZER_THREADS", "0"))
TOKENIZE_BATCH_SIZE = int(os.environ.get("NAR_TTS_TEXT_BATCH_SIZE", "2048"))
ROW_GROUP_SIZE = int(os.environ.get("NAR_TTS_ROW_GROUP_SIZE", "1024"))
MAX_LEN = 2048
MIN_LEN = 4
DELETE_SOURCE_CACHE = True
TOKEN = os.environ.get("HF_TOKEN")

_TOKENIZER = None
_LAYOUT = None


def _log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _initialize_worker(tokenizer_threads):
    global _TOKENIZER, _LAYOUT
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(tokenizer_threads)
    _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    _LAYOUT = TokenLayout.from_tokenizer(_TOKENIZER)


def worker(args):
    """Batch-encode one ``(language, repository, shard)`` into QA sequences."""
    lang, repo, filename = args
    out_path = os.path.join(OUTPUT_DIR, f"{lang}-{os.path.basename(filename)}")
    if os.path.exists(out_path):
        return "skip", 0, 0
    if _TOKENIZER is None:  # supports a direct worker() call in tests/tools
        _initialize_worker(max(1, TOKENIZER_THREADS or CPU_COUNT - 1))

    src = hf_hub_download(repo, filename, repo_type="dataset", token=TOKEN)
    rows_written = dropped = 0
    parquet = pq.ParquetFile(src)
    with AsyncParquetWriter(
            out_path, queue_size=2,
            row_group_size=ROW_GROUP_SIZE) as sink:
        for batch in parquet.iter_batches(
                batch_size=TOKENIZE_BATCH_SIZE,
                columns=["question", "answer"], use_threads=True):
            raw_questions = batch.column(0).to_pylist()
            raw_answers = batch.column(1).to_pylist()
            pairs = [(question, answer)
                     for question, answer in zip(raw_questions, raw_answers)
                     if question and answer]
            dropped += len(raw_questions) - len(pairs)
            if not pairs:
                continue

            questions, answers = zip(*pairs)
            # One native batch call has much lower Python overhead and lets
            # the tokenizer's bounded Rayon pool work across both columns.
            encoded = batch_tokenize(
                _TOKENIZER, list(questions) + list(answers))
            split = len(questions)
            question_ids, answer_ids = encoded[:split], encoded[split:]
            keep = [i for i, (question, answer) in enumerate(
                zip(question_ids, answer_ids))
                if MIN_LEN <= len(question) + len(answer) + 5 <= MAX_LEN]
            dropped += len(question_ids) - len(keep)
            if keep:
                offsets, values = _LAYOUT.pack_qa_sequences(
                    (question_ids[i] for i in keep),
                    (answer_ids[i] for i in keep))
                rows_written += len(keep)
                sink.write_packed(offsets, values)
    if DELETE_SOURCE_CACHE:
        try:
            os.remove(os.path.realpath(src))
        except OSError:
            pass
    return "ok", rows_written, dropped


def output_path(task):
    lang, _, filename = task
    return os.path.join(OUTPUT_DIR, f"{lang}-{os.path.basename(filename)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = []
    for lang, repo in SOURCES:
        shards = sorted(
            filename for filename in list_repo_files(
                repo, repo_type="dataset", token=TOKEN)
            if filename.endswith(".parquet"))
        tasks.extend((lang, repo, filename) for filename in shards)

    if not tasks:
        _log("no parquet shards found")
        return
    all_tasks = tasks
    tasks = [task for task in all_tasks if not os.path.exists(output_path(task))]
    cached_tasks = len(all_tasks) - len(tasks)
    if not tasks:
        _log(f"all {cached_tasks} source shards are already encoded")
        return
    workers = min(NUM_WORKERS, len(tasks))
    tokenizer_threads = TOKENIZER_THREADS or max(
        1, (CPU_COUNT - workers) // workers)
    _log(f"encoding {len(tasks)} shards ({cached_tasks} cached, "
         f"{len(SOURCES)} sources) -> {OUTPUT_DIR} | "
         f"{workers} processes x {tokenizer_threads} tokenizer threads | "
         f"batch={TOKENIZE_BATCH_SIZE}")

    done = rows = dropped = skipped = 0
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
            workers, initializer=_initialize_worker,
            initargs=(tokenizer_threads,)) as pool:
        for status, count, rejected in pool.imap_unordered(
                worker, tasks, chunksize=1):
            done += 1
            rows += count
            dropped += rejected
            skipped += status == "skip"
            if done % 10 == 0 or done == len(tasks):
                elapsed = max(time.time() - t0, 1e-9)
                _log(f"{done}/{len(tasks)} shards | {rows} new rows | "
                     f"{dropped} dropped | {skipped} cached | "
                     f"{rows / elapsed:.1f} rows/s")

    names = [name for name in os.listdir(OUTPUT_DIR) if name.endswith(".parquet")]
    total_rows = sum(
        pq.read_metadata(os.path.join(OUTPUT_DIR, name)).num_rows for name in names)
    _log(f"DONE | {len(names)} output shards | {total_rows} total rows")


if __name__ == "__main__":
    main()
