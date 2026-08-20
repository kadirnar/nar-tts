"""High-throughput building blocks shared by the dataset encoders.

The speech path is a bounded producer/consumer pipeline:

    parquet -> parallel decode -> duration buckets -> Mimi GPU -> parquet writer

Only the Mimi stage owns the GPU. CPU decoding continues while it runs, text is
batch-tokenized while queued CUDA kernels finish, and Arrow compression happens
on a background thread. Bounded queues keep that overlap without allowing a
large source shard to consume unbounded RAM.
"""

import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent import futures
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

_END = object()


@dataclass
class EncodeStats:
    """Counters and utilization information for one encoded speech shard."""

    input_rows: int = 0
    output_rows: int = 0
    dropped_rows: int = 0
    audio_samples: int = 0
    padded_samples: int = 0
    elapsed_seconds: float = 0.0

    @property
    def padding_efficiency(self):
        if not self.padded_samples:
            return 1.0
        return self.audio_samples / self.padded_samples


class _ProducerError:
    def __init__(self, exception):
        self.exception = exception


@dataclass
class _PackedRows:
    offsets: object
    values: object


def _initialize_decode_process():
    # A process per clip plus PyTorch's default intra-op pool grossly
    # oversubscribes a CPU. Resampling individual clips is already parallel here.
    import torch

    torch.set_num_threads(1)


def _decode_audio_record(*args):
    # Keep the CPU-only text pipeline from importing torch/torchaudio/Mimi just
    # because it shares the Arrow writer and batch-tokenizer helpers in this file.
    from nar_tts.core.audio import decode_audio_record

    return decode_audio_record(*args)


class ParallelAudioDecoder:
    """Persistent bounded executor for audio decode and resampling.

    Threads are the default because libsndfile and torchaudio do their heavy work
    in native code and NumPy waveforms then reach the GPU process without an IPC
    copy. ``backend="process"`` remains useful for codecs whose decoder holds the
    GIL.
    """

    def __init__(self, workers, prefetch=None, backend="thread"):
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if backend not in {"thread", "process"}:
            raise ValueError("backend must be 'thread' or 'process'")
        self.workers = int(workers)
        self.prefetch = max(self.workers, int(prefetch or self.workers * 4))
        self.backend = backend
        self._executor = None

    def __enter__(self):
        if self._executor is not None:
            return self
        if self.backend == "thread":
            self._executor = futures.ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="audio-decode")
        else:
            self._executor = futures.ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context("spawn"),
                initializer=_initialize_decode_process)
        return self

    def __exit__(self, exc_type, exc, traceback):
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def decode(self, records, target_sr, min_seconds, max_seconds):
        """Yield successfully decoded ``(wave, text)`` rows as they finish."""
        if self._executor is None:
            raise RuntimeError("ParallelAudioDecoder must be used as a context manager")

        output = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def put(item):
            while not stop.is_set():
                try:
                    output.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def produce():
            pending = set()
            record_iter = iter(records)

            def submit_one():
                try:
                    audio, text = next(record_iter)
                except StopIteration:
                    return False
                pending.add(self._executor.submit(
                    _decode_audio_record, audio, text, target_sr,
                    min_seconds, max_seconds))
                return True

            try:
                for _ in range(self.prefetch):
                    if not submit_one():
                        break
                while pending and not stop.is_set():
                    done, _ = futures.wait(
                        pending, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        pending.remove(future)
                        decoded = future.result()
                        # Refill before a possibly blocking queue put so remaining
                        # workers stay busy while the GPU consumes prior buckets.
                        submit_one()
                        if decoded is not None and not put(decoded):
                            break
            except Exception as exc:  # noqa: BLE001 - relay producer/Arrow failures
                put(_ProducerError(exc))
            finally:
                for future in pending:
                    future.cancel()
                put(_END)

        producer = threading.Thread(
            target=produce, name="audio-decode-producer", daemon=True)
        producer.start()
        try:
            while True:
                item = output.get()
                if item is _END:
                    break
                if isinstance(item, _ProducerError):
                    raise item.exception
                yield item
        finally:
            stop.set()
            producer.join()


def iter_parquet_audio_records(path, read_batch_size=2048,
                               audio_column="audio", text_column="text"):
    """Stream Python audio/text rows without materializing a whole shard."""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=[audio_column, text_column], use_threads=True):
        audios = batch.column(0).to_pylist()
        texts = batch.column(1).to_pylist()
        yield from zip(audios, texts)


def iter_length_batches(clips, batch_size, bucket_size=None,
                        max_padded_samples=None):
    """Sort bounded windows by duration and emit compute-budgeted batches.

    Mimi computes over the longest waveform in a batch. Local sorting therefore
    removes most zero-padding while retaining streaming and bounded memory.
    ``max_padded_samples`` caps ``batch * longest_length`` for smaller GPUs.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    bucket_size = max(batch_size, int(bucket_size or batch_size * 8))

    def emit(bucket):
        bucket.sort(key=lambda clip: clip[0].size)
        batch = []
        for clip in bucket:
            proposed_size = (len(batch) + 1) * clip[0].size
            over_budget = (max_padded_samples is not None
                           and proposed_size > max_padded_samples)
            if batch and (len(batch) >= batch_size or over_budget):
                yield batch
                batch = []
            batch.append(clip)
        if batch:
            yield batch

    bucket = []
    for clip in clips:
        bucket.append(clip)
        if len(bucket) >= bucket_size:
            yield from emit(bucket)
            bucket = []
    if bucket:
        yield from emit(bucket)


def batch_tokenize(tokenizer, texts):
    """Use the fast tokenizer's native batch/Rayon path without extra outputs."""
    if not texts:
        return []
    encoded = tokenizer(
        list(texts), add_special_tokens=False, padding=False, truncation=False,
        return_attention_mask=False, return_token_type_ids=False)
    return encoded["input_ids"]


class AsyncParquetWriter:
    """Compress list<int32> rows off the critical GPU thread, then atomically commit."""

    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])

    def __init__(self, path, compression="snappy", queue_size=4,
                 row_group_size=256):
        self.path = os.fspath(path)
        self.tmp_path = self.path + ".tmp"
        self.compression = compression
        self.row_group_size = int(row_group_size)
        if self.row_group_size < 1:
            raise ValueError("row_group_size must be at least 1")
        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._error = None
        self._thread = None

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="parquet-writer", daemon=True)
        self._thread.start()
        return self

    def _to_array(self, rows):
        if isinstance(rows, _PackedRows):
            return pa.ListArray.from_arrays(
                pa.array(rows.offsets, type=pa.int32()),
                pa.array(rows.values, type=pa.int32()))
        return pa.array(rows, type=pa.list_(pa.int32()))

    def _write_array(self, writer, array):
        writer.write_table(
            pa.Table.from_arrays([array], schema=self.schema),
            row_group_size=len(array))

    def _run(self):
        pending = []
        pending_rows = 0
        try:
            with pq.ParquetWriter(
                    self.tmp_path, self.schema, compression=self.compression,
                    use_dictionary=False, write_statistics=False) as writer:
                while True:
                    rows = self._queue.get()
                    if rows is _END:
                        break
                    array = self._to_array(rows)
                    pending.append(array)
                    pending_rows += len(array)
                    if pending_rows >= self.row_group_size:
                        combined = (pending[0] if len(pending) == 1
                                    else pa.concat_arrays(pending))
                        while len(combined) >= self.row_group_size:
                            self._write_array(
                                writer, combined.slice(0, self.row_group_size))
                            combined = combined.slice(self.row_group_size)
                        pending = [combined] if len(combined) else []
                        pending_rows = len(combined)
                if pending:
                    combined = (pending[0] if len(pending) == 1
                                else pa.concat_arrays(pending))
                    self._write_array(writer, combined)
        except Exception as exc:  # noqa: BLE001 - report errors on the caller thread
            self._error = exc

    def _enqueue(self, rows):
        while True:
            if self._error is not None:
                raise self._error
            try:
                self._queue.put(rows, timeout=0.1)
                return
            except queue.Full:
                pass

    def write(self, rows):
        """Queue ordinary Python list rows for backward-compatible callers."""
        if rows:
            self._enqueue(rows)

    def write_packed(self, offsets, values):
        """Queue zero-copy Arrow-ready list offsets and contiguous int32 values."""
        if len(offsets) > 1:
            self._enqueue(_PackedRows(offsets, values))

    def close(self, commit=True):
        if self._thread is None:
            return
        while self._thread.is_alive():
            if self._error is not None:
                break
            try:
                self._queue.put(_END, timeout=0.1)
                break
            except queue.Full:
                pass
        self._thread.join()
        self._thread = None
        error, self._error = self._error, None
        if error is None and commit:
            os.replace(self.tmp_path, self.path)
            return
        try:
            os.remove(self.tmp_path)
        except OSError:
            pass
        if error is not None:
            raise error

    def __exit__(self, exc_type, exc, traceback):
        self.close(commit=exc_type is None)


def encode_audio_parquet(src_path, out_path, tokenizer, layout, codec, *,
                         decoder, batch_size=32, bucket_size=None,
                         max_batch_seconds=None, read_batch_size=2048,
                         min_seconds=0.2, max_seconds=30.0,
                         writer_queue_size=4, row_group_size=256):
    """Stream one audio/text parquet shard through the complete fast pipeline."""
    import torch

    started = time.perf_counter()
    stats = EncodeStats(input_rows=pq.ParquetFile(src_path).metadata.num_rows)
    records = iter_parquet_audio_records(src_path, read_batch_size=read_batch_size)
    decoded = decoder.decode(
        records, codec.sampling_rate, min_seconds=min_seconds,
        max_seconds=max_seconds)
    max_padded_samples = None
    if max_batch_seconds is not None:
        max_padded_samples = int(max_batch_seconds * codec.sampling_rate)
    batches = iter_length_batches(
        decoded, batch_size=batch_size, bucket_size=bucket_size,
        max_padded_samples=max_padded_samples)

    with AsyncParquetWriter(
            out_path, queue_size=writer_queue_size,
            row_group_size=row_group_size) as sink:

        def encode_batch(clips):
            waves = [clip[0] for clip in clips]
            texts = [clip[1] for clip in clips]
            out_of_memory = False
            try:
                device_codes, n_frames = codec.encode_to_device(waves)
            except torch.cuda.OutOfMemoryError:
                if len(clips) == 1:
                    raise
                out_of_memory = True
            if out_of_memory:
                # A conservative budget should prevent this in steady state. If
                # hardware/model settings differ, recover once and split instead
                # of losing an otherwise valid shard after hours of work. This is
                # outside the except block so its traceback no longer pins tensors.
                torch.cuda.empty_cache()
                middle = len(clips) // 2
                encode_batch(clips[:middle])
                encode_batch(clips[middle:])
                return

            # Mimi has enqueued its CUDA kernels. The Rust tokenizer can now use
            # CPU cores while the GPU finishes before codes_to_numpy synchronizes.
            text_ids = batch_tokenize(tokenizer, texts)
            codes = codec.codes_to_numpy(device_codes, dtype=torch.int32)
            offsets, values = layout.pack_tts_sequences(
                text_ids, codes, n_frames=n_frames)
            sink.write_packed(offsets, values)

            stats.output_rows += len(text_ids)
            stats.audio_samples += sum(wave.size for wave in waves)
            stats.padded_samples += max(wave.size for wave in waves) * len(waves)

        for clips in batches:
            encode_batch(clips)

    stats.dropped_rows = stats.input_rows - stats.output_rows
    stats.elapsed_seconds = time.perf_counter() - started
    return stats
