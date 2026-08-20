import io
import os
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from nar_tts.core.tokens import TokenLayout
from nar_tts.preprocessing.fast_pipeline import (
    AsyncParquetWriter,
    ParallelAudioDecoder,
    batch_tokenize,
    encode_audio_parquet,
    iter_length_batches,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts, **kwargs):
        self.calls += 1
        return {"input_ids": [[ord(char) for char in text] for text in texts]}


class FakeCodec:
    sampling_rate = 24_000

    def encode_to_device(self, waves):
        n_frames = [(wave.size + 1_919) // 1_920 for wave in waves]
        codes = np.zeros((len(waves), 2, max(n_frames)), dtype=np.int64)
        codes[:, 1] = 1
        return codes, n_frames

    @staticmethod
    def codes_to_numpy(codes, dtype=None):
        return codes


def make_wav_bytes(seconds, sampling_rate):
    buffer = io.BytesIO()
    sf.write(
        buffer, np.zeros(int(seconds * sampling_rate), dtype=np.float32),
        sampling_rate, format="WAV")
    return buffer.getvalue()


class FastPipelineTest(unittest.TestCase):
    def test_batch_audio_ids_and_tts_rows_match_scalar_path(self):
        layout = TokenLayout(base=100, eot=2, num_codebooks=3, codebook_size=8)
        codes = np.arange(2 * 3 * 4).reshape(2, 3, 4) % 8
        n_frames = [4, 2]
        texts = [[7, 8], [9]]

        expected_audio = [
            layout.codes_to_ids(codes[i, :, :n_frames[i]]) for i in range(2)]
        expected_rows = [
            layout.tts_sequence(texts[i], codes[i, :, :n_frames[i]])
            for i in range(2)]

        self.assertEqual(layout.codes_batch_to_ids(codes, n_frames), expected_audio)
        self.assertEqual(
            layout.tts_sequences(texts, codes, n_frames), expected_rows)
        offsets, values = layout.pack_tts_sequences(texts, codes, n_frames)
        packed_rows = [values[offsets[i]:offsets[i + 1]].tolist()
                       for i in range(len(texts))]
        self.assertEqual(packed_rows, expected_rows)

    def test_packed_qa_rows_match_list_path(self):
        layout = TokenLayout(base=100, eot=2)
        questions = [[10, 11], [12]]
        answers = [[20], [21, 22, 23]]
        expected = layout.qa_sequences(questions, answers)
        offsets, values = layout.pack_qa_sequences(questions, answers)
        packed = [values[offsets[i]:offsets[i + 1]].tolist()
                  for i in range(len(questions))]
        self.assertEqual(packed, expected)

    def test_length_batches_honor_clip_and_padded_sample_limits(self):
        lengths = [100, 10, 90, 20, 80, 30, 70, 40]
        clips = [(np.zeros(length, dtype=np.float32), str(length))
                 for length in lengths]
        batches = list(iter_length_batches(
            clips, batch_size=3, bucket_size=8, max_padded_samples=180))

        emitted = [int(clip[1]) for batch in batches for clip in batch]
        self.assertCountEqual(emitted, lengths)
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            if len(batch) > 1:
                padded = len(batch) * max(clip[0].size for clip in batch)
                self.assertLessEqual(padded, 180)

    def test_batch_tokenizer_uses_one_native_call(self):
        tokenizer = FakeTokenizer()
        encoded = batch_tokenize(tokenizer, ["ab", "c"])
        self.assertEqual(encoded, [[97, 98], [99]])
        self.assertEqual(tokenizer.calls, 1)

    def test_async_writer_preserves_rows_and_supports_empty_shards(self):
        rows = [[1, 2], [3], [4, 5, 6], [7], [8], [9]]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rows.parquet")
            with AsyncParquetWriter(
                    path, row_group_size=3, queue_size=2) as writer:
                writer.write(rows[:2])
                writer.write(rows[2:])
            self.assertEqual(
                pq.read_table(path).column("input_ids").to_pylist(), rows)

            empty_path = os.path.join(directory, "empty.parquet")
            with AsyncParquetWriter(empty_path):
                pass
            self.assertEqual(pq.read_metadata(empty_path).num_rows, 0)

            aborted_path = os.path.join(directory, "aborted.parquet")
            with self.assertRaisesRegex(RuntimeError, "stop"), \
                    AsyncParquetWriter(aborted_path) as writer:
                writer.write([[1, 2, 3]])
                raise RuntimeError("stop")
            self.assertFalse(os.path.exists(aborted_path))
            self.assertFalse(os.path.exists(aborted_path + ".tmp"))

    def test_end_to_end_streaming_shard_drops_invalid_rows(self):
        audios = [
            {"bytes": make_wav_bytes(0.1, 16_000), "path": None},
            {"bytes": make_wav_bytes(0.2, 24_000), "path": None},
            {"bytes": make_wav_bytes(0.3, 22_050), "path": None},
            None,
            {"bytes": make_wav_bytes(0.8, 24_000), "path": None},
        ]
        with tempfile.TemporaryDirectory() as directory:
            src = os.path.join(directory, "source.parquet")
            out = os.path.join(directory, "encoded.parquet")
            pq.write_table(pa.table({
                "audio": audios,
                "text": ["a", "bb", "ccc", "bad", "too long"],
            }), src)
            layout = TokenLayout(
                base=100, eot=2, num_codebooks=2, codebook_size=8)
            with ParallelAudioDecoder(3, prefetch=6) as decoder:
                stats = encode_audio_parquet(
                    src, out, FakeTokenizer(), layout, FakeCodec(),
                    decoder=decoder, batch_size=2, bucket_size=4,
                    max_batch_seconds=0.5, min_seconds=0.15,
                    max_seconds=0.5, row_group_size=2)

            rows = pq.read_table(out).column("input_ids").to_pylist()
            self.assertEqual(stats.input_rows, 5)
            self.assertEqual(stats.output_rows, 2)
            self.assertEqual(stats.dropped_rows, 3)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row[0] == layout.soh for row in rows))
            self.assertTrue(all(row[-1] == layout.eos_speech for row in rows))


if __name__ == "__main__":
    unittest.main()
