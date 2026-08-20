import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from nar_tts.evaluation.data_quality import (
    AuditThresholds,
    audit_manifest,
    build_distillation_manifest,
)
from nar_tts.evaluation.metrics import (
    analyze_waveform,
    event_alignment_score,
    scale_invariant_sdr,
    speaker_drift_score,
    transcript_error_rate,
)
from nar_tts.evaluation.verifiers import (
    TransformersSpeakerVerifier,
    VerificationResult,
)
from nar_tts.inference.infer import (
    NarTTS,
    SynthesisRequest,
    crossfade_waveforms,
)
from nar_tts.inference.quality import Candidate, score_candidates, select_winners


class MetricTest(unittest.TestCase):
    def test_lexical_event_and_drift_metrics(self):
        self.assertEqual(transcript_error_rate("Merhaba!", "merhaba", "wer"), 0.0)
        event = event_alignment_score(
            [{"type": "laugh", "start_seconds": 1.0}],
            [{"type": "laugh", "start_seconds": 1.2}],
        )
        self.assertEqual(event.f1, 1.0)
        drift = speaker_drift_score([1.0, 0.0], [[1.0, 0.0], [0.9, 0.1]])
        self.assertGreater(drift["score"], 0.8)

    def test_waveform_diagnostics_and_reconstruction_metric(self):
        sample_rate = 8000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        clean = 0.1 * np.sin(2 * np.pi * 220 * time)
        clipped = np.sign(clean)
        clean_metrics = analyze_waveform(clean, sample_rate)
        clipped_metrics = analyze_waveform(clipped, sample_rate)
        self.assertGreater(
            clean_metrics.technical_quality, clipped_metrics.technical_quality
        )
        self.assertGreater(scale_invariant_sdr(clean, clean), 100.0)

    def test_candidate_gate_batches_verification_and_selects_pass(self):
        sample_rate = 8000
        wave = 0.1 * np.sin(
            2 * np.pi * 220 * np.arange(sample_rate, dtype=np.float32) / sample_rate
        )

        class Verifier:
            def __init__(self):
                self.calls = 0

            def verify(self, waves, sample_rate, texts, **kwargs):
                del sample_rate, texts, kwargs
                self.calls += 1
                return [
                    VerificationResult(cer=value, speaker_similarity=0.9)
                    for value in (0.4, 0.0)[: len(waves)]
                ]

        candidates = [
            Candidate(0, index, [index], wave.copy(), True) for index in range(2)
        ]
        verifier = Verifier()
        score_candidates(
            candidates,
            sample_rate=sample_rate,
            target_texts=["test"],
            expected_durations=[1.0],
            verifier=verifier,
            reference_waves=[wave],
            controls=[None],
        )
        self.assertEqual(verifier.calls, 1)
        self.assertFalse(candidates[0].accepted)
        self.assertTrue(candidates[1].accepted)
        self.assertIs(select_winners(candidates, 1)[0], candidates[1])

    def test_candidate_gate_rejects_missing_required_asr_metric(self):
        sample_rate = 8000
        wave = 0.1 * np.sin(
            2 * np.pi * 220 * np.arange(sample_rate, dtype=np.float32) / sample_rate
        )

        class SpeakerOnlyVerifier:
            def verify(self, waves, sample_rate, texts, **kwargs):
                del sample_rate, texts, kwargs
                return [
                    VerificationResult(speaker_similarity=0.9) for _ in waves
                ]

        candidate = Candidate(0, 0, [0], wave, True)
        score_candidates(
            [candidate],
            sample_rate=sample_rate,
            target_texts=["test"],
            expected_durations=[1.0],
            verifier=SpeakerOnlyVerifier(),
            reference_waves=[wave],
            controls=[None],
        )
        self.assertFalse(candidate.accepted)
        self.assertIn("asr_missing", candidate.rejection_reasons)

    def test_crossfade_removes_exact_overlap_length(self):
        first = np.ones(100, dtype=np.float32)
        second = np.zeros(100, dtype=np.float32)
        output = crossfade_waveforms([first, second], 1000, milliseconds=10)
        self.assertEqual(output.size, 190)
        self.assertTrue(np.all(np.diff(output[90:100]) <= 0))


class InferenceOrchestrationTest(unittest.TestCase):
    def test_speaker_verifier_caches_duplicate_reference_embeddings(self):
        verifier = TransformersSpeakerVerifier(reference_cache_size=4)
        calls = []

        def embeddings(waves, sample_rate):
            del sample_rate
            calls.append(len(waves))
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in waves]

        verifier.embeddings = embeddings
        wave = np.zeros(100, dtype=np.float32)
        first = verifier.similarities([wave, wave], [wave, wave], 8000)
        second = verifier.similarities([wave], [wave], 8000)
        self.assertEqual(first, [1.0, 1.0])
        self.assertEqual(second, [1.0])
        self.assertEqual(calls, [2, 1, 1])

    def test_reference_arrays_are_encoded_once_and_reused(self):
        engine = NarTTS.__new__(NarTTS)
        engine.reference_cache = OrderedDict()
        engine.reference_cache_size = 4

        class Codec:
            sampling_rate = 8000

            def __init__(self):
                self.calls = 0

            def encode(self, waves):
                self.calls += 1
                return np.zeros((len(waves), 2, 1), dtype=np.int64), [1] * len(waves)

        class Layout:
            def codes_batch_to_ids(self, codes, n_frames):
                return [[10, 11] for _ in range(len(codes))]

        engine.codec = Codec()
        engine.layout = Layout()
        wave = np.zeros(800, dtype=np.float32)
        request = SynthesisRequest(
            text="test",
            reference_audio=None,
            reference_text="reference",
            reference_waveform=wave,
            reference_sample_rate=8000,
        )
        first = engine._references([request, request])
        second = engine._references([request])
        self.assertEqual(engine.codec.calls, 1)
        self.assertIs(first[0], first[1])
        self.assertIs(first[0], second[0])

    def test_adaptive_best_of_n_retries_only_failed_requests(self):
        engine = NarTTS.__new__(NarTTS)
        engine.config = {"best_of_n": {"initial": 2, "maximum": 4}}
        engine.codec = SimpleNamespace(sampling_rate=10)
        calls = []
        engine._prepare = lambda requests: list(requests)

        def generate(indexed, number, candidate_offset=0):
            calls.append(([index for index, _ in indexed], number, candidate_offset))
            return [
                Candidate(
                    request_index=index,
                    candidate_index=candidate_offset + candidate,
                    token_ids=[],
                    waveform=np.zeros(10, dtype=np.float32),
                    valid=True,
                )
                for index, _ in indexed
                for candidate in range(number)
            ]

        def score(candidates, prepared):
            del prepared
            for candidate in candidates:
                candidate.accepted = (
                    candidate.request_index == 1 or candidate.candidate_index >= 2
                )
                candidate.score = float(candidate.accepted)

        engine._generate = generate
        engine._score = score
        engine._write_results = lambda prepared, candidates, winners, timings=None: (
            list(winners)
        )
        requests = [
            SynthesisRequest(
                text=f"text {index}",
                reference_audio=None,
                reference_text="reference",
                reference_waveform=np.zeros(10, dtype=np.float32),
                reference_sample_rate=10,
            )
            for index in range(2)
        ]
        winners = engine.synthesize_batch(requests)
        self.assertEqual(calls, [([0, 1], 2, 0), ([0], 2, 2)])
        self.assertTrue(all(winner.accepted for winner in winners))


class QualityDataTest(unittest.TestCase):
    def test_audit_keeps_metadata_and_rejects_duplicate_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wave = 0.1 * np.sin(
                2 * np.pi * 220 * np.arange(8000, dtype=np.float32) / 8000
            )
            sf.write(root / "voice.wav", wave, 8000)
            manifest = root / "input.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": index,
                            "audio": "voice.wav",
                            "text": text,
                            "speaker": "a",
                            "license": "owned",
                        }
                    )
                    for index, text in enumerate(("Merhaba", "Tekrar"))
                )
                + "\n",
                encoding="utf-8",
            )
            report = audit_manifest(
                manifest,
                root / "accepted.jsonl",
                root / "rejected.jsonl",
                thresholds=AuditThresholds(require_license=True),
            )
            self.assertEqual(report["accepted"], 1)
            self.assertEqual(report["reasons"]["duplicate_audio"], 1)
            accepted = json.loads((root / "accepted.jsonl").read_text().splitlines()[0])
            self.assertEqual(accepted["speaker"], "a")
            self.assertIn("quality", accepted)

    def test_distillation_accepts_only_verified_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "sample.json"
            report_path.write_text(
                json.dumps(
                    {
                        "id": "x",
                        "text": "Merhaba",
                        "winner": {
                            "audio_path": "winner.wav",
                            "score": 0.9,
                            "accepted": True,
                            "verification": {"cer": 0.0},
                        },
                        "hard_case": True,
                    }
                ),
                encoding="utf-8",
            )
            result = build_distillation_manifest(
                [report_path], root / "distilled.jsonl"
            )
            self.assertEqual(result["selected"], 1)
            row = json.loads((root / "distilled.jsonl").read_text())
            self.assertTrue(row["hard_case"])

    def test_distillation_rejects_a_high_scoring_failed_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "failed.json"
            report_path.write_text(
                json.dumps(
                    {
                        "winner": {
                            "audio_path": "failed.wav",
                            "score": 0.99,
                            "accepted": False,
                            "verification": {"cer": 0.0},
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = build_distillation_manifest(
                [report_path], root / "distilled.jsonl"
            )
            self.assertEqual(result["selected"], 0)


if __name__ == "__main__":
    unittest.main()
