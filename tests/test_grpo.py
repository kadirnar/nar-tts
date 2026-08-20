import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from nar_tts.core.generation import (
    AudioTokenLogitsProcessor,
    audio_token_bounds,
    constrain_audio_logits,
    constrained_audio_log_probs,
    parse_audio_completion,
)
from nar_tts.core.model_ids import QWEN3_ASR_MODEL_ID, WAVLM_SPEAKER_MODEL_ID
from nar_tts.core.tokens import TokenLayout
from nar_tts.integrations.sglang import NarAudioSGLangLogitsProcessor
from nar_tts.integrations.vllm import (
    GRAMMAR_ARGUMENT,
    NarAudioRequestLogitsProcessor,
    audio_grammar_arguments,
)
from nar_tts.training.grpo import _training_args
from nar_tts.training.grpo_config import (
    GRPOConfigError,
    load_grpo_config,
    validate_grpo_config,
)
from nar_tts.training.grpo_data import (
    prepare_grpo_example,
    split_tts_training_sequence,
)
from nar_tts.training.grpo_rewards import (
    SpeechRewardSuite,
    _canonical_asr_language,
    duration_consistency_reward,
    error_rate_reward,
    nll_reward,
    transcript_error_rate,
    weighted_harmonic_mean,
)


class FakeEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    eos_token_id = 12
    pad_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return FakeEncoding([ord(character) for character in text])

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in ids)


class FakeSavedNarTokenizer:
    eos_token_id = 12
    pad_token_id = 2

    def __len__(self):
        return 30

    def get_vocab(self):
        return {"<custom_token_0>": 10}


class GenerationConstraintTest(unittest.TestCase):
    def setUp(self):
        self.layout = TokenLayout(base=10, eot=2, num_codebooks=2, codebook_size=4)

    def test_completion_parser_requires_full_frames_and_eos(self):
        codes = np.array([[0, 1], [2, 3]])
        audio_ids = self.layout.codes_to_ids(codes)
        parsed = parse_audio_completion(
            [*audio_ids, self.layout.eos_speech], self.layout
        )
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.num_frames, 2)
        np.testing.assert_array_equal(parsed.codes, codes)

        self.assertFalse(parse_audio_completion(audio_ids, self.layout).valid)
        malformed = [*audio_ids]
        malformed[1] = audio_token_bounds(self.layout, 0)[0]
        self.assertFalse(
            parse_audio_completion(
                [*malformed, self.layout.eos_speech], self.layout
            ).valid
        )

    def test_layout_recovers_base_and_text_eot_from_saved_grpo_tokenizer(self):
        layout = TokenLayout.from_tokenizer(FakeSavedNarTokenizer())
        self.assertEqual(layout.base, 10)
        self.assertEqual(layout.eot, 2)

    def test_logits_processor_follows_codebooks_and_frame_boundaries(self):
        processor = AudioTokenLogitsProcessor(
            self.layout, prompt_length=3, min_frames=1, max_frames=2
        )
        scores = torch.arange(30, dtype=torch.float32).repeat(2, 1)

        first = processor(torch.ones((2, 3), dtype=torch.long), scores)
        lower, upper = audio_token_bounds(self.layout, 0)
        self.assertTrue(torch.isfinite(first[:, lower:upper]).all())
        self.assertTrue(torch.isneginf(first[:, self.layout.eos_speech]).all())

        second = processor(torch.ones((2, 4), dtype=torch.long), scores)
        lower, upper = audio_token_bounds(self.layout, 1)
        self.assertTrue(torch.isfinite(second[:, lower:upper]).all())

        boundary = processor(torch.ones((2, 5), dtype=torch.long), scores)
        self.assertTrue(torch.isfinite(boundary[:, self.layout.eos_speech]).all())

        forced = processor(torch.ones((2, 7), dtype=torch.long), scores)
        self.assertEqual(torch.isfinite(forced).sum(dim=-1).tolist(), [1, 1])
        self.assertTrue(torch.isfinite(forced[:, self.layout.eos_speech]).all())

    def test_backend_processors_share_the_exact_audio_grammar(self):
        scores = torch.arange(30, dtype=torch.float32)
        expected = constrain_audio_logits(
            scores,
            self.layout,
            generated_tokens=2,
            min_frames=1,
            max_frames=2,
        )
        request_processor = NarAudioRequestLogitsProcessor(
            self.layout, min_frames=1, max_frames=2
        )
        actual = request_processor([10, 11], scores.clone())
        torch.testing.assert_close(actual, expected)

        class Request:
            def __init__(self):
                self.output_ids = [10, 11]

        grammar = audio_grammar_arguments(self.layout, 1, 2)
        sglang_scores = scores.repeat(1, 1)
        NarAudioSGLangLogitsProcessor()(sglang_scores, [
            {GRAMMAR_ARGUMENT: grammar, "__req__": Request()}
        ])
        torch.testing.assert_close(sglang_scores[0], expected)

    def test_constrained_log_probs_match_small_uniform_action_spaces(self):
        vocab_size = audio_token_bounds(self.layout, 1)[1] + 1
        logits = torch.zeros((1, 4, vocab_size), requires_grad=True)
        first_cb = audio_token_bounds(self.layout, 0)[0]
        second_cb = audio_token_bounds(self.layout, 1)[0]
        selected = torch.tensor(
            [[first_cb, second_cb, self.layout.eos_speech, self.layout.eot]]
        )
        log_probs, entropies = constrained_audio_log_probs(
            logits,
            selected,
            self.layout,
            min_frames=1,
            max_frames=2,
            pad_token_id=self.layout.eot,
            compute_entropy=True,
        )
        expected_log_probs = torch.tensor(
            [[-math.log(4), -math.log(4), -math.log(5), 0.0]]
        )
        expected_entropies = torch.tensor(
            [[math.log(4), math.log(4), math.log(5), 0.0]]
        )
        torch.testing.assert_close(log_probs, expected_log_probs)
        torch.testing.assert_close(entropies, expected_entropies)
        (-log_probs[:, :3].mean()).backward()
        self.assertIsNotNone(logits.grad)


class GRPODataTest(unittest.TestCase):
    def test_split_pretraining_row_without_raw_audio(self):
        layout = TokenLayout(base=200, eot=2, num_codebooks=2, codebook_size=8)
        tokenizer = FakeTokenizer()
        text = "Hi"
        codes = np.array([[1, 2, 3], [4, 5, 6]])
        row = layout.tts_sequence([ord(char) for char in text], codes)
        example = split_tts_training_sequence(row, tokenizer, layout, frame_rate=10.0)
        self.assertEqual(example.target_text, text)
        self.assertEqual(example.prompt[-1], layout.sos)
        self.assertEqual(example.reference_audio_ids, layout.codes_to_ids(codes))
        self.assertEqual(example.target_duration_seconds, 0.3)

    def test_text_and_voice_clone_prompt_modes(self):
        layout = TokenLayout(base=200, eot=2, num_codebooks=2, codebook_size=8)
        tokenizer = FakeTokenizer()
        text_example = prepare_grpo_example(
            {"text": "Hello"},
            tokenizer,
            layout,
            {"mode": "text", "columns": {"text": "text"}},
        )
        self.assertEqual(text_example.prompt[-1], layout.sos)

        reference = layout.codes_to_ids(np.array([[1, 2], [3, 4]]))
        clone_example = prepare_grpo_example(
            {"text": "world", "ref_text": "hello", "ref_ids": reference},
            tokenizer,
            layout,
            {
                "mode": "voice_clone_tokens",
                "columns": {
                    "text": "text",
                    "reference_text": "ref_text",
                    "reference_audio_ids": "ref_ids",
                },
            },
        )
        self.assertEqual(clone_example.prompt[-len(reference) :], reference)
        self.assertEqual(clone_example.reference_text, "hello")


class RewardMathTest(unittest.TestCase):
    def test_qwen3_asr_language_resolution(self):
        self.assertEqual(_canonical_asr_language("en"), "English")
        self.assertEqual(_canonical_asr_language("japanese"), "Japanese")
        self.assertIsNone(_canonical_asr_language(""))
        with self.assertRaises(ValueError):
            _canonical_asr_language("not-a-language")

    def test_multilingual_error_metrics_and_reward_formulas(self):
        self.assertEqual(transcript_error_rate("Hello!", "hello", "wer"), 0.0)
        self.assertEqual(transcript_error_rate("猫です。", "猫で", "cer"), 1 / 3)
        self.assertAlmostEqual(error_rate_reward(0.0), 1.0)
        self.assertAlmostEqual(nll_reward(3.0, alpha=3.0), math.exp(-1))
        self.assertAlmostEqual(weighted_harmonic_mean([0.5, 1.0], [1.0, 1.0]), 2 / 3)

    def test_duration_rewards(self):
        self.assertEqual(duration_consistency_reward(1.0, 1.1), 1.0)
        self.assertEqual(duration_consistency_reward(2.0, 1.0), 0.0)
        self.assertAlmostEqual(
            duration_consistency_reward(2.0, 1.0, mode="smooth_log", scale=1.0),
            0.5,
        )

    def test_qwen3_asr_nll_masks_the_audio_and_language_prefix(self):
        layout = TokenLayout(base=10, eot=2, num_codebooks=2, codebook_size=4)
        suite = SpeechRewardSuite(
            layout,
            {"asr": {"nll_reduction": "mean"}},
        )

        class Processor:
            def apply_transcription_request(self, audio, language):
                del audio, language
                return {
                    "input_ids": torch.tensor([[0, 1, 2]]),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                }

            def apply_chat_template(self, *args, **kwargs):
                del args, kwargs
                ids = torch.tensor([[0, 1, 2, 3, 4]])
                return {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "labels": ids.clone(),
                }

        class Model:
            def __call__(self, **kwargs):
                del kwargs
                return SimpleNamespace(logits=torch.zeros((1, 5, 8)))

        suite.asr_processor = Processor()
        suite.asr_model = Model()
        values = suite._qwen3_asr_nll(
            [np.zeros(16, dtype=np.float32)],
            ["test"],
            ["English"],
            torch.device("cpu"),
            torch.float32,
        )
        self.assertAlmostEqual(values[0], math.log(8), places=6)

    def test_fused_reward_combines_components_without_redecoding(self):
        layout = TokenLayout(base=10, eot=2, num_codebooks=2, codebook_size=4)
        codes = np.array([[0, 1], [2, 3]])
        completion = [*layout.codes_to_ids(codes), layout.eos_speech]

        class FakeCodec:
            sampling_rate = 24_000

            def __init__(self):
                self.calls = 0

            def decode_batch(self, code_batches):
                self.calls += 1
                return [
                    np.zeros(item.shape[-1] * 10, dtype=np.float32)
                    for item in code_batches
                ]

        suite = SpeechRewardSuite(
            layout,
            {
                "weights": {"intelligibility": 0.75, "speaker": 0.25},
                "speaker": {"enabled": True},
            },
        )
        suite.codec = FakeCodec()
        suite._intelligibility_scores = lambda waves, texts, languages: (
            [0.5],
            ["test"],
            [0.1],
            [1.0],
        )
        suite._speaker_scores = lambda waves, reference: [1.0]
        reward = suite(
            completion_ids=[completion],
            target_text=["test"],
            reference_audio_ids=[layout.codes_to_ids(codes)],
        )
        self.assertEqual(suite.codec.calls, 1)
        self.assertAlmostEqual(reward[0], 0.625)


class ScenarioConfigTest(unittest.TestCase):
    def test_all_grpo_scenarios_validate(self):
        config_dir = Path(__file__).resolve().parents[1] / "nar_tts" / "configs"
        expected_world_sizes = {
            "grpo_intelligibility.yaml": 1,
            "grpo_multireward.yaml": 8,
            "grpo_sglang.yaml": 1,
            "grpo_style_fast.yaml": 1,
            "grpo_style_slow.yaml": 1,
            "grpo_text_streaming.yaml": 1,
            "grpo_vllm.yaml": 1,
        }
        for filename, world_size in expected_world_sizes.items():
            with self.subTest(filename=filename):
                config = load_grpo_config(config_dir / filename)
                derived = validate_grpo_config(config, world_size=world_size)
                self.assertGreater(derived["max_completion_length"], 1)
                args = _training_args(
                    config,
                    derived,
                    FakeTokenizer(),
                    world_size=world_size,
                    layout=TokenLayout(base=10, eot=2),
                )
                self.assertAlmostEqual(args.warmup_steps, 0.03)
                self.assertEqual(args.get_warmup_steps(1000), 30)

    def test_quality_configs_never_select_a_small_asr_or_wavlm_model(self):
        config_dir = Path(__file__).resolve().parents[1] / "nar_tts" / "configs"
        for path in config_dir.glob("grpo_*.yaml"):
            config = load_grpo_config(path)
            asr = config["rewards"].get("asr", {})
            if asr.get("enabled", True):
                self.assertEqual(asr.get("model"), QWEN3_ASR_MODEL_ID)
            if config["rewards"]["weights"].get("speaker", 0) > 0:
                speaker = config["rewards"]["speaker"]
                self.assertEqual(speaker.get("backend"), "espnet")
                self.assertEqual(speaker.get("model"), WAVLM_SPEAKER_MODEL_ID)

    def test_small_asr_checkpoint_is_rejected(self):
        config_dir = Path(__file__).resolve().parents[1] / "nar_tts" / "configs"
        config = load_grpo_config(config_dir / "grpo_intelligibility.yaml")
        config["rewards"]["asr"]["model"] = "Qwen/Qwen3-ASR-0.6B-hf"
        with self.assertRaisesRegex(GRPOConfigError, "Qwen3-ASR-1.7B"):
            validate_grpo_config(config, world_size=1)

    def test_unsloth_is_kept_out_of_the_incompatible_grpo_environment(self):
        config_dir = Path(__file__).resolve().parents[1] / "nar_tts" / "configs"
        config = load_grpo_config(config_dir / "grpo_intelligibility.yaml")
        config["model"]["loader"] = "unsloth"
        with self.assertRaisesRegex(GRPOConfigError, "finetune_unsloth"):
            validate_grpo_config(config, world_size=1)

    def test_invalid_group_size_fails_before_training(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "nar_tts"
            / "configs"
            / "grpo_multireward.yaml"
        )
        config = load_grpo_config(config_path)
        with self.assertRaises(GRPOConfigError):
            validate_grpo_config(config, world_size=1)


if __name__ == "__main__":
    unittest.main()
