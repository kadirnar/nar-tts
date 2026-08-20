import tempfile
import unittest

from datasets import Dataset
from peft import LoraConfig, TaskType
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import GRPOConfig

from nar_tts.core.tokens import TokenLayout
from nar_tts.training.grpo_trainer import NarGRPOTrainer


class TinyGRPOIntegrationTest(unittest.TestCase):
    def test_one_cpu_step_uses_direct_prompts_and_constrained_policy(self):
        layout = TokenLayout(base=10, eot=2, num_codebooks=2, codebook_size=4)
        vocabulary = {f"t{index}": index for index in range(30)}
        backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="t0"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="t0",
            pad_token="t2",
            eos_token="t12",
        )
        tokenizer.padding_side = "left"
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=30,
                n_layer=1,
                n_head=1,
                n_embd=16,
                n_positions=32,
                bos_token_id=None,
                eos_token_id=layout.eos_speech,
                pad_token_id=layout.eot,
            )
        )
        dataset = Dataset.from_dict(
            {
                "prompt": [[3, 4], [3, 5]],
                "target_text": ["a", "b"],
                "reference_audio_ids": [[], []],
                "reference_text": ["", ""],
                "target_duration_seconds": [-1.0, -1.0],
                "language": ["", ""],
            }
        )

        with tempfile.TemporaryDirectory() as output_dir:
            args = GRPOConfig(
                output_dir=output_dir,
                max_steps=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                num_generations=2,
                max_completion_length=5,
                use_cpu=True,
                bf16=False,
                fp16=False,
                report_to="none",
                gradient_checkpointing=False,
                logging_steps=1,
                save_strategy="no",
                remove_unused_columns=False,
                temperature=1.0,
                beta=0.1,
                loss_type="dapo",
                mask_truncated_completions=True,
                disable_tqdm=True,
            )

            def reward(completion_ids, **kwargs):
                del kwargs
                return [float(ids[0]) for ids in completion_ids]

            trainer = NarGRPOTrainer(
                model=model,
                args=args,
                reward_funcs=reward,
                train_dataset=dataset,
                processing_class=tokenizer,
                peft_config=LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=2,
                    lora_alpha=4,
                    target_modules=["c_attn"],
                    fan_in_fan_out=True,
                ),
                token_layout=layout,
                min_audio_frames=1,
                max_audio_frames=2,
                constrain_log_probs=True,
            )
            result = trainer.train()
        self.assertGreaterEqual(result.training_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
