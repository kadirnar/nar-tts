import torch
import wandb
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from transformers import Trainer

from nar_tts.core.data import AlternatingDistributedSampler


class FSDPTrainer(Trainer):
    """Trainer that saves a single consolidated checkpoint under FSDP.

    FSDP shards the weights across GPUs; this gathers them to rank-0 on CPU and
    writes one ordinary `save_pretrained` checkpoint that inference can load.
    """
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        if not isinstance(self.model, FSDP):
            return super().save_model(output_dir, _internal_call=_internal_call)
        policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, policy):
            cpu_state_dict = self.model.state_dict()
        if self.args.should_save:
            self.model.save_pretrained(output_dir, state_dict=cpu_state_dict)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)


class RatioTrainer(FSDPTrainer):
    """FSDPTrainer for the decaying text/speech pretraining recipe.

    Feeds the live training step into GradualRatioDataset, uses the
    interleave-preserving sampler, and logs separate text/speech losses.
    """
    def __init__(self, *args, initial_ratio, final_ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.text_step = 0
        self.audio_step = 0
        self.total_steps = self._total_steps()
        if hasattr(self.train_dataset, "total_steps"):
            self.train_dataset.total_steps = self.total_steps

    def _total_steps(self):
        per_epoch = len(self.train_dataset) // (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps * self.args.world_size)
        return int(per_epoch * self.args.num_train_epochs)

    def current_ratio(self):
        if not self.total_steps:
            return self.initial_ratio
        progress = min(self.state.global_step / self.total_steps, 1.0)
        ratio = self.initial_ratio - (self.initial_ratio - self.final_ratio) * progress
        return max(round(ratio), self.final_ratio)

    def get_train_dataloader(self):
        sampler = AlternatingDistributedSampler(
            self.train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank())
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler, collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0, pin_memory=self.args.dataloader_pin_memory)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if hasattr(self.train_dataset, "set_current_step"):
            self.train_dataset.set_current_step(self.state.global_step)
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        if not (self.is_world_process_zero() and "loss" in logs):
            return
        ratio = self.current_ratio()
        wandb.log({"current_ratio": ratio, "global_step": self.state.global_step})
        # Within each (ratio + 1)-step cycle the first `ratio` steps are text.
        if self.state.global_step % (ratio + 1) < ratio:
            wandb.log({"text_loss": logs["loss"], "text_step": self.text_step})
            self.text_step += 1
        else:
            wandb.log({"audio_loss": logs["loss"], "audio_step": self.audio_step})
            self.audio_step += 1
