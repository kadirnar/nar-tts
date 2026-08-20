import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


def make_collator(pad_token):
    """Build a collate_fn that pads variable-length `input_ids` rows to tensors.

    Missing attention_mask -> all ones. Missing labels -> copy of input_ids
    (causal LM). Labels pad with -100 so padded positions are ignored by the loss.
    """
    def pad(seqs, value):
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in seqs],
            batch_first=True, padding_value=value)

    def collate(features):
        input_ids = [f["input_ids"] for f in features]
        if any("attention_mask" not in f for f in features):
            attention_mask = [[1] * len(ids) for ids in input_ids]
        else:
            attention_mask = [f["attention_mask"] for f in features]
        labels = (input_ids if any("labels" not in f for f in features)
                  else [f["labels"] for f in features])
        return {"input_ids": pad(input_ids, pad_token),
                "attention_mask": pad(attention_mask, 0),
                "labels": pad(labels, -100)}

    return collate


class AlternatingDistributedSampler(DistributedSampler):
    """Strided, non-shuffled sampler that preserves the text/speech interleave.

    GradualRatioDataset lays samples out in a deliberate text-then-speech
    pattern; shuffling would destroy it, so each rank just strides its slice.
    """
    def __init__(self, dataset, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        return iter(indices[self.rank:self.total_size:self.num_replicas])


class GradualRatioDataset(Dataset):
    """Interleaves a text dataset and a speech dataset at a *decaying* ratio.

    Early training mixes `initial_ratio` text batches per speech batch; by the
    end it reaches `final_ratio`. This keeps the LM's text ability alive while
    it learns speech, then shifts focus to speech. The ratio is recomputed live
    from the training step (set each step via `set_current_step`).

    Example ("1:0"): start 1 text : 1 speech (50/50), end 0 text : pure speech.
    """
    def __init__(self, text_ds, speech_ds, batch_total,
                 initial_ratio, final_ratio, total_steps):
        self.text_ds = text_ds
        self.speech_ds = speech_ds
        self.batch_total = batch_total
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.total_steps = total_steps
        self.current_step = 0

        # Size to the most data the largest ratio could need, so we never run dry.
        max_ratio = max(initial_ratio, final_ratio)
        num_cycles = min(len(text_ds) // (batch_total * max_ratio),
                         len(speech_ds) // batch_total)
        self.length = num_cycles * (initial_ratio + 1) * batch_total

    def set_current_step(self, step):
        self.current_step = step

    def current_ratio(self):
        """Linearly interpolate from initial_ratio to final_ratio by progress."""
        if not self.total_steps:
            return self.initial_ratio
        progress = min(self.current_step / self.total_steps, 1.0)
        ratio = self.initial_ratio - (self.initial_ratio - self.final_ratio) * progress
        return max(round(ratio), self.final_ratio)

    def __len__(self):
        return int(self.length)

    def __getitem__(self, index):
        ratio = self.current_ratio()
        cycle_length = (ratio + 1) * self.batch_total
        cycle = index // cycle_length
        pos = index % cycle_length

        if pos < ratio * self.batch_total:              # text slot
            idx = cycle * ratio * self.batch_total + pos
            return self.text_ds[idx % len(self.text_ds)]
        # speech slot
        idx = cycle * self.batch_total + (pos - ratio * self.batch_total)
        return self.speech_ds[idx % len(self.speech_ds)]
