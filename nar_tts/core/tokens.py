from dataclasses import dataclass

import numpy as np

# Mimi RVQ shape and the reserved-id offset. Changing any of these means
# re-encoding every dataset AND retraining — they are baked into the token ids.
NUM_CODEBOOKS = 32          # full Mimi depth; ~4x longer sequences than 8 codebooks
CODEBOOK_SIZE = 2048        # entries per codebook
AUDIO_OFFSET = 10           # first 10 ids above `base` are reserved for specials

# Special-token offsets relative to `base`.
SOS, EOS_SPEECH, SOH, EOH, SOA = 1, 2, 3, 4, 5


@dataclass(frozen=True)
class TokenLayout:
    """Orpheus token layout — the single source of truth for every stage.

    A speech clip is turned into discrete codes by the Mimi codec, and those
    codes are appended to the language model's vocabulary as extra tokens. The
    LM then learns to *continue* text with audio tokens, which is what makes it
    a TTS model. Everything is an offset from the base text vocab:

        base                = len(text tokenizer)             # e.g. Qwen3-0.6B
        <custom_token_i>    = base + i                         # the added tokens
        SOS / EOS_SPEECH    = base + 1 / base + 2              # start / end of speech
        SOH / EOH / SOA     = base + 3 / base + 4 / base + 5   # turn markers
        audio (cb, code)    = base + AUDIO_OFFSET + cb*CODEBOOK_SIZE + code

    Training-sequence layouts (`eot` is the tokenizer's own end-of-text id):

        TTS : [SOH] text [eot] [EOH] [SOA] [SOS] <audio tokens> [EOS_SPEECH]
        QA  : [SOH] question [eot] [EOH] [SOA] answer [eot]

    Binding the scheme to a tokenizer's vocab size keeps encoding and decoding
    from ever drifting apart.
    """

    base: int                       # len(tokenizer)
    eot: int                        # tokenizer.eos_token_id
    num_codebooks: int = NUM_CODEBOOKS
    codebook_size: int = CODEBOOK_SIZE

    @classmethod
    def from_tokenizer(cls, tokenizer, num_codebooks=NUM_CODEBOOKS):
        return cls(base=len(tokenizer), eot=tokenizer.eos_token_id,
                   num_codebooks=num_codebooks)

    # --- special tokens ---------------------------------------------------
    @property
    def sos(self):        return self.base + SOS
    @property
    def eos_speech(self): return self.base + EOS_SPEECH
    @property
    def soh(self):        return self.base + SOH
    @property
    def eoh(self):        return self.base + EOH
    @property
    def soa(self):        return self.base + SOA

    # --- vocabulary expansion --------------------------------------------
    @property
    def num_added_tokens(self):
        """Count of <custom_token_i> ids (all codes + the reserved specials)."""
        return self.num_codebooks * self.codebook_size + AUDIO_OFFSET

    def added_token_strings(self):
        """Placeholder strings to register with `tokenizer.add_tokens`."""
        return [f"<custom_token_{i}>" for i in range(self.num_added_tokens + 1)]

    # --- audio codes <-> LM token ids ------------------------------------
    def codes_to_ids(self, codes):
        """Mimi codes (num_codebooks, T) -> flat per-frame-interleaved LM ids."""
        codes = np.asarray(codes)
        offsets = (self.base + AUDIO_OFFSET
                   + np.arange(self.num_codebooks, dtype=np.int64) * self.codebook_size)
        ids = codes.astype(np.int64, copy=False) + offsets[:, None]
        return ids.T.reshape(-1).tolist()               # frame-major order

    def codes_batch_to_ids(self, codes, n_frames=None):
        """Batched ``codes_to_ids`` for a padded ``(B, nq, T)`` batch."""
        codes = np.asarray(codes)
        if codes.ndim != 3 or codes.shape[1] != self.num_codebooks:
            raise ValueError(
                f"expected codes with shape (batch, {self.num_codebooks}, frames), "
                f"got {codes.shape!r}")
        if n_frames is None:
            n_frames = [codes.shape[-1]] * codes.shape[0]
        if len(n_frames) != codes.shape[0]:
            raise ValueError("n_frames must have one entry per code batch")

        offsets = (self.base + AUDIO_OFFSET
                   + np.arange(self.num_codebooks, dtype=np.int64) * self.codebook_size)
        limit = codes.shape[-1]
        return [np.add(
                    codes[i, :, :max(0, min(int(frames), limit))], offsets[:, None]
                ).T.reshape(-1).tolist()
                for i, frames in enumerate(n_frames)]

    def ids_to_codes(self, ids):
        """Generated LM ids -> Mimi codes (num_codebooks, T); stop at 1st invalid.

        Returns None if not even one full frame of valid codes is present.
        """
        valid = []
        for i, tid in enumerate(ids):
            cb = i % self.num_codebooks
            code = tid - self.base - AUDIO_OFFSET - cb * self.codebook_size
            if 0 <= code < self.codebook_size:
                valid.append(code)
            else:
                break
        n_frames = len(valid) // self.num_codebooks
        if n_frames == 0:
            return None
        arr = np.asarray(valid[: n_frames * self.num_codebooks], dtype=np.int64)
        return arr.reshape(n_frames, self.num_codebooks).T      # (num_codebooks, T)

    # --- training-sequence builders --------------------------------------
    def tts_sequence(self, text_ids, codes):
        """[SOH] text [eot] [EOH] [SOA] [SOS] <audio> [EOS_SPEECH]."""
        return ([self.soh] + list(text_ids)
                + [self.eot, self.eoh, self.soa, self.sos]
                + self.codes_to_ids(codes)
                + [self.eos_speech])

    def tts_sequences(self, text_ids, codes, n_frames=None):
        """Build a batch of TTS rows while vectorizing audio-id conversion."""
        text_ids = list(text_ids)
        audio_ids = self.codes_batch_to_ids(codes, n_frames=n_frames)
        if len(text_ids) != len(audio_ids):
            raise ValueError("text_ids and codes must have the same batch size")
        middle = [self.eot, self.eoh, self.soa, self.sos]
        return [([self.soh] + list(text) + middle + audio + [self.eos_speech])
                for text, audio in zip(text_ids, audio_ids)]

    def pack_tts_sequences(self, text_ids, codes, n_frames=None):
        """Build TTS rows as Arrow-ready offsets and contiguous int32 values.

        Unlike ``tts_sequences``, this hot-path method never creates Python ints
        for the (much larger) audio portion of a row. ``offsets`` and ``values``
        can be passed directly to ``pyarrow.ListArray.from_arrays``.
        """
        text_ids = list(text_ids)
        codes = np.asarray(codes)
        if codes.ndim != 3 or codes.shape[1] != self.num_codebooks:
            raise ValueError(
                f"expected codes with shape (batch, {self.num_codebooks}, frames), "
                f"got {codes.shape!r}")
        if len(text_ids) != codes.shape[0]:
            raise ValueError("text_ids and codes must have the same batch size")
        if n_frames is None:
            n_frames = [codes.shape[-1]] * codes.shape[0]
        if len(n_frames) != codes.shape[0]:
            raise ValueError("n_frames must have one entry per code batch")

        frame_counts = np.clip(
            np.asarray(n_frames, dtype=np.int64), 0, codes.shape[-1])
        text_lengths = np.fromiter(
            (len(text) for text in text_ids), dtype=np.int64,
            count=len(text_ids))
        # SOH + text + four turn/speech markers + audio + EOS_SPEECH.
        row_lengths = text_lengths + frame_counts * self.num_codebooks + 6
        offsets = np.empty(len(text_ids) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(row_lengths, out=offsets[1:])
        values = np.empty(int(offsets[-1]), dtype=np.int32)
        middle = np.asarray(
            [self.eot, self.eoh, self.soa, self.sos], dtype=np.int32)
        audio_offsets = np.asarray(
            self.base + AUDIO_OFFSET
            + np.arange(self.num_codebooks, dtype=np.int64) * self.codebook_size,
            dtype=np.int32)

        for i, (text, frames) in enumerate(zip(text_ids, frame_counts)):
            position = int(offsets[i])
            values[position] = self.soh
            position += 1
            text_length = len(text)
            values[position:position + text_length] = text
            position += text_length
            values[position:position + 4] = middle
            position += 4
            frames = int(frames)
            audio_length = frames * self.num_codebooks
            audio = values[position:position + audio_length].reshape(
                frames, self.num_codebooks)
            np.add(
                codes[i, :, :frames].T, audio_offsets,
                out=audio, casting="unsafe")
            position += audio_length
            values[position] = self.eos_speech
        return offsets, values

    def qa_sequence(self, question_ids, answer_ids):
        """[SOH] question [eot] [EOH] [SOA] answer [eot]."""
        return ([self.soh] + list(question_ids)
                + [self.eot, self.eoh, self.soa]
                + list(answer_ids) + [self.eot])

    def qa_sequences(self, question_ids, answer_ids):
        """Build QA rows from two already batch-tokenized columns."""
        question_ids, answer_ids = list(question_ids), list(answer_ids)
        if len(question_ids) != len(answer_ids):
            raise ValueError("question_ids and answer_ids must have the same batch size")
        middle = [self.eot, self.eoh, self.soa]
        return [([self.soh] + list(question) + middle + list(answer) + [self.eot])
                for question, answer in zip(question_ids, answer_ids)]

    def pack_qa_sequences(self, question_ids, answer_ids):
        """Build QA rows as Arrow-ready offsets and contiguous int32 values."""
        question_ids, answer_ids = list(question_ids), list(answer_ids)
        if len(question_ids) != len(answer_ids):
            raise ValueError("question_ids and answer_ids must have the same batch size")
        row_lengths = np.fromiter(
            (len(question) + len(answer) + 5
             for question, answer in zip(question_ids, answer_ids)),
            dtype=np.int64, count=len(question_ids))
        offsets = np.empty(len(question_ids) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(row_lengths, out=offsets[1:])
        values = np.empty(int(offsets[-1]), dtype=np.int32)
        middle = np.asarray([self.eot, self.eoh, self.soa], dtype=np.int32)

        for i, (question, answer) in enumerate(zip(question_ids, answer_ids)):
            position = int(offsets[i])
            values[position] = self.soh
            position += 1
            question_length = len(question)
            values[position:position + question_length] = question
            position += question_length
            values[position:position + 3] = middle
            position += 3
            answer_length = len(answer)
            values[position:position + answer_length] = answer
            position += answer_length
            values[position] = self.eot
        return offsets, values
