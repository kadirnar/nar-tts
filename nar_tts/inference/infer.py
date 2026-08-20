import os

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nar_tts.core.audio import MimiCodec
from nar_tts.core.tokens import TokenLayout

# Config (hardcoded; no CLI args). Single-GPU, multilingual voice cloning: given a
# reference clip (+ its transcript) and a target text in ANY language, the model
# speaks the target in the reference voice by pure continuation:
#   prompt = [SOH] ref_text+target_text [eot] [EOH] [SOA] [SOS] <ref audio codes>
#   model continues -> <target audio codes> [EOS_SPEECH]
# Run:  python -m nar_tts.inference.infer
CKPT = "checkpoints/checkpoint-171622"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "infer_out"
DEVICE = "cuda:0"

# Generation
MAX_NEW_TOKENS = 9600
MIN_NEW_TOKENS = 960            # floor avoids early EOS truncating the clip
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Jobs: (out_name, ref_wav, ref_text, target_text). Any language works.
REF_EN = "andrew_00.wav"
REF_EN_TEXT = "all sorts of things, but mainly just browsing FaceTime and streaming most of all."
REF_JA = "irodori-tts-v2-demo-2.wav"
REF_JA_TEXT = "ごめん、少し遅れてしまった。終電電車がとても混んでいたんだ。へいでも、ちゃんと来られてよかった。"

JOBS = [
    ("en_1", REF_EN, REF_EN_TEXT, "Hello, this is a test of the text to speech model."),
    ("en_2", REF_EN, REF_EN_TEXT, "Artificial intelligence is transforming the way we live and work."),
    ("ja_1", REF_JA, REF_JA_TEXT, "こんにちは、これは音声合成モデルのテストです。"),
    ("ja_2", REF_JA, REF_JA_TEXT, "桜の花が満開で、とても美しい春の一日でした。"),
]


class NarTTS:
    """Voice-cloning TTS engine: continues a text prompt into Mimi speech tokens."""

    def __init__(self, checkpoint=CKPT, tokenizer_name=TOKENIZER_NAME, device=DEVICE):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.layout = TokenLayout.from_tokenizer(self.tok)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16).to(device).eval()
        self.codec = MimiCodec(device, num_codebooks=self.layout.num_codebooks)

    @torch.no_grad()
    def clone(self, text, ref_wav, ref_text, output_path=None):
        """Speak `text` in the voice of `ref_wav` (primed by its transcript `ref_text`).

        Returns the mono float32 waveform (None if no valid audio), and writes it
        to `output_path` when given.
        """
        layout, tok, codec = self.layout, self.tok, self.codec
        ref_ids = layout.codes_to_ids(codec.encode_file(ref_wav))
        text_ids = tok((ref_text + " " + text).strip(), add_special_tokens=False).input_ids
        prompt = [layout.soh] + text_ids + [layout.eot, layout.eoh, layout.soa, layout.sos] + ref_ids
        inp = torch.tensor([prompt], device=self.device)
        out = self.model.generate(
            inp, attention_mask=torch.ones_like(inp),
            max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS,
            do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=layout.eos_speech, pad_token_id=tok.eos_token_id)
        codes = layout.ids_to_codes(out[0, inp.shape[1]:].tolist())
        if codes is None:
            return None
        wav = codec.decode(codes)
        if output_path:
            sf.write(output_path, wav, samplerate=codec.sampling_rate)
        return wav


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.cuda.set_device(DEVICE)
    engine = NarTTS()
    for name, ref_wav, ref_text, target in JOBS:
        out_path = os.path.join(OUTPUT_DIR, name + ".wav")
        wav = engine.clone(target, ref_wav, ref_text, output_path=out_path)
        if wav is None:
            print(f"{name}: FAILED (no valid audio)", flush=True)
        else:
            print(f"{name}: {wav.shape[0] / engine.codec.sampling_rate:.2f}s -> {out_path}",
                  flush=True)


if __name__ == "__main__":
    main()
