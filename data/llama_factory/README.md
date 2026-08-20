# LLaMA-Factory dataset slot

Place a UTF-8 `language_retention.jsonl` file in this directory. Each line uses
the standard Alpaca fields:

```json
{"instruction":"Answer the question.","input":"...","output":"..."}
```

The checked-in dataset registry and post-pretraining config refer to this file.
Keep speech-codec SFT and GRPO in Nar's native trainers; this LLaMA-Factory stage
is for retaining or adapting the text/language capabilities of a pretrained Nar
checkpoint.
