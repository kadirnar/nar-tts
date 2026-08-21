# Emotion and non-verbal sound support

## Controls

Nar uses three independent controls:

- Emotion: `neutral`, `joy`, `sadness`, `anger`, `fear`, `surprise`
- Delivery: `neutral`, `crying_speech`, `speech_laugh`
- Event: `laugh`, `chuckle`, `sob`, `cry`, `sniff`, `sigh`, `gasp`, `breath`

`speech_laugh` means that the text is spoken with laughter, while `laugh`
represents a separate laughter event. Likewise, `crying_speech` and `sob` are
annotated separately.

## Example

```json
{
  "text": "I thought of you today.",
  "emotion": "sadness",
  "intensity": 0.9,
  "delivery": "crying_speech",
  "events": [{"type": "sob", "after_word": 1, "duration": "short"}]
}
```

The controls are rendered as text markup that does not modify the tokenizer:

```text
<nar_control emotion=sadness intensity=0.900 delivery=crying_speech>
I <nar_event type=sob after_word=1 duration=short count=1> thought of you today.
```

## Training

1. Check the Mimi round trip with `nar-tts codec-check`.
2. Use `nar-tts audit-data` to separate corrupt, duplicate, or improperly
   licensed recordings.
3. Split neutral and expressive data without speaker or text leakage.
4. Create the training Parquet file with `nar-tts encode-expressive`.
5. Include both neutral and expressive recordings in SFT.
6. Enable emotion and event GRPO weights only after the fixed evaluation set
   passes its acceptance criteria.

Store emotion, intensity, delivery, event type, position, duration, speaker,
language, source, and license metadata with each sample. Recordings of the same
text by the same speaker at different emotion intensities make the control
signal easier to learn.

## Evaluation

- WER/CER after removing control markup
- Emotion and intensity accuracy
- F1 for event type, count, and position
- Speaker similarity and speaker drift
- Neutral-quality regression
- Blinded human A/B testing

The current checkpoint has not learned this markup, so it cannot produce crying
speech or laughter on its own. These capabilities require an expressive SFT
checkpoint.
