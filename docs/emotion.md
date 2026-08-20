# Emotion ve non-verbal ses desteği

## Kontroller

Nar üç ayrı kontrol kullanır:

- Emotion: `neutral`, `joy`, `sadness`, `anger`, `fear`, `surprise`
- Söyleyiş: `neutral`, `crying_speech`, `speech_laugh`
- Olay: `laugh`, `chuckle`, `sob`, `cry`, `sniff`, `sigh`, `gasp`, `breath`

`speech_laugh` metnin kahkahalı okunmasını, `laugh` ise ayrı bir kahkaha
olayını belirtir. `crying_speech` ve `sob` da ayrı etiketlenir.

## Örnek

```json
{
  "text": "Bugün seni düşündüm.",
  "emotion": "sadness",
  "intensity": 0.9,
  "delivery": "crying_speech",
  "events": [{"type": "sob", "after_word": 1, "duration": "short"}]
}
```

Kontroller tokenizer değiştirmeyen metin işaretlerine dönüştürülür:

```text
<nar_control emotion=sadness intensity=0.900 delivery=crying_speech>
Bugün <nar_event type=sob after_word=1 duration=short count=1> seni düşündüm.
```

## Eğitim

1. `nar-tts codec-check` ile Mimi dönüşümünü kontrol et.
2. `nar-tts audit-data` ile bozuk, tekrar veya lisansı eksik kayıtları ayır.
3. Neutral ve expressive veriyi speaker/metin sızıntısı olmadan böl.
4. `nar-tts encode-expressive` ile eğitim Parquet dosyasını üret.
5. Neutral ve expressive kayıtları birlikte SFT eğitimine ver.
6. Sabit test kümesi başarılıysa emotion/event GRPO ağırlıklarını aç.

Veride emotion, intensity, delivery, event türü, konumu, süresi, speaker,
dil, kaynak ve lisans alanları tutulmalıdır. Aynı metnin aynı speaker tarafından
farklı emotion seviyelerinde okunması kontrolü daha belirgin hale getirir.

## Değerlendirme

- Kontrol işaretleri çıkarıldıktan sonra WER/CER
- Emotion ve intensity doğruluğu
- Event türü, sayısı ve konumu için F1
- Speaker similarity ve speaker drift
- Neutral kalite gerilemesi
- Kör insan A/B testi

Mevcut checkpoint bu işaretleri öğrenmediği için tek başına ağlayarak veya
kahkahalı okuyamaz. Bu özellikler için expressive SFT checkpoint'i gerekir.
