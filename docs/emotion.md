# Emotion ve non-verbal ses desteği

## Kontrol şeması

Nar üç şeyi ayrı kontrol eder:

- Emotion: `neutral`, `joy`, `sadness`, `anger`, `fear`, `surprise`
- Söyleyiş: `neutral`, `crying_speech`, `speech_laugh`
- Olay: `laugh`, `chuckle`, `sob`, `cry`, `sniff`, `sigh`, `gasp`, `breath`

`speech_laugh` metni kahkahalı okur; `laugh` ayrı bir kahkaha olayı üretir.
Ağlamaklı konuşma ile hıçkırık da aynı nedenle farklı etiketlenir
([WESR](https://arxiv.org/abs/2601.04508)).

```json
{
  "text": "Bugün seni düşündüm.",
  "emotion": "sadness",
  "intensity": 0.9,
  "delivery": "crying_speech",
  "events": [{"type": "sob", "after_word": 1, "duration": "short"}]
}
```

Kod bunu mevcut tokenizer'ı bozmayan metin markup'ına dönüştürür:

```text
<nar_control emotion=sadness intensity=0.900 delivery=crying_speech>
Bugün <nar_event type=sob after_word=1 duration=short count=1> seni düşündüm.
```

CLI örneği README'dedir. Eski checkpoint markup'ı görmediği için bu komut tek
başına emotion öğretmez; expressive SFT checkpoint'i gerekir.

## Eğitim

1. `nar-tts codec-check` ile Mimi'nin laugh, cry, sob ve speech-laugh seslerini
   koruduğunu dinleme testiyle birlikte doğrula.
2. `nar-tts audit-data` ile bozuk, duplicate ve lisansı eksik kayıtları ayır.
3. Neutral ve expressive kayıtları aynı speaker'larda, speaker/metin sızıntısı
   olmayan train/test bölümlerine ayır.
4. `nar-tts encode-expressive` ile kontrol, event zamanı, speaker, kaynak ve
   lisans metadata'sını koru.
5. Neutral + tagged expressive SFT yap. Aynı metnin farklı emotion okumaları,
   içerik ile emotion'ın ayrışmasını kolaylaştırır.
6. Bağımsız CER, event F1, speaker drift ve kör insan A/B testi başarılı olursa
   emotion/event GRPO ağırlıklarını küçük değerlerle aç.

Zaman kontrollü laughter için küçük expressive veri ile büyük neutral verinin
karıştırılması işe yarayabilir ([ELaTE](https://arxiv.org/abs/2402.07383)).
Tür, sıklık ve süre etiketleri kontrolü artırırken doğallık ayrı korunmalıdır
([fine-grained NV control](https://arxiv.org/abs/2605.25504)). Sürekli valence,
arousal ve intensity kontrolü için
[EmoCtrl-TTS](https://arxiv.org/abs/2407.12229),
[EmoSphere](https://arxiv.org/abs/2406.07803) ve
[EmoInstruct-TTS](https://arxiv.org/abs/2606.20650) izlenebilir.

## Veri

Öncelik, ticari kullanıma uygun kendi Türkçe/İngilizce/Japonca kayıtlarımızdır.
Araştırma karşılaştırması için:

| Kaynak | Kullanım | Dikkat |
|---|---|---|
| [SMIIP-NV](https://axunyii.github.io/SMIIP-NV/) | Laugh/cry/cough ve zamanlar | Mandarin, ticari değil |
| [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) | Beş emotion | Araştırma lisansı |
| [EmoV-DB](https://www.openslr.org/115/) | Emotional TTS baseline | Az speaker |
| [Expresso](https://arxiv.org/abs/2308.05725) | Doğal style ve olaylar | Lisansı ayrıca doğrula |
| [TurEV-DB](https://aclanthology.org/2020.sltu-1.52/) | Türkçe evaluation | Eğitim için küçük |
| [VocalSound](https://arxiv.org/abs/2205.03433) | Event detector | TTS metin eşleşmesi yok |

Nadir `cry` örneklerinde önce ortak non-verbal sınıf, sonra uzmanlaşma ve
laugh/breath aktarımı denenebilir
([Beyond Words](https://arxiv.org/abs/2607.01563)).

## Değerlendirme

- Markup çıkarıldıktan sonra WER/CER
- Event türü, sayısı, konumu ve süresi için F1/position error
- Emotion ve intensity sıralama doğruluğu
- Speaker similarity ve kayan pencere speaker drift
- Neutral gerilemesi, naturalness/emotion MOS ve insan A/B testi

[NV-Bench](https://arxiv.org/abs/2603.15352), instruction alignment ile
acoustic fidelity'yi ayrı ölçer. Sentetik seslerde SER genellemesi zayıf
olabildiğinden emotion classifier tek reward veya tek başarı ölçütü olmamalıdır
([SER incelemesi](https://arxiv.org/abs/2603.16483)).

İlk üretim hedefi: yüksek kaliteli `neutral`, `sadness + crying_speech`,
`joy + speech_laugh`, `sob` ve `laugh`. Diğer emotion'lar bu çekirdek kalite
korunduktan sonra açılmalıdır.
