# Kalite sistemi

## Eklenenler

Nar artık tek inference yapılandırmasıyla şu akışı kullanır:

1. Türkçe/İngilizce sayı, tarih, para, kısaltma ve kullanıcı sözlüğü
   normalizasyonu.
2. Aynı referans ses için içerik tabanlı Mimi token cache'i.
3. Gerçek batch üretim ve KV cache.
4. Önce iki aday; kalite kapısı geçilmezse toplam dört aday.
5. Qwen3-ASR ödülünden bağımsız Whisper doğrulaması.
6. Speaker similarity, CER, süre, clipping, sessizlik ve tekrar kontrolü.
7. Kazanan WAV, bütün adaylar ve makine okunabilir JSON raporu.
8. Uzun metinde cümle parçalama, önceki parçadan akustik bağlam ve crossfade.

Varsayılan ayarlar inference koduna gömülüdür. Kalıcı değişiklikler için isteğe
bağlı [`override.yaml`](../nar_tts/configs/inference/override.yaml) kullanılabilir.
Best-of-N ve doğrulama ek hesaplama yapar. JSON raporundaki `real_time_factor`
hız/kalite deneylerini karşılaştırır.

## Veri döngüsü

```text
ham manifest
  -> audit-data
  -> codec-check
  -> encode-expressive
  -> SFT
  -> GRPO
  -> bağımsız evaluate + dinleme testi
  -> hard cases / distill
  -> sonraki SFT veya GRPO turu
```

`audit-data`; bozuk, duplicate, aşırı sessiz, clipping içeren ve metin/süre
oranı şüpheli kayıtları ayırır. `encode-expressive`, `input_ids` yanında
speaker, emotion, event, kaynak ve lisans alanlarını da Parquet'te tutar.
Best-of-N raporlarından yalnız eşikleri geçen kazananlar `distill` ile yeni SFT
manifestine alınır. Bu satırlar `hard_case=true` taşır ve GRPO'da daha sık
örneklenebilir.

## GRPO

Tek [`grpo.yaml`](../nar_tts/configs/train/grpo.yaml) şu aktif bileşenleri ayrı ayrı
prompt grubu içinde normalize eder:

- Qwen3-ASR CER + ground-truth NLL
- speaker similarity
- süre tutarlılığı
- teknik sinyal kalitesi
- referansa göre kaba prosody uyumu
- kayan pencerelerde speaker drift

Emotion ve non-verbal event reward kodu hazırdır, ancak ağırlığı sıfırdır.
Sentetik Türkçe ses üzerinde bağımsız doğrulanmış classifier seçilmeden bu iki
reward açılmamalıdır. Tek SER modelini başarı ölçütü ve reward olarak birlikte
kullanmak reward hacking üretir.

## Başarı ölçütü

Her model sürümünde aynı sabit test kümesi kullanılmalıdır:

- Türkçe/İngilizce/Japonca CER ve WER
- speaker similarity ve uzun-form speaker drift
- p50/p95 RTF, VRAM ve ilk parça gecikmesi
- clipping, sessizlik, tekrar ve truncation oranı
- emotion doğruluğu, event F1 ve konum hatası
- naturalness, emotion ve speaker için kör insan A/B testi

`technical_quality` bir sinyal kontrolüdür; MOS veya doğallık modeli değildir.
Emotion classifier skoru da tek başına ürün kalitesi kanıtı değildir.

## Yeniden eğitim gerektirenler

Emotion markup'ı eski checkpoint'e yeni yetenek kazandırmaz. Ağlamaklı konuşma,
speech-laugh, kahkaha ve hıçkırık için etiketli expressive SFT gerekir. Codec,
ses token düzeni veya özel kontrol tokenları değişirse tüm ses verisi yeniden
encode edilmeli ve model yeniden eğitilmelidir. Bu nedenle alternatif codec,
yeni decoder ve gerçek frame-level streaming mevcut checkpoint'e otomatik
uygulanmaz; ayrı model nesli olarak ölçülmelidir.
