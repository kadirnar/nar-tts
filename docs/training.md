# Eğitim sırası ve veri formatları

## İlk ayar

Eğitim dosyaları `nar_tts/configs/train/`, isteğe bağlı inference override'ı
`nar_tts/configs/inference/` altındadır. Model adına özel config yoktur.

Tokenizer değerlerini göster:

```bash
nar-tts inspect-tokenizer --model /path/to/base-model
```

Çıktıdaki `text_eos_token_id` ve `suggested_pad_token_id` değerlerini
`pretrain.yaml`, `finetune.yaml` ve `grpo.yaml` içindeki `tokens` bölümüne yaz.
Her dosyada `REQUIRED` olarak işaretlenen model ve veri yollarını da doldur.

SFT için `model.loader`, `transformers` veya `unsloth` olabilir.
`peft.enabled: true` iki yükleyicide de LoRA'yı açar; varsayılan tam eğitimdir.

## Eğitim sırası

1. Ham sesi Mimi tokenlarına dönüştür:

   ```bash
   nar-tts preprocess --config nar_tts/configs/train/preprocess.yaml
   ```

2. Base modeli text ve TTS verisiyle eğit:

   ```bash
   accelerate launch --config_file nar_tts/configs/train/launch/fsdp.yaml \
     nar_tts/training/pretrain.py --config nar_tts/configs/train/pretrain.yaml
   ```

3. İsteğe bağlı kaliteli SFT yap:

   ```bash
   accelerate launch --config_file nar_tts/configs/train/launch/fsdp.yaml \
     nar_tts/training/finetune.py --config nar_tts/configs/train/finetune.yaml
   ```

4. İsteğe bağlı kalite GRPO eğitimi yap:

   ```bash
   torchrun --standalone --nproc-per-node=8 \
     nar_tts/training/grpo.py --config nar_tts/configs/train/grpo.yaml
   ```

Her aşamada checkpoint ile birlikte kaydedilen tokenizer bir sonraki aşamada
kullanılmalıdır.

## Parquet formatları

Preprocess girdisinde `audio` ve `text` sütunları zorunludur. `audio`, Hugging
Face Audio nesnesi veya dosya yolu olabilir.

| Aşama | Zorunlu alan | İsteğe bağlı alanlar |
|---|---|---|
| Text pretrain | `input_ids: list[int]` | yok |
| TTS pretrain/SFT | `input_ids: list[int]` | `speaker`, `language`, `emotion`, `events` |
| GRPO `tts_tokens` | `input_ids: list[int]` | `language`, `emotion`, `intensity`, `delivery`, `events`, `hard_case` |

`input_ids`, metin ve Mimi ses tokenlarını içeren tamamlanmış Nar dizisidir.
Ham WAV dosyası eğitim sırasında tekrar okunmaz.

Expressive JSONL girişi şu biçimdedir:

```json
{"audio":"voice.wav","text":"Merhaba","speaker":"spk1","language":"tr","emotion":"joy","intensity":0.8,"delivery":"speech_laugh","events":[{"type":"laugh","after_word":1}],"license":"owned"}
```

Bu dosyayı Parquet'e dönüştür:

```bash
nar-tts encode-expressive --manifest expressive.jsonl \
  --output expressive.parquet --tokenizer /path/to/tokenizer
```

## W&B

Üç eğitim config'inde de `logging.enabled: true` ve `report_to: wandb` bulunur.
İlk kullanımdan önce `wandb login` çalıştırılır.
`project`, `run_name`, `entity`, `group`, `tags` ve `mode` alanları isteğe göre
değiştirilebilir. W&B kapatılacaksa yalnızca `logging.enabled: false` yazılır.
