import unittest

from nar_tts.core.controls import (
    SpeechControl,
    TextFrontend,
    VocalEvent,
    integer_to_words,
    render_controlled_text,
    split_long_text,
    strip_control_markup,
)


class SpeechControlTest(unittest.TestCase):
    def test_emotion_delivery_and_event_markup_are_deterministic(self):
        control = SpeechControl(
            emotion="sadness",
            intensity=0.9,
            delivery="crying_speech",
            valence=-0.8,
            events=(VocalEvent("sob", after_word=1, duration="short"),),
        )
        rendered = render_controlled_text("Bugün seni düşündüm.", control)
        self.assertEqual(
            rendered,
            "<nar_control emotion=sadness intensity=0.900 "
            "delivery=crying_speech valence=-0.800> Bugün "
            "<nar_event type=sob after_word=1 duration=short count=1> "
            "seni düşündüm.",
        )
        self.assertEqual(strip_control_markup(rendered), "Bugün seni düşündüm.")

    def test_neutral_text_does_not_change_existing_checkpoint_prompt(self):
        self.assertEqual(
            render_controlled_text("Merhaba dünya.", SpeechControl()),
            "Merhaba dünya.",
        )

    def test_invalid_controls_fail_before_model_loading(self):
        with self.assertRaises(ValueError):
            SpeechControl(emotion="unknown")
        with self.assertRaises(ValueError):
            SpeechControl(intensity=1.1)
        with self.assertRaises(ValueError):
            VocalEvent("laugh", after_word=1, at_seconds=0.5)


class TextFrontendTest(unittest.TestCase):
    def test_turkish_date_currency_percent_and_abbreviation(self):
        frontend = TextFrontend("tr")
        self.assertEqual(
            frontend.normalize("Dr. Ali 21.08.2026'da ₺150 ve %25 ödedi."),
            "doktor Ali yirmi bir ağustos iki bin yirmi altı'da yüz elli "
            "Türk lirası ve yüzde yirmi beş ödedi.",
        )

    def test_number_words_and_user_lexicon(self):
        self.assertEqual(integer_to_words(2026, "tr"), "iki bin yirmi altı")
        self.assertEqual(integer_to_words(2026, "en"), "two thousand twenty six")
        frontend = TextFrontend("tr", lexicon={"Nar": "nar ti ti es"})
        self.assertEqual(frontend.normalize("Nar 3 sürümü"), "nar ti ti es üç sürümü")

    def test_long_text_prefers_sentence_boundaries(self):
        self.assertEqual(
            split_long_text(
                "Birinci cümle. İkinci cümle! Üçüncü cümle?", max_characters=25
            ),
            ["Birinci cümle.", "İkinci cümle!", "Üçüncü cümle?"],
        )


if __name__ == "__main__":
    unittest.main()
