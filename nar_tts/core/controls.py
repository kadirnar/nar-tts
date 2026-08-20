"""Stable text, emotion, and non-verbal control schema for Nar TTS.

The markup deliberately uses ordinary text tokens.  Existing checkpoints keep
their audio-token layout, while expressive checkpoints can learn the same
schema without a tokenizer migration.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

EMOTIONS = frozenset({"neutral", "joy", "sadness", "anger", "fear", "surprise"})
DELIVERIES = frozenset({"neutral", "crying_speech", "speech_laugh"})
EVENT_TYPES = frozenset(
    {"laugh", "chuckle", "sob", "cry", "sniff", "sigh", "gasp", "breath"}
)
EVENT_DURATIONS = frozenset({"short", "medium", "long"})

_CONTROL_TAG = re.compile(r"<nar_control\b[^>]*>", re.IGNORECASE)
_EVENT_TAG = re.compile(r"<nar_event\b[^>]*>", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", re.UNICODE)


def _bounded(value, name: str, lower: float, upper: float) -> float:
    value = float(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower:g} and {upper:g}")
    return value


@dataclass(frozen=True)
class VocalEvent:
    """A controlled non-verbal event and its intended position."""

    type: str
    after_word: int | None = None
    at_seconds: float | None = None
    duration: str | float = "medium"
    count: int = 1

    def __post_init__(self):
        event_type = str(self.type).strip().casefold()
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported vocal event: {self.type!r}")
        object.__setattr__(self, "type", event_type)
        if self.after_word is not None and int(self.after_word) < 0:
            raise ValueError("after_word cannot be negative")
        if self.at_seconds is not None and float(self.at_seconds) < 0:
            raise ValueError("at_seconds cannot be negative")
        if self.after_word is not None and self.at_seconds is not None:
            raise ValueError("set after_word or at_seconds, not both")
        if isinstance(self.duration, str):
            duration = self.duration.strip().casefold()
            if duration not in EVENT_DURATIONS:
                raise ValueError(
                    "event duration must be short, medium, long, or seconds"
                )
            object.__setattr__(self, "duration", duration)
        elif float(self.duration) <= 0:
            raise ValueError("event duration in seconds must be positive")
        if int(self.count) < 1:
            raise ValueError("event count must be positive")
        object.__setattr__(self, "count", int(self.count))
        if self.after_word is not None:
            object.__setattr__(self, "after_word", int(self.after_word))
        if self.at_seconds is not None:
            object.__setattr__(self, "at_seconds", float(self.at_seconds))

    @classmethod
    def from_dict(cls, value: Mapping) -> VocalEvent:
        return cls(
            type=value.get("type", ""),
            after_word=value.get("after_word"),
            at_seconds=value.get("at_seconds"),
            duration=value.get("duration", "medium"),
            count=value.get("count", 1),
        )

    def tag(self) -> str:
        fields = [f"type={self.type}"]
        if self.after_word is not None:
            fields.append(f"after_word={self.after_word}")
        if self.at_seconds is not None:
            fields.append(f"at_seconds={self.at_seconds:.3f}")
        duration = (
            self.duration
            if isinstance(self.duration, str)
            else f"{float(self.duration):.3f}s"
        )
        fields.extend((f"duration={duration}", f"count={self.count}"))
        return "<nar_event " + " ".join(fields) + ">"

    def asdict(self) -> dict:
        return {
            "type": self.type,
            "after_word": self.after_word,
            "at_seconds": self.at_seconds,
            "duration": self.duration,
            "count": self.count,
        }


@dataclass(frozen=True)
class SpeechControl:
    """Independent emotion, delivery, and event controls for one utterance."""

    emotion: str = "neutral"
    intensity: float = 0.0
    delivery: str = "neutral"
    valence: float | None = None
    arousal: float | None = None
    events: tuple[VocalEvent, ...] = field(default_factory=tuple)

    def __post_init__(self):
        emotion = str(self.emotion).strip().casefold()
        delivery = str(self.delivery).strip().casefold()
        if emotion not in EMOTIONS:
            raise ValueError(f"unsupported emotion: {self.emotion!r}")
        if delivery not in DELIVERIES:
            raise ValueError(f"unsupported delivery: {self.delivery!r}")
        object.__setattr__(self, "emotion", emotion)
        object.__setattr__(self, "delivery", delivery)
        object.__setattr__(
            self, "intensity", _bounded(self.intensity, "intensity", 0.0, 1.0)
        )
        if self.valence is not None:
            object.__setattr__(
                self, "valence", _bounded(self.valence, "valence", -1.0, 1.0)
            )
        if self.arousal is not None:
            object.__setattr__(
                self, "arousal", _bounded(self.arousal, "arousal", -1.0, 1.0)
            )
        events = tuple(
            item if isinstance(item, VocalEvent) else VocalEvent.from_dict(item)
            for item in self.events
        )
        object.__setattr__(self, "events", events)

    @classmethod
    def from_dict(cls, value: Mapping | None) -> SpeechControl:
        value = value or {}
        return cls(
            emotion=value.get("emotion", "neutral"),
            intensity=value.get("intensity", 0.0),
            delivery=value.get("delivery", "neutral"),
            valence=value.get("valence"),
            arousal=value.get("arousal"),
            events=tuple(value.get("events") or ()),
        )

    @property
    def is_neutral(self) -> bool:
        return (
            self.emotion == "neutral"
            and self.intensity == 0.0
            and self.delivery == "neutral"
            and not self.events
            and self.valence is None
            and self.arousal is None
        )

    def header(self) -> str:
        fields = [
            f"emotion={self.emotion}",
            f"intensity={self.intensity:.3f}",
            f"delivery={self.delivery}",
        ]
        if self.valence is not None:
            fields.append(f"valence={self.valence:.3f}")
        if self.arousal is not None:
            fields.append(f"arousal={self.arousal:.3f}")
        return "<nar_control " + " ".join(fields) + ">"

    def asdict(self) -> dict:
        return {
            "emotion": self.emotion,
            "intensity": self.intensity,
            "delivery": self.delivery,
            "valence": self.valence,
            "arousal": self.arousal,
            "events": [event.asdict() for event in self.events],
        }


def render_controlled_text(
    text: str,
    control: SpeechControl | Mapping | None = None,
    *,
    include_neutral: bool = False,
) -> str:
    """Serialize controls and insert word-positioned events deterministically."""
    text = _SPACE.sub(" ", str(text)).strip()
    if not text:
        raise ValueError("text cannot be empty")
    control = (
        control
        if isinstance(control, SpeechControl)
        else SpeechControl.from_dict(control)
    )
    tokens = _WORD.findall(text)
    by_position: dict[int, list[VocalEvent]] = {}
    unpositioned = []
    for event in control.events:
        if event.after_word is None:
            unpositioned.append(event)
        else:
            by_position.setdefault(event.after_word, []).append(event)

    output = [event.tag() for event in by_position.get(0, ())]
    word_count = 0
    for token in tokens:
        output.append(token)
        if any(character.isalnum() for character in token):
            word_count += 1
            output.extend(event.tag() for event in by_position.get(word_count, ()))
    # Positions beyond the text remain explicit rather than being silently lost.
    for position in sorted(key for key in by_position if key > word_count):
        output.extend(event.tag() for event in by_position[position])
    output.extend(event.tag() for event in unpositioned)

    # Remove spaces before common punctuation after tokenization.
    body = " ".join(output)
    body = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", body)
    body = re.sub(r"([\(\[\{])\s+", r"\1", body)
    if control.is_neutral and not include_neutral:
        return body
    return f"{control.header()} {body}"


def strip_control_markup(text: str) -> str:
    """Remove Nar control tags before lexical ASR/CER evaluation."""
    text = _CONTROL_TAG.sub(" ", str(text))
    text = _EVENT_TAG.sub(" ", text)
    return html.unescape(_SPACE.sub(" ", text).strip())


_TR_ONES = (
    "",
    "bir",
    "iki",
    "üç",
    "dört",
    "beş",
    "altı",
    "yedi",
    "sekiz",
    "dokuz",
)
_TR_TENS = (
    "",
    "on",
    "yirmi",
    "otuz",
    "kırk",
    "elli",
    "altmış",
    "yetmiş",
    "seksen",
    "doksan",
)
_TR_SCALES = ("", "bin", "milyon", "milyar", "trilyon", "katrilyon")
_EN_SMALL = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_EN_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_EN_SCALES = ("", "thousand", "million", "billion", "trillion", "quadrillion")


def _tr_under_thousand(value: int) -> str:
    parts = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        if hundreds > 1:
            parts.append(_TR_ONES[hundreds])
        parts.append("yüz")
    tens, ones = divmod(remainder, 10)
    if tens:
        parts.append(_TR_TENS[tens])
    if ones:
        parts.append(_TR_ONES[ones])
    return " ".join(parts)


def _en_under_thousand(value: int) -> str:
    parts = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        parts.extend((_EN_SMALL[hundreds], "hundred"))
    if remainder >= 20:
        tens, ones = divmod(remainder, 10)
        parts.append(_EN_TENS[tens])
        if ones:
            parts.append(_EN_SMALL[ones])
    elif remainder:
        parts.append(_EN_SMALL[remainder])
    return " ".join(parts)


def integer_to_words(value: int, language: str = "tr") -> str:
    """Expand a signed integer in Turkish or English without extra packages."""
    value = int(value)
    language = language.casefold().split("-")[0]
    if value == 0:
        return "sıfır" if language == "tr" else "zero"
    if abs(value) >= 1000 ** len(_TR_SCALES):
        return str(value)
    negative = value < 0
    value = abs(value)
    scales = _TR_SCALES if language == "tr" else _EN_SCALES
    convert = _tr_under_thousand if language == "tr" else _en_under_thousand
    chunks = []
    scale_index = 0
    while value:
        value, chunk = divmod(value, 1000)
        if chunk:
            words = convert(chunk)
            if language == "tr" and scale_index == 1 and chunk == 1:
                words = ""
            chunks.append(
                " ".join(part for part in (words, scales[scale_index]) if part)
            )
        scale_index += 1
    result = " ".join(reversed(chunks))
    if negative:
        result = ("eksi " if language == "tr" else "minus ") + result
    return result


_TR_MONTHS = {
    1: "ocak",
    2: "şubat",
    3: "mart",
    4: "nisan",
    5: "mayıs",
    6: "haziran",
    7: "temmuz",
    8: "ağustos",
    9: "eylül",
    10: "ekim",
    11: "kasım",
    12: "aralık",
}


class TextFrontend:
    """Conservative Unicode, lexicon, date, currency, and number frontend."""

    def __init__(
        self,
        language: str = "tr",
        *,
        expand_numbers: bool = True,
        expand_abbreviations: bool = True,
        lexicon: Mapping[str, str] | None = None,
    ):
        self.language = str(language or "tr").casefold().replace("_", "-")
        self.expand_numbers = bool(expand_numbers)
        self.expand_abbreviations = bool(expand_abbreviations)
        self.lexicon = {
            str(key).casefold(): str(value) for key, value in (lexicon or {}).items()
        }

    @property
    def base_language(self) -> str:
        return self.language.split("-")[0]

    def _expand_date(self, match) -> str:
        day, month, year = (int(value) for value in match.groups())
        if self.base_language != "tr" or month not in _TR_MONTHS:
            return match.group(0)
        return f"{integer_to_words(day, 'tr')} {_TR_MONTHS[month]} {integer_to_words(year, 'tr')}"

    def _number_text(self, token: str) -> str:
        token = str(token)
        decimal_separator = "," if "," in token else "." if "." in token else None
        try:
            value = Decimal(
                token.replace(".", "").replace(",", ".") if "," in token else token
            )
        except InvalidOperation:
            return token
        if value == value.to_integral_value():
            return integer_to_words(int(value), self.base_language)
        left, right = token.replace("-", "").split(decimal_separator, 1)
        left_words = integer_to_words(int(left or 0), self.base_language)
        digit_names = [
            integer_to_words(int(character), self.base_language) for character in right
        ]
        separator = "virgül" if self.base_language == "tr" else "point"
        prefix = (
            "eksi "
            if token.startswith("-") and self.base_language == "tr"
            else "minus "
            if token.startswith("-")
            else ""
        )
        return f"{prefix}{left_words} {separator} {' '.join(digit_names)}"

    def _number(self, match) -> str:
        return self._number_text(match.group(0))

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text))
        text = text.translate(
            str.maketrans(
                {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "..."}
            )
        )
        if self.expand_abbreviations:
            abbreviations = (
                {"dr.": "doktor", "prof.": "profesör", "doç.": "doçent"}
                if self.base_language == "tr"
                else {
                    "dr.": "doctor",
                    "prof.": "professor",
                    "mr.": "mister",
                    "mrs.": "missus",
                }
            )
            for source, target in abbreviations.items():
                text = re.sub(
                    rf"(?<!\w){re.escape(source)}", target, text, flags=re.IGNORECASE
                )
        if self.base_language == "tr" and self.expand_numbers:
            text = re.sub(
                r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", self._expand_date, text
            )
            text = re.sub(
                r"(?:₺\s*(\d+)|\b(\d+)\s*TL\b)",
                lambda m: (
                    f"{integer_to_words(int(m.group(1) or m.group(2)), 'tr')} Türk lirası"
                ),
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"%(\d+(?:[.,]\d+)?)",
                lambda m: "yüzde " + self._number_text(m.group(1)),
                text,
            )
        elif self.base_language == "en" and self.expand_numbers:
            text = re.sub(
                r"\$(\d+(?:\.\d+)?)",
                lambda m: self._number_text(m.group(1)) + " dollars",
                text,
            )
            text = re.sub(
                r"%(\d+(?:\.\d+)?)",
                lambda m: self._number_text(m.group(1)) + " percent",
                text,
            )
        if self.expand_numbers and self.base_language in {"tr", "en"}:
            text = re.sub(r"(?<!\w)-?\d+(?:[.,]\d+)?(?!\w)", self._number, text)
        if self.lexicon:
            pattern = re.compile(
                r"(?<!\w)("
                + "|".join(
                    re.escape(key)
                    for key in sorted(self.lexicon, key=len, reverse=True)
                )
                + r")(?!\w)",
                re.IGNORECASE,
            )
            text = pattern.sub(
                lambda match: self.lexicon[match.group(0).casefold()], text
            )
        return _SPACE.sub(" ", text).strip()


def split_long_text(text: str, max_characters: int = 280) -> list[str]:
    """Split at sentence/phrase boundaries while preserving every character."""
    text = _SPACE.sub(" ", str(text)).strip()
    if not text:
        return []
    if max_characters < 20:
        raise ValueError("max_characters must be at least 20")
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(sentence) <= max_characters:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_characters:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        parts = re.split(r"(?<=[,;:，；：])\s*", sentence)
        for part in parts:
            candidate = f"{current} {part}".strip()
            if current and len(candidate) > max_characters:
                chunks.append(current)
                current = part
            else:
                current = candidate
            while len(current) > max_characters:
                boundary = current.rfind(" ", 0, max_characters + 1)
                boundary = boundary if boundary > 0 else max_characters
                chunks.append(current[:boundary].strip())
                current = current[boundary:].strip()
    if current:
        chunks.append(current)
    return chunks


def controls_from_columns(
    emotion=None,
    intensity=None,
    delivery=None,
    events: Iterable[Mapping | VocalEvent] | None = None,
    valence=None,
    arousal=None,
) -> SpeechControl:
    """Build a control object from nullable dataset columns."""
    return SpeechControl(
        emotion=emotion or "neutral",
        intensity=0.0 if intensity in (None, "") else intensity,
        delivery=delivery or "neutral",
        events=tuple(events or ()),
        valence=None if valence in (None, "") else valence,
        arousal=None if arousal in (None, "") else arousal,
    )
