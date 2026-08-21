import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TURKISH_CHARACTERS = frozenset("çÇğĞıİöÖşŞüÜ")
TURKISH_PROSE_MARKERS = frozenset(
    {
        "ama",
        "ancak",
        "artık",
        "asla",
        "bir",
        "bu",
        "bütün",
        "çıktı",
        "degil",
        "değil",
        "dosya",
        "egitim",
        "eğitim",
        "göre",
        "icin",
        "için",
        "ile",
        "kalite",
        "kullanici",
        "kullanılır",
        "kullanilir",
        "kullanıcı",
        "metin",
        "olarak",
        "önce",
        "sistemi",
        "sonra",
        "ve",
        "veri",
        "yalnizca",
        "yalnızca",
    }
)


class DocumentationLanguageTest(unittest.TestCase):
    def test_docs_are_written_in_english(self):
        violations = []

        for path in sorted((REPOSITORY_ROOT / "docs").rglob("*.md")):
            content = path.read_text(encoding="utf-8")
            characters = sorted(set(content) & TURKISH_CHARACTERS)
            words = set(re.findall(r"[^\W\d_]+", content.casefold()))
            markers = sorted(words & TURKISH_PROSE_MARKERS)

            if characters or markers:
                details = []
                if characters:
                    details.append(f"characters={''.join(characters)!r}")
                if markers:
                    details.append(f"words={', '.join(markers)}")
                relative_path = path.relative_to(REPOSITORY_ROOT)
                violations.append(f"{relative_path}: " + "; ".join(details))

        self.assertFalse(
            violations,
            "Documentation under docs/ must be English-only:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
