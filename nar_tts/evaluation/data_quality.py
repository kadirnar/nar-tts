"""Manifest-level audio/text quality audit and hard-case feedback storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from nar_tts.evaluation.metrics import analyze_waveform


def read_jsonl(path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row at {path}:{line_number} is not an object")
            yield value


def write_jsonl_atomic(path, rows: Iterable[Mapping]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    count = 0
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return count


@dataclass(frozen=True)
class AuditThresholds:
    minimum_seconds: float = 0.4
    maximum_seconds: float = 30.0
    minimum_characters: int = 2
    maximum_characters: int = 1000
    maximum_clipping_ratio: float = 0.001
    maximum_silence_ratio: float = 0.75
    maximum_dc_offset: float = 0.05
    minimum_rms_dbfs: float = -45.0
    maximum_rms_dbfs: float = -3.0
    minimum_characters_per_second: float = 1.0
    maximum_characters_per_second: float = 35.0
    require_license: bool = False

    def __post_init__(self):
        if self.minimum_seconds < 0 or self.maximum_seconds < self.minimum_seconds:
            raise ValueError("audio durations must satisfy 0 <= minimum <= maximum")
        if (
            self.minimum_characters < 0
            or self.maximum_characters < self.minimum_characters
        ):
            raise ValueError("text lengths must satisfy 0 <= minimum <= maximum")
        for name in ("maximum_clipping_ratio", "maximum_silence_ratio"):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            self.minimum_characters_per_second < 0
            or self.maximum_characters_per_second < self.minimum_characters_per_second
        ):
            raise ValueError("character rates must satisfy 0 <= minimum <= maximum")


def _resolve_audio_path(row: Mapping, base_dir: Path) -> Path:
    value = row.get("audio") or row.get("audio_path") or row.get("path")
    if isinstance(value, Mapping):
        value = value.get("path")
    if not value:
        raise ValueError("missing audio path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=20)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifest(
    manifest,
    accepted_path,
    rejected_path,
    *,
    thresholds: AuditThresholds | Mapping | None = None,
    text_column: str = "text",
) -> dict:
    """Audit a JSONL manifest, preserving all metadata and rejection reasons."""
    import soundfile as sf

    thresholds = (
        thresholds
        if isinstance(thresholds, AuditThresholds)
        else AuditThresholds(**(thresholds or {}))
    )
    manifest = Path(manifest).resolve()
    accepted, rejected = [], []
    audio_hashes = {}
    text_speaker_keys = {}
    for index, source in enumerate(read_jsonl(manifest)):
        row = dict(source)
        reasons = []
        text = str(row.get(text_column, "")).strip()
        if (
            not thresholds.minimum_characters
            <= len(text)
            <= thresholds.maximum_characters
        ):
            reasons.append("text_length")
        if thresholds.require_license and not row.get("license"):
            reasons.append("missing_license")
        try:
            audio_path = _resolve_audio_path(row, manifest.parent)
            wave, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            metrics = analyze_waveform(wave, sample_rate)
            digest = _file_digest(audio_path)
            if digest in audio_hashes:
                reasons.append("duplicate_audio")
                row["duplicate_of"] = audio_hashes[digest]
            else:
                audio_hashes[digest] = row.get("id", index)
            if (
                not thresholds.minimum_seconds
                <= metrics.duration_seconds
                <= thresholds.maximum_seconds
            ):
                reasons.append("duration")
            if metrics.clipping_ratio > thresholds.maximum_clipping_ratio:
                reasons.append("clipping")
            if metrics.silence_ratio > thresholds.maximum_silence_ratio:
                reasons.append("silence")
            if abs(metrics.dc_offset) > thresholds.maximum_dc_offset:
                reasons.append("dc_offset")
            if (
                not thresholds.minimum_rms_dbfs
                <= metrics.rms_dbfs
                <= thresholds.maximum_rms_dbfs
            ):
                reasons.append("level")
            characters_per_second = len(text) / max(metrics.duration_seconds, 1e-6)
            if (
                not thresholds.minimum_characters_per_second
                <= characters_per_second
                <= thresholds.maximum_characters_per_second
            ):
                reasons.append("text_audio_ratio")
            row["audio"] = os.fspath(audio_path)
            row["quality"] = {
                **metrics.asdict(),
                "characters_per_second": characters_per_second,
                "content_hash": digest,
            }
        except Exception as error:  # noqa: BLE001 - retain failures in the audit
            reasons.append("audio_error")
            row["audio_error"] = f"{type(error).__name__}: {error}"

        speaker = str(row.get("speaker", "")).strip()
        duplicate_key = (speaker, " ".join(text.casefold().split()))
        if speaker and text and duplicate_key in text_speaker_keys:
            reasons.append("duplicate_speaker_text")
            row.setdefault("duplicate_of", text_speaker_keys[duplicate_key])
        elif speaker and text:
            text_speaker_keys[duplicate_key] = row.get("id", index)
        if reasons:
            row["rejection_reasons"] = sorted(set(reasons))
            rejected.append(row)
        else:
            accepted.append(row)
    write_jsonl_atomic(accepted_path, accepted)
    write_jsonl_atomic(rejected_path, rejected)
    reason_counts = {}
    for row in rejected:
        for reason in row["rejection_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "input": len(accepted) + len(rejected),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "reasons": dict(sorted(reason_counts.items())),
        "thresholds": asdict(thresholds),
    }


class HardCaseStore:
    """Append-only JSONL feedback for pronunciation and generation failures."""

    def __init__(self, path):
        self.path = Path(path)

    def append(self, row: Mapping) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def build_distillation_manifest(
    reports: Iterable[str | os.PathLike],
    output_path,
    *,
    minimum_score: float = 0.75,
    maximum_cer: float = 0.10,
) -> dict:
    """Select independently verified Best-of-N winners for later SFT."""
    selected = []
    skipped = 0
    for report_path in reports:
        report_path = Path(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        winner = report.get("winner") or {}
        verification = winner.get("verification") or {}
        score = float(winner.get("score", 0.0))
        cer = verification.get("cer")
        audio_path = winner.get("audio_path")
        if (
            not audio_path
            or winner.get("accepted") is not True
            or cer is None
            or score < minimum_score
            or float(cer) > maximum_cer
        ):
            skipped += 1
            continue
        selected.append(
            {
                "id": report.get("id", report_path.stem),
                "audio": audio_path,
                "text": report.get("text", ""),
                "reference_audio": report.get("reference_audio"),
                "reference_text": report.get("reference_text", ""),
                "control": report.get("control", {}),
                "teacher_score": score,
                "teacher_cer": cer,
                "hard_case": bool(report.get("hard_case", False)),
                "source_report": os.fspath(report_path.resolve()),
            }
        )
    write_jsonl_atomic(output_path, selected)
    return {
        "selected": len(selected),
        "skipped": skipped,
        "output": os.fspath(Path(output_path).resolve()),
    }
