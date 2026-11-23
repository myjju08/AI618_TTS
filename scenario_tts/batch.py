"""CSV planning, resumable generation, and per-sample audit records."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, TextIO

import numpy as np
import soundfile as sf

from .audio import mix_audio
from .backends import SPEAKERS, Backend
from .presets import ACCENTS, BACKGROUNDS, PRESETS, build_instruction


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    text: str


@dataclass(frozen=True)
class Job:
    sample_id: str
    sentence_id: str
    text: str
    preset: str
    accent: str
    speaker: str
    instruct: str
    seed: int
    background: str
    snr_db: float


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_BASE_FIELDS = list(Job.__dataclass_fields__) + [
    "artifact_id", "backend", "model_id", "backend_config", "resolved_device", "resolved_dtype",
    "resolved_attention", "status", "error", "created_at",
    "clean_path", "mixed_path", "sample_rate", "duration_seconds", "mixed_duration_seconds",
    "background_sha256", "background_source", "background_offset_seconds", "target_snr_db",
    "achieved_snr_db", "mix_gain", "lead_seconds", "tail_seconds", "fade_seconds",
]


def _validate_sentences(sentences: list[Sentence]) -> None:
    if not sentences:
        raise ValueError("The CSV must contain at least one sentence.")
    seen = set()
    for sentence in sentences:
        if not _ID_PATTERN.fullmatch(sentence.sentence_id):
            raise ValueError(f"Unsafe sentence_id {sentence.sentence_id!r}: use 1–64 letters, digits, underscores, dots or hyphens; start with a letter or digit.")
        if sentence.sentence_id in seen:
            raise ValueError(f"Duplicate sentence_id: {sentence.sentence_id}")
        if not sentence.text.strip() or "\x00" in sentence.text:
            raise ValueError(f"Sentence {sentence.sentence_id} has empty or invalid text.")
        seen.add(sentence.sentence_id)


def read_sentences(source: str | Path | TextIO) -> list[Sentence]:
    """Read UTF-8 CSV; IDs remain strings (including leading zeros)."""
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8-sig", newline="") as stream:
            return read_sentences(stream)
    content = source.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")), strict=True)
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("CSV headers must be present and unique.")
    if not {"sentence_id", "text"}.issubset(reader.fieldnames):
        raise ValueError("CSV must include sentence_id and text columns.")
    sentences = []
    for line, row in enumerate(reader, start=2):
        if None in row or row.get("sentence_id") is None or row.get("text") is None:
            raise ValueError(f"Malformed CSV row {line}; check quoting and column counts.")
        sentences.append(Sentence(row["sentence_id"].strip(), row["text"]))
    _validate_sentences(sentences)
    return sentences


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _with_id(job: Job) -> Job:
    data = asdict(job)
    data.pop("sample_id")
    return replace(job, sample_id=f"{job.sentence_id}-{job.preset}-{job.accent}-{_digest(data)[:16]}")


def create_jobs(
    sentences: Iterable[Sentence], presets: Iterable[str], accents: Iterable[str], speaker: str = "Ryan",
    base_seed: int = 42, tone: str | None = None, background: str | None = None, snr_db: float | None = None,
) -> list[Job]:
    sentences, presets, accents = list(sentences), list(dict.fromkeys(presets)), list(dict.fromkeys(accents))
    _validate_sentences(sentences)
    if not presets or not accents:
        raise ValueError("Select at least one preset and accent.")
    if any(preset not in PRESETS for preset in presets):
        raise ValueError("Unknown scenario preset.")
    if any(accent not in ACCENTS for accent in accents):
        raise ValueError("Unknown accent.")
    if speaker not in SPEAKERS:
        raise ValueError(f"Speaker must be one of {SPEAKERS}.")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or not 0 <= base_seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 4294967295.")
    if background is not None and background not in BACKGROUNDS:
        raise ValueError("Unknown background.")
    if snr_db is not None and (not math.isfinite(snr_db) or not -20 <= snr_db <= 80):
        raise ValueError("SNR must be finite and between -20 and 80 dB.")
    jobs = []
    for sentence in sentences:
        for preset in presets:
            for accent in accents:
                seed = int(_digest([base_seed, sentence.sentence_id, sentence.text, preset, accent, speaker])[:8], 16)
                scenario = PRESETS[preset]
                jobs.append(_with_id(Job("", sentence.sentence_id, sentence.text, preset, accent, speaker,
                                         build_instruction(preset, accent, tone), seed,
                                         scenario.background if background is None else background,
                                         float(scenario.snr_db if snr_db is None else snr_db))))
    return jobs


def regenerate_job(job: Job, seed: int | None = None) -> Job:
    if seed is None:
        seed = (job.seed + 1) % 2**32
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 4294967295.")
    if seed == job.seed:
        raise ValueError("Choose a different seed to generate a new sample; reuse the original job to resume.")
    return _with_id(replace(job, seed=seed))


def load_manifest(output_dir: str | Path) -> list[dict]:
    path = Path(output_dir) / "metadata.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if rows and any("artifact_id" not in row for row in rows):
        raise ValueError("Existing metadata.csv has an incompatible schema; choose a new output directory.")
    return rows


def _write_manifest(output_dir: Path, rows: list[dict]) -> None:
    fields = _BASE_FIELDS + sorted(set().union(*(row.keys() for row in rows)) - set(_BASE_FIELDS))
    temporary = output_dir / f".metadata-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_dir / "metadata.csv")
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _output_lock(directory: Path):
    lock = directory / ".batch.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another batch is using {directory}. If a previous process crashed, remove .batch.lock after checking it is stopped.") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def _audio_file_valid(directory: Path, relative_path: str, sample_rate: int, duration: float) -> bool:
    try:
        path = (directory / relative_path).resolve()
        if not path.is_relative_to(directory.resolve()):
            return False
        info = sf.info(path)
        return info.channels == 1 and info.samplerate == sample_rate and abs(info.frames / sample_rate - duration) < 1 / sample_rate
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _atomic_wav(path: Path, waveform: np.ndarray, sr: int) -> None:
    temporary = path.with_name(f".{path.stem}-{uuid.uuid4().hex}.tmp")
    try:
        sf.write(temporary, waveform, sr, format="WAV", subtype="FLOAT")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_batch(
    jobs: Iterable[Job], output_dir: str | Path, backend: Backend, background_dir: str | Path | None = None,
    progress: Callable[[int, int, dict], None] | None = None,
) -> list[dict]:
    jobs = list(jobs)
    if not jobs:
        raise ValueError("No jobs to generate.")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", job.sample_id) for job in jobs):
        raise ValueError("Unsafe sample_id; use create_jobs to construct generation requests.")
    if any(job.background not in BACKGROUNDS for job in jobs):
        raise ValueError("Unknown background; use create_jobs to construct generation requests.")
    if background_dir is not None:
        background_dir = Path(background_dir).expanduser().resolve()
        if not background_dir.is_dir():
            raise ValueError(f"Background directory does not exist or is not a directory: {background_dir}")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    preparation_attempted = False
    preparation_error = None
    with _output_lock(output_dir):
        manifest = load_manifest(output_dir)
        positions = {row["artifact_id"]: index for index, row in enumerate(manifest)}
        for folder in ("clean", "mixed"):
            (output_dir / folder).mkdir(exist_ok=True)
            if not (output_dir / folder).resolve().is_relative_to(output_dir):
                raise ValueError(f"Output subdirectory {folder} resolves outside the output directory.")
        for number, job in enumerate(jobs, 1):
            custom = Path(background_dir) / f"{job.background}.wav" if background_dir and job.background != "none" else None
            background_hash = "procedural-v1"
            background_error = None
            if custom is not None:
                try:
                    background_hash = hashlib.sha256(custom.read_bytes()).hexdigest()
                except FileNotFoundError:
                    pass  # An absent optional scene recording uses the labeled procedural bed.
                except OSError as exc:
                    background_hash = "unreadable"
                    background_error = exc
            backend_config = getattr(backend, "config", {})
            identity = _digest([asdict(job), backend.name, backend.model_id, backend_config, background_hash])[:12]
            artifact_id = f"{job.sample_id}-{identity}"
            index = positions.get(artifact_id)
            existing = manifest[index] if index is not None else None
            if existing and existing.get("status") == "success":
                try:
                    sr = int(existing["sample_rate"])
                    valid = _audio_file_valid(output_dir, existing["clean_path"], sr, float(existing["duration_seconds"])) and _audio_file_valid(output_dir, existing["mixed_path"], sr, float(existing["mixed_duration_seconds"]))
                except (KeyError, TypeError, ValueError):
                    valid = False
                if valid:
                    results.append(existing)
                    if progress:
                        progress(number, len(jobs), existing)
                    continue
            row = {**asdict(job), "artifact_id": artifact_id, "backend": backend.name, "model_id": backend.model_id,
                   "backend_config": json.dumps(backend_config, sort_keys=True),
                   "created_at": datetime.now(timezone.utc).isoformat(), "status": "failed", "error": "",
                   "clean_path": "", "mixed_path": "", "background_sha256": background_hash}
            try:
                if background_error is not None:
                    raise RuntimeError(f"Could not read background file {custom}: {background_error}") from background_error
                # Prepare lazily, so a fully cached batch never imports/downloads a
                # model. A network/load failure is attempted once, not once per job.
                if not preparation_attempted:
                    preparation_attempted = True
                    prepare = getattr(backend, "prepare", None)
                    if callable(prepare):
                        try:
                            prepare()
                        except Exception as exc:
                            preparation_error = exc
                if preparation_error is not None:
                    raise RuntimeError(f"Backend preparation failed: {preparation_error}") from preparation_error
                speech, sample_rate = backend.generate(text=job.text, speaker=job.speaker, instruct=job.instruct, seed=job.seed)
                speech = np.asarray(speech, dtype=np.float32)
                if speech.ndim != 1 or not speech.size or not np.isfinite(speech).all():
                    raise ValueError("The backend returned invalid mono audio.")
                if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
                    raise ValueError("The backend returned an invalid sample rate.")
                # A common gain prevents clean-file clipping without changing speech/noise SNR.
                speech = speech * min(1.0, 0.98 / max(float(np.max(np.abs(speech))), 1e-12))
                mixed = mix_audio(speech, sample_rate, background=job.background, snr_db=job.snr_db, seed=job.seed, background_dir=background_dir)
                clean_path, mixed_path = f"clean/{artifact_id}.wav", f"mixed/{artifact_id}.wav"
                _atomic_wav(output_dir / clean_path, speech, sample_rate)
                _atomic_wav(output_dir / mixed_path, mixed.audio, sample_rate)
                row.update(mixed.metadata)
                row.update(status="success", clean_path=clean_path, mixed_path=mixed_path, sample_rate=sample_rate,
                           duration_seconds=len(speech) / sample_rate, mixed_duration_seconds=len(mixed.audio) / sample_rate)
                row.update({field: getattr(backend, field, "") for field in ("resolved_device", "resolved_dtype", "resolved_attention")})
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            if index is None:
                positions[artifact_id] = len(manifest)
                manifest.append(row)
            else:
                manifest[index] = row
            _write_manifest(output_dir, manifest)
            results.append(row)
            if progress:
                progress(number, len(jobs), row)
    return results
