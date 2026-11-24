"""Independent ASR and waveform audit. This does not perform human listening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SMALL_NUMBERS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()
ASR_OPTIONS = {
    "language": "en",
    "beam_size": 5,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "vad_filter": False,
    "initial_prompt": None,
    "hotwords": None,
}


def normalize_words(text: str) -> list[str]:
    """Ignore case/punctuation and equivalent numeric spellings from 0 to 20."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text.lower().replace("’", "'"))
    return [SMALL_NUMBERS[int(w)] if w.isdigit() and int(w) <= 20 else w for w in words]


def word_errors(reference: str, hypothesis: str) -> dict:
    expected, actual = normalize_words(reference), normalize_words(hypothesis)
    if not expected:
        raise ValueError("Reference must contain words.")
    # Each cell holds (total edits, substitutions, deletions, insertions).
    previous = [(i, 0, 0, i) for i in range(len(actual) + 1)]
    for i, word in enumerate(expected, 1):
        current = [(i, 0, i, 0)]
        for j, candidate in enumerate(actual, 1):
            if word == candidate:
                current.append(previous[j - 1])
            else:
                diagonal, above, left = previous[j - 1], previous[j], current[j - 1]
                current.append(
                    min(
                        (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                        (above[0] + 1, above[1], above[2] + 1, above[3]),
                        (left[0] + 1, left[1], left[2], left[3] + 1),
                        key=lambda item: item[0],
                    )
                )
        previous = current
    edits, substitutions, deletions, insertions = previous[-1]
    return {
        "wer": edits / len(expected),
        "word_count": len(expected),
        "edits": edits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def waveform_metrics(clean: np.ndarray, mixed: np.ndarray, sr: int, row: dict) -> dict:
    if clean.ndim != 1 or mixed.ndim != 1 or not clean.size or not mixed.size:
        raise ValueError("Expected nonempty mono WAVs.")
    if not np.isfinite(clean).all() or not np.isfinite(mixed).all():
        raise ValueError("Waveform contains NaN/Inf.")
    speech_rms = rms(clean)
    if speech_rms <= 1e-12:
        raise ValueError("Speech is silent.")
    gain = float(row["mix_gain"])
    start = round(float(row["lead_seconds"]) * sr)
    tail = round(float(row["tail_seconds"]) * sr)
    if len(mixed) != start + len(clean) + tail:
        raise ValueError("Mixed length does not match recorded lead/speech/tail.")
    noise = mixed[start : start + len(clean)] - clean * gain
    measured = None
    if row["background"] != "none":
        noise_rms = rms(noise)
        if noise_rms <= 1e-12:
            raise ValueError("Expected background but recovered no noise.")
        measured = 20 * math.log10(speech_rms * gain / noise_rms)
    return {
        "sample_rate": sr,
        "speech_seconds": len(clean) / sr,
        "mixed_seconds": len(mixed) / sr,
        "words_per_second_including_pauses": len(normalize_words(row["text"])) / (len(clean) / sr),
        "clean_rms_dbfs": 20 * math.log10(speech_rms),
        "clean_peak": float(np.max(np.abs(clean))),
        "mixed_peak": float(np.max(np.abs(mixed))),
        "clean_clipped_samples": int(np.sum(np.abs(clean) >= 1.0)),
        "mixed_clipped_samples": int(np.sum(np.abs(mixed) >= 1.0)),
        "measured_snr_db": measured,
        "snr_error_db": abs(measured - float(row["snr_db"])) if measured is not None else None,
        "clean_mixed_equal": bool(np.array_equal(clean, mixed)),
    }


def pitch_metrics(clean: np.ndarray, sr: int) -> dict:
    import librosa

    signal = librosa.resample(clean, orig_sr=sr, target_sr=16000)
    f0, _, _ = librosa.pyin(signal, sr=16000, fmin=60, fmax=500, frame_length=1024, hop_length=160)
    voiced = f0[np.isfinite(f0)]
    if not voiced.size:
        return {"f0_median_hz": None, "f0_p10_p90_semitones": None}
    low, high = np.percentile(voiced, [10, 90])
    return {
        "f0_median_hz": float(np.median(voiced)),
        "f0_p10_p90_semitones": float(12 * np.log2(high / low)),
    }


def contained_file(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe audio path: {relative}")
    return path


def audit(input_dir: Path, output_dir: Path, model_name: str, threads: int, pitch: bool) -> dict:
    from faster_whisper import WhisperModel

    output_dir.mkdir(parents=True, exist_ok=True)
    with (input_dir / "metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    settings = {
        "model": model_name,
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": threads,
        "audio_input": "SoundFile mono WAV -> scipy.resample_poly 16000 Hz float32",
        "options": ASR_OPTIONS,
        "versions": {
            name: version(name)
            for name in ["faster-whisper", "ctranslate2", "numpy", "scipy", "soundfile"]
            + (["librosa"] if pitch else [])
        },
    }
    cache_path = output_dir / "transcript_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    model = None

    def transcribe(path: Path) -> tuple[str, str]:
        nonlocal model
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = hashlib.sha256((digest + json.dumps(settings, sort_keys=True)).encode()).hexdigest()
        if key not in cache:
            if model is None:
                print(f"Loading independent Whisper {model_name} (CPU int8)", flush=True)
                model = WhisperModel(
                    model_name, device="cpu", compute_type="int8", cpu_threads=threads
                )
            # Expected text is never passed to the recognizer, avoiding answer leakage.
            samples, sample_rate = sf.read(path, dtype="float32")
            if sample_rate != 16000:
                divisor = math.gcd(sample_rate, 16000)
                samples = resample_poly(samples, 16000 // divisor, sample_rate // divisor)
            # Supplying 16 kHz samples avoids a second decoder and its version-specific APIs.
            segments, _ = model.transcribe(samples.astype(np.float32), **ASR_OPTIONS)
            cache[key] = " ".join(segment.text.strip() for segment in segments).strip()
            cache_path.write_text(
                json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        return cache[key], digest

    results = []
    for row in source_rows:
        result = {
            key: row[key]
            for key in (
                "artifact_id",
                "sentence_id",
                "text",
                "preset",
                "accent",
                "seed",
                "background",
                "snr_db",
            )
        }
        result["generation_status"] = row["status"]
        if row["status"] != "success":
            result.update(audit_status="generation_failed", error=row.get("error", ""))
            results.append(result)
            continue
        try:
            if row.get("backend") != "qwen":
                raise ValueError(
                    "Speech-quality audits accept real Qwen audio only, never Demo tones."
                )
            clean_path = contained_file(input_dir, row["clean_path"])
            mixed_path = contained_file(input_dir, row["mixed_path"])
            clean, sr = sf.read(clean_path, dtype="float64")
            mixed, mixed_sr = sf.read(mixed_path, dtype="float64")
            if sr != mixed_sr:
                raise ValueError("Clean/mixed sample rates differ.")
            result.update(waveform_metrics(clean, mixed, sr, row))
            if pitch:
                result.update(pitch_metrics(clean, sr))
            for kind, path in (("clean", clean_path), ("mixed", mixed_path)):
                transcript, digest = transcribe(path)
                result[f"{kind}_transcript"] = transcript
                result[f"{kind}_sha256"] = digest
                result.update(
                    {f"{kind}_{k}": v for k, v in word_errors(row["text"], transcript).items()}
                )
            result["audit_status"] = "measured"
            print(
                f"{row['sentence_id']} {row['preset']} {row['accent']}: "
                f"WER clean={result['clean_wer']:.3f}, mixed={result['mixed_wer']:.3f} | "
                f"{result['clean_transcript']} / {result['mixed_transcript']}",
                flush=True,
            )
        except Exception as exc:
            result.update(audit_status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"Audit error: {result['error']}", flush=True)
        results.append(result)
    measured = [r for r in results if r["audit_status"] == "measured"]
    summary = {"total_samples": len(results), "measured_samples": len(measured)}
    for kind in ("clean", "mixed"):
        word_count = sum(r[f"{kind}_word_count"] for r in measured)
        summary[f"{kind}_corpus_wer"] = (
            sum(r[f"{kind}_edits"] for r in measured) / word_count if word_count else None
        )
        summary[f"{kind}_exact_matches"] = sum(r[f"{kind}_edits"] == 0 for r in measured)
    summary["clipped_samples"] = sum(
        r["clean_clipped_samples"] + r["mixed_clipped_samples"] for r in measured
    )
    summary["max_snr_error_db"] = max(
        (r["snr_error_db"] for r in measured if r["snr_error_db"] is not None), default=None
    )
    report = {
        "evaluated_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "method": "Independent ASR + waveform analysis; no human or direct audio listening assessment.",
        "human_listening_performed": False,
        "accent_accuracy_validated": False,
        "subjective_tone_validated": False,
        "normalization": "lowercase, punctuation ignored, numeric 0..20 normalized to words",
        "asr_settings": settings,
        "pitch_settings": {
            "enabled": pitch,
            "method": "librosa.pyin",
            "sr": 16000,
            "fmin": 60,
            "fmax": 500,
            "frame_length": 1024,
            "hop_length": 160,
        },
        "limitations": [
            "ASR agreement is not proof of naturalness or perfect pronunciation.",
            "Duration/pitch differences cannot establish requested emotions or accents.",
            "Different preset/accent jobs also use different seeds; style effects are confounded with sampling.",
            "Default backgrounds are procedural approximations; real recordings were not evaluated.",
        ],
        "summary": summary,
        "samples": results,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = list(dict.fromkeys(key for row in results for key in row))
    if fields:
        with (output_dir / "samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    print(json.dumps(summary, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing metadata.csv and WAV files"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--pitch", action="store_true", help="Measure descriptive pYIN pitch statistics"
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    report = audit(args.input_dir, args.output, args.model, args.threads, args.pitch)
    return int(report["summary"]["measured_samples"] != report["summary"]["total_samples"])


if __name__ == "__main__":
    raise SystemExit(main())
