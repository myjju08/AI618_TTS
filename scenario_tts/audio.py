"""Deterministic ambience mixing at a specified speech-window RMS SNR.

Built-in sound beds are procedural approximations for testing the pipeline, not
recordings of real places. Put licensed ``cafe.wav``, ``station.wav``, etc. in a
background directory to substitute actual ambience without changing code.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

from .presets import BACKGROUNDS


@dataclass(frozen=True)
class MixResult:
    audio: np.ndarray
    metadata: dict


def _mono_audio(audio: np.ndarray, name: str) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2 and 1 <= audio.shape[1] <= 8:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"{name} must be a non-empty mono or channel-last audio array.")
    if not np.isfinite(audio).all():
        raise ValueError(f"{name} contains non-finite audio values.")
    if _rms(audio) <= 1e-12:
        raise ValueError(f"{name} is silent; a meaningful SNR cannot be computed.")
    return audio


def _rms(audio: np.ndarray) -> float:
    # Scaling first avoids overflow for otherwise finite input values.
    peak = float(np.max(np.abs(audio)))
    return peak * float(np.sqrt(np.mean(np.square(audio / peak)))) if peak else 0.0


def _lowpass(noise: np.ndarray, frequency: float, sample_rate: int) -> np.ndarray:
    cutoff = min(frequency, sample_rate * 0.45)
    return sosfilt(butter(2, cutoff, fs=sample_rate, output="sos"), noise)


def _procedural_ambience(kind: str, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Make modest noise textures; none of these are simulated intelligible speech."""
    count = max(16, sample_rate * 12)
    time = np.arange(count, dtype=np.float64) / sample_rate
    white = rng.normal(0.0, 1.0, count)
    rumble = _lowpass(white, 170.0, sample_rate)
    diffuse = _lowpass(rng.normal(0.0, 1.0, count), 2400.0, sample_rate)
    bed = 0.25 * rumble + 0.025 * white
    if kind in {"cafe", "crowd", "station"}:
        # Slowly moving diffuse noise gives a sense of activity without words.
        for frequency in (0.29, 0.53, 0.83):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            modulation = 0.5 + 0.5 * np.sin(2.0 * np.pi * frequency * time + phase)
            bed += 0.13 * diffuse * modulation
    if kind == "cafe":
        for _ in range(8):
            onset = rng.uniform(0.0, 11.5)
            elapsed = time - onset
            envelope = np.exp(-np.maximum(elapsed, 0.0) * 18.0) * (elapsed >= 0.0)
            frequency = min(rng.uniform(1700.0, 3100.0), sample_rate * 0.4)
            bed += 0.04 * envelope * np.sin(2.0 * np.pi * frequency * elapsed)
    elif kind == "station":
        bed += 0.35 * rumble * (0.6 + 0.4 * np.sin(time * 0.9) ** 2)
    elif kind == "street":
        for center in (2.0, 6.5, 10.0):
            whoosh = np.exp(-0.5 * ((time - center) / 1.0) ** 2)
            bed += (0.6 * rumble + 0.12 * diffuse) * whoosh
    elif kind == "room":
        bed = 0.2 * rumble + 0.01 * white
    return bed


def _fit_background(
    source: np.ndarray, count: int, sample_rate: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Crop long recordings; loop short ones using 50 ms overlap crossfades."""
    if source.size >= count:
        offset = int(rng.integers(0, source.size - count + 1))
        return source[offset : offset + count].copy(), offset
    overlap = min(round(0.05 * sample_rate), source.size // 4)
    if overlap < 2:
        offset = int(rng.integers(0, source.size))
        return np.resize(np.roll(source, -offset), count), offset
    # Construct a seamless period *before* choosing an offset. Rotating the raw
    # source first would introduce an un-crossfaded internal end-to-start seam.
    fade_in = np.linspace(0.0, 1.0, overlap)
    cycle = source[overlap:].copy()
    cycle[-overlap:] = source[-overlap:] * (1.0 - fade_in) + source[:overlap] * fade_in
    offset = int(rng.integers(0, cycle.size))
    return np.resize(np.roll(cycle, -offset), count), offset + overlap


def mix_audio(
    speech: np.ndarray,
    sample_rate: int,
    background: str = "none",
    snr_db: float = 20.0,
    seed: int = 0,
    background_dir: Path | str | None = None,
    lead_seconds: float = 0.5,
    tail_seconds: float = 0.5,
    fade_seconds: float = 0.2,
) -> MixResult:
    """Mix mono audio, with SNR measured over the complete speech window.

    Internal pauses count toward speech RMS; this is not a standardized active-
    speech SNR measure or a claim about real-world sound pressure. Background
    fades are applied before measuring its RMS. A common gain then limits the
    mixture to a 0.98 peak, preserving SNR. ``none`` adds no padding or fades and
    copies speech, applying only the same peak safeguard if necessary.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    if background not in BACKGROUNDS:
        raise ValueError(f"Unknown background: {background!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    if not np.isfinite(snr_db) or not -60.0 <= snr_db <= 120.0:
        raise ValueError("SNR must be finite and between -60 and 120 dB.")
    for name, value in (("lead_seconds", lead_seconds), ("tail_seconds", tail_seconds), ("fade_seconds", fade_seconds)):
        if not np.isfinite(value) or not 0.0 <= value <= 60.0:
            raise ValueError(f"{name} must be finite and between 0 and 60 seconds.")
    speech = _mono_audio(speech, "Speech")
    metadata = {
        "background_source": "none",
        "background_offset_seconds": 0.0,
        "target_snr_db": None,
        "achieved_snr_db": None,
        "mix_gain": 1.0,
        "lead_seconds": 0.0,
        "tail_seconds": 0.0,
        "fade_seconds": 0.0,
    }
    if background == "none":
        gain = min(1.0, 0.98 / float(np.max(np.abs(speech))))
        metadata["mix_gain"] = gain
        return MixResult((speech * gain).astype(np.float32), metadata)

    lead = round(lead_seconds * sample_rate)
    tail = round(tail_seconds * sample_rate)
    count = lead + speech.size + tail
    rng = np.random.default_rng(seed)
    source_path = Path(background_dir) / f"{background}.wav" if background_dir is not None else None
    if source_path is not None and source_path.is_file():
        source, source_rate = sf.read(source_path, dtype="float64", always_2d=False)
        source = _mono_audio(source, "Background")
        if source_rate != sample_rate:
            divisor = math.gcd(sample_rate, source_rate)
            source = resample_poly(source, sample_rate // divisor, source_rate // divisor)
        metadata["background_source"] = str(source_path.resolve())
    else:
        source = _procedural_ambience(background, sample_rate, rng)
        metadata["background_source"] = f"procedural:{background} (approximation)"
    noise, offset = _fit_background(source, count, sample_rate, rng)
    fade = min(round(fade_seconds * sample_rate), count // 2)
    if fade >= 2:
        ramp = np.linspace(0.0, 1.0, fade)
        noise[:fade] *= ramp
        noise[-fade:] *= ramp[::-1]
    window = slice(lead, lead + speech.size)
    noise_rms = _rms(noise[window])
    if noise_rms <= 1e-12:
        raise ValueError("The selected background segment is silent during speech.")
    speech_rms = _rms(speech)
    noise *= speech_rms / (10.0 ** (snr_db / 20.0) * noise_rms)
    mixed = noise.copy()
    mixed[window] += speech
    if not np.isfinite(mixed).all():
        raise ValueError("Input audio amplitude is too large to mix safely.")
    gain = min(1.0, 0.98 / float(np.max(np.abs(mixed))))
    achieved_snr = 20.0 * math.log10(speech_rms / _rms(noise[window]))
    metadata.update(
        background_offset_seconds=offset / sample_rate,
        target_snr_db=float(snr_db),
        achieved_snr_db=achieved_snr,
        mix_gain=gain,
        lead_seconds=lead / sample_rate,
        tail_seconds=tail / sample_rate,
        fade_seconds=fade / sample_rate if fade >= 2 else 0.0,
    )
    return MixResult((mixed * gain).astype(np.float32), metadata)
