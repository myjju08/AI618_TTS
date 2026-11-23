"""Speech engines. The demo backend emits test tones, never simulated speech."""

from __future__ import annotations

import importlib
import random
import sys
import threading
from importlib import metadata
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKERS = ("Ryan", "Aiden")
_INFERENCE_LOCK = threading.RLock()


class Backend(Protocol):
    name: str
    model_id: str

    def generate(self, *, text: str, speaker: str, instruct: str, seed: int) -> tuple[np.ndarray, int]: ...


def _validate_request(text: str, speaker: str, instruct: str, seed: int) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Speech text must not be empty.")
    if speaker not in SPEAKERS:
        raise ValueError(f"Speaker must be one of {SPEAKERS}.")
    if not isinstance(instruct, str):
        raise ValueError("instruct must be a string separate from text.")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 4294967295.")


class DemoBackend:
    """Deterministic NON-SPEECH test signal for testing the file/mixing pipeline."""

    name = "demo"
    model_id = "non-speech-test-tone-v1"

    def generate(self, *, text: str, speaker: str, instruct: str, seed: int) -> tuple[np.ndarray, int]:
        _validate_request(text, speaker, instruct, seed)
        sr = 24000
        duration = max(1.0, min(8.0, len(text.split()) * 0.25))
        t = np.arange(round(duration * sr), dtype=np.float64) / sr
        frequency = 180.0 + np.random.default_rng(seed).uniform(0, 120)
        envelope = np.minimum(1.0, t / 0.04) * np.minimum(1.0, (duration - t) / 0.04)
        envelope *= 0.6 + 0.4 * np.sin(2 * np.pi * 2.8 * t) ** 2
        signal = 0.2 * envelope * (np.sin(2 * np.pi * frequency * t) + 0.2 * np.sin(4 * np.pi * frequency * t))
        return signal.astype(np.float32), sr


class QwenBackend:
    """Lazy-loaded 1.7B CustomVoice adapter; no network requests on construction."""

    name = "qwen"

    def __init__(self, device: str = "auto", model_id: str = DEFAULT_MODEL, dtype: str = "auto", attn_implementation: str = "auto", max_new_tokens: int = 2048):
        if device not in ("auto", "cpu", "cuda") and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("device must be auto, cpu, cuda, or cuda:N.")
        if dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("dtype must be auto, float32, float16, or bfloat16.")
        if attn_implementation not in ("auto", "sdpa", "eager", "flash_attention_2"):
            raise ValueError("Unsupported attention implementation.")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        if not model_id or "0.6B" in model_id or "VoiceDesign" in model_id or model_id.endswith("-Base"):
            raise ValueError("Use the 1.7B CustomVoice model; other model variants do not implement this style-control contract.")
        self.device = device
        self.model_id = model_id
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._torch = None
        self.resolved_device = None
        self.resolved_dtype = None
        self.resolved_attention = None

    @property
    def config(self) -> dict:
        versions = {"python": sys.version.split()[0]}
        for package in ("qwen-tts", "torch", "transformers", "numpy"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = "not-installed"
        return {"device": self.device, "dtype": self.dtype, "attention": self.attn_implementation,
                "max_new_tokens": self.max_new_tokens, "runtime_versions": versions}

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            qwen_tts = importlib.import_module("qwen_tts")
        except ImportError as exc:
            raise RuntimeError("Qwen dependencies are missing. Install the project with its tts extra: pip install -e '.[tts]'.") from exc
        device = self.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device = "cuda:0"
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but PyTorch cannot access an NVIDIA GPU. Install CUDA-enabled PyTorch or choose cpu.")
            if int(device.split(":")[1]) >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {device} does not exist.")
        if self.attn_implementation == "flash_attention_2" and device == "cpu":
            raise ValueError("FlashAttention 2 requires a supported GPU; use sdpa or eager on CPU.")
        dtype = self.dtype
        if dtype == "auto":
            if device == "cpu":
                dtype = "float32"
            else:
                with torch.cuda.device(device):
                    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        if device == "cpu" and dtype == "float16":
            raise ValueError("Use float32 or bfloat16 for CPU inference.")
        attention = self.attn_implementation if self.attn_implementation != "auto" else ("eager" if device == "cpu" else "sdpa")
        try:
            self._model = qwen_tts.Qwen3TTSModel.from_pretrained(
                self.model_id, device_map=device, dtype=getattr(torch, dtype), attn_implementation=attention
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load {self.model_id} on {device}: {exc}. Check Hugging Face access, available RAM/VRAM, and the qwen installation.") from exc
        self._torch = torch
        self.resolved_device = device
        self.resolved_dtype = dtype
        self.resolved_attention = attention

    def prepare(self) -> None:
        """Load once before a batch; a failed attempt can be retried by a later batch."""
        with _INFERENCE_LOCK:
            self._load()

    def generate(self, *, text: str, speaker: str, instruct: str, seed: int) -> tuple[np.ndarray, int]:
        _validate_request(text, speaker, instruct, seed)
        # Both model initialization and RNG-dependent inference are serialized across
        # instances: torch and Python random state are process-global.
        with _INFERENCE_LOCK:
            self._load()
            torch = self._torch
            python_state, numpy_state = random.getstate(), np.random.get_state()
            # manual_seed also changes all CUDA generators, so preserve every GPU
            # RNG even if inference uses CPU or a single selected GPU.
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            try:
                with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if cuda_devices:
                        torch.cuda.manual_seed_all(seed)
                    wavs, sample_rate = self._model.generate_custom_voice(
                        text=text, language="English", speaker=speaker, instruct=instruct,
                        max_new_tokens=self.max_new_tokens,
                    )
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
            if len(wavs) != 1:
                raise RuntimeError("Qwen returned an unexpected number of waveforms.")
            waveform = wavs[0]
            if hasattr(waveform, "detach"):
                waveform = waveform.detach().cpu().numpy()
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                raise RuntimeError("Qwen returned an empty, non-finite, or non-mono waveform.")
            if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
                raise RuntimeError("Qwen returned an invalid sample rate.")
            return waveform, int(sample_rate)
