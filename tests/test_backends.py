from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_tts.backends import DemoBackend, QwenBackend


def test_demo_is_deterministic_non_speech():
    backend = DemoBackend()
    request = dict(text="The next train arrives soon.", speaker="Ryan", instruct="Calmly.", seed=42)
    first, sr = backend.generate(**request)
    second, _ = backend.generate(**request)
    assert "non-speech" in backend.model_id
    assert sr == 24000
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all() and first.size > sr
    assert np.max(np.abs(first)) < 1


def test_qwen_forwards_text_instruction_separately_and_loads_once(monkeypatch):
    events = {"loads": [], "calls": [], "seeds": []}
    waveform = np.sin(np.arange(2400) * 0.1).astype(np.float32)
    def generate(**kwargs):
        events["calls"].append(kwargs)
        return [waveform], 24000
    def load(*args, **kwargs):
        events["loads"].append((args, kwargs))
        return SimpleNamespace(generate_custom_voice=generate)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float32="float32", float16="float16", bfloat16="bfloat16",
        random=SimpleNamespace(fork_rng=lambda **kwargs: nullcontext()),
        inference_mode=lambda: nullcontext(), manual_seed=events["seeds"].append,
    )
    def importer(name):
        return {"torch": torch, "qwen_tts": SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=load))}[name]
    monkeypatch.setattr("scenario_tts.backends.importlib.import_module", importer)
    backend = QwenBackend(max_new_tokens=512)
    assert not events["loads"]  # Construction must not download model weights.
    kwargs = dict(text="Please leave your bag beside the door.", speaker="Ryan", instruct="Use a British accent.", seed=7)
    before = np.random.get_state()
    audio, sr = backend.generate(**kwargs)
    backend.generate(**{**kwargs, "seed": 8})
    assert len(events["loads"]) == 1
    assert events["loads"][0][1] == dict(device_map="cpu", dtype="float32", attn_implementation="eager")
    assert events["calls"][0] == dict(text=kwargs["text"], speaker="Ryan", language="English", instruct=kwargs["instruct"], max_new_tokens=512)
    assert events["seeds"] == [7, 8]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1])
    np.testing.assert_array_equal(audio, waveform)
    assert sr == 24000


def test_qwen_missing_dependencies_are_actionable(monkeypatch):
    def missing(name):
        raise ImportError(name)
    monkeypatch.setattr("scenario_tts.backends.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match="tts extra"):
        QwenBackend().generate(text="Hello.", speaker="Ryan", instruct="Calm.", seed=1)


@pytest.mark.parametrize("kwargs", [{"device": "mps"}, {"dtype": "int8"}, {"model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}, {"max_new_tokens": 0}])
def test_qwen_rejects_unsupported_configuration(kwargs):
    with pytest.raises(ValueError):
        QwenBackend(**kwargs)


def test_unavailable_cuda_fails_before_model_download(monkeypatch):
    def should_not_load(*args, **kwargs):
        raise AssertionError("must not download")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    modules = {"torch": torch, "qwen_tts": SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=should_not_load))}
    monkeypatch.setattr("scenario_tts.backends.importlib.import_module", modules.__getitem__)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        QwenBackend(device="cuda").generate(text="Hello.", speaker="Ryan", instruct="Calm.", seed=1)
