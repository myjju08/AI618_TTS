"""Independent QA math regressions; no speech models or optional DSP imports."""

import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_audit_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_audio.py"
    spec = importlib.util.spec_from_file_location("scenario_tts_audio_audit_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_audit_module()


@pytest.mark.parametrize(
    "reference,hypothesis,substitutions,deletions,insertions,expected_wer",
    [
        ("the next train", "the last train", 1, 0, 0, 1 / 3),
        ("the next train", "the train", 0, 1, 0, 1 / 3),
        ("the train", "the next train", 0, 0, 1, 1 / 2),
        ("the next train", "", 0, 3, 0, 1),
        ("the red train stops", "a red train", 1, 1, 0, 1 / 2),
        # WER is an edit count/reference length, not a probability capped at one.
        ("train", "a very late train arrives now", 0, 0, 5, 5),
    ],
)
def test_word_errors_count_edit_types_and_allow_wer_above_one(
    reference, hypothesis, substitutions, deletions, insertions, expected_wer
):
    result = audit.word_errors(reference, hypothesis)
    assert result["substitutions"] == substitutions
    assert result["deletions"] == deletions
    assert result["insertions"] == insertions
    assert result["edits"] == substitutions + deletions + insertions
    assert result["wer"] == pytest.approx(expected_wer)


def test_numeric_case_and_punctuation_normalization_avoids_false_word_errors():
    expected = "The next train will arrive in five minutes."
    recognized = "THE next train will arrive in 5 minutes!"
    result = audit.word_errors(expected, recognized)
    assert result["word_count"] == 8
    assert result["edits"] == 0
    assert result["wer"] == 0
    assert audit.normalize_words("It’s 0, 20, 21.") == ["it's", "zero", "twenty", "21"]


@pytest.mark.parametrize("reference", ["", "... !"])
def test_empty_reference_is_rejected(reference):
    with pytest.raises(ValueError, match="Reference must contain words"):
        audit.word_errors(reference, "hallucinated words")


def row_metadata(**overrides):
    row = {
        "mix_gain": "1",
        "lead_seconds": "0",
        "tail_seconds": "0",
        "background": "none",
        "snr_db": "20",
        "text": "The train arrives soon.",
    }
    return row | overrides


@pytest.mark.parametrize("gain", [1.0, 0.4])
def test_snr_is_reconstructed_from_audio_with_common_gain_and_excludes_padding(gain):
    # Construct signals independently of the application's mixing implementation.
    # Clean RMS is 0.2, in-window background RMS is 0.02 => 20 dB SNR.
    clean = np.array([0.2, -0.2, 0.2, -0.2])
    background = np.full(4, 0.02)
    # Deliberately louder noise outside speech would spoil a whole-file RMS test.
    mixed = gain * np.concatenate(([0.8, -0.8], clean + background, [0.7, -0.7]))
    row = row_metadata(
        mix_gain=str(gain), lead_seconds="0.5", tail_seconds="0.5", background="station"
    )
    result = audit.waveform_metrics(clean, mixed, 4, row)
    assert result["measured_snr_db"] == pytest.approx(20.0, abs=1e-10)
    assert result["snr_error_db"] == pytest.approx(0.0, abs=1e-10)
    assert result["speech_seconds"] == 1
    assert result["mixed_seconds"] == 2
    assert result["clean_rms_dbfs"] == pytest.approx(20 * np.log10(0.2))
    assert result["words_per_second_including_pauses"] == 4
    assert not result["clean_mixed_equal"]


def test_clipping_counts_include_exact_full_scale_and_report_true_peaks():
    clean = np.array([1.0, -1.0, 0.5, 0.0])
    mixed = np.array([1.25, -1.01, 1.0, 0.0])
    result = audit.waveform_metrics(clean, mixed, 4, row_metadata())
    assert result["clean_clipped_samples"] == 2
    assert result["mixed_clipped_samples"] == 3
    assert result["clean_peak"] == 1
    assert result["mixed_peak"] == 1.25


def test_length_mismatch_is_rejected_before_noise_subtraction():
    clean = np.full(4, 0.2)
    with pytest.raises(ValueError, match="Mixed length"):
        audit.waveform_metrics(
            clean, np.full(7, 0.1), 4, row_metadata(lead_seconds="0.5", tail_seconds="0.5")
        )


def test_no_background_equal_audio_has_no_snr_measurement():
    clean = np.array([0.1, -0.2, 0.3, -0.1])
    result = audit.waveform_metrics(clean, clean.copy(), 4, row_metadata())
    assert result["clean_mixed_equal"]
    assert result["measured_snr_db"] is None
    assert result["snr_error_db"] is None
    assert result["clean_clipped_samples"] == result["mixed_clipped_samples"] == 0


@pytest.mark.parametrize(
    "clean,mixed,match",
    [
        (np.array([]), np.array([0.1]), "nonempty mono"),
        (np.ones((4, 2)), np.ones(4), "nonempty mono"),
        (np.array([np.nan]), np.array([0.1]), "NaN/Inf"),
        (np.array([0.1]), np.array([np.inf]), "NaN/Inf"),
        (np.zeros(4), np.zeros(4), "silent"),
    ],
)
def test_invalid_waveforms_cannot_be_reported_as_successful_measurements(clean, mixed, match):
    with pytest.raises(ValueError, match=match):
        audit.waveform_metrics(clean, mixed, 4, row_metadata())


def test_declared_background_without_recoverable_noise_is_rejected():
    clean = np.full(4, 0.2)
    with pytest.raises(ValueError, match="recovered no noise"):
        audit.waveform_metrics(clean, clean.copy(), 4, row_metadata(background="station"))


def test_audio_paths_must_exist_and_stay_within_the_input_directory(tmp_path):
    directory = tmp_path / "batch"
    directory.mkdir()
    (directory / "clean").mkdir()
    valid = directory / "clean" / "valid.wav"
    valid.write_bytes(b"placeholder")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"placeholder")
    assert audit.contained_file(directory, "clean/valid.wav") == valid.resolve()
    for unsafe in ("../outside.wav", str(outside.resolve()), "missing.wav", "clean"):
        with pytest.raises(ValueError, match="Missing or unsafe"):
            audit.contained_file(directory, unsafe)


def test_audio_symlinks_cannot_escape_the_input_directory(tmp_path):
    directory = tmp_path / "batch"
    directory.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"placeholder")
    link = directory / "linked.wav"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Creating symlinks is unavailable on this host.")
    with pytest.raises(ValueError, match="Missing or unsafe"):
        audit.contained_file(directory, "linked.wav")


def test_module_and_numeric_audit_helpers_do_not_import_optional_ml_packages(monkeypatch):
    original_import = builtins.__import__
    forbidden = {"faster_whisper", "ctranslate2", "librosa", "torch", "qwen_tts"}

    def import_without_optional_models(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"Optional model/DSP dependency imported eagerly: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_optional_models)
    independent = load_audit_module()
    assert independent.word_errors("five minutes", "5 minutes")["wer"] == 0
    clean = np.array([0.1, -0.1])
    assert independent.waveform_metrics(clean, clean.copy(), 2, row_metadata())["clean_mixed_equal"]
