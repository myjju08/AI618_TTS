import numpy as np
import pytest
import soundfile as sf

from scenario_tts.audio import _fit_background, mix_audio

RATE = 8000


def speech_fixture(amplitude=0.25):
    time = np.arange(RATE * 2) / RATE
    return amplitude * np.sin(2 * np.pi * 173.0 * time)


def rms(values):
    return np.sqrt(np.mean(values ** 2))


@pytest.mark.parametrize("snr", [20.0, 10.0, 5.0])
def test_measured_snr_lead_tail_fades_and_safe_peaks(snr):
    speech = speech_fixture(amplitude=0.95)
    result = mix_audio(speech, RATE, "cafe", snr_db=snr, seed=41)
    assert result.audio.size == speech.size + RATE
    lead = RATE // 2
    gain = result.metadata["mix_gain"]
    recovered_noise = result.audio[lead : lead + speech.size].astype(float) - speech * gain
    measured = 20 * np.log10(rms(speech * gain) / rms(recovered_noise))
    assert measured == pytest.approx(snr, abs=0.001)
    assert result.metadata["achieved_snr_db"] == pytest.approx(snr, abs=1e-9)
    assert np.max(np.abs(result.audio)) <= 0.980001
    assert result.audio[0] == 0
    assert result.audio[-1] == 0
    assert rms(result.audio[100:lead]) > 0
    assert "approximation" in result.metadata["background_source"]


def test_same_seed_repeats_mix_and_different_seed_changes_it():
    speech = speech_fixture()
    a = mix_audio(speech, RATE, "street", seed=9)
    b = mix_audio(speech, RATE, "street", seed=9)
    c = mix_audio(speech, RATE, "street", seed=10)
    np.testing.assert_array_equal(a.audio, b.audio)
    assert a.metadata == b.metadata
    assert not np.array_equal(a.audio, c.audio)


def test_none_has_no_padding_or_noise_and_does_not_mutate_input():
    speech = speech_fixture().astype(np.float32)
    original = speech.copy()
    result = mix_audio(speech, RATE)
    np.testing.assert_array_equal(result.audio, speech)
    assert result.audio is not speech
    np.testing.assert_array_equal(speech, original)
    assert result.metadata["target_snr_db"] is None
    assert result.metadata["lead_seconds"] == 0


def test_peak_safeguard_for_clean_output():
    result = mix_audio(speech_fixture(amplitude=1.5), RATE)
    assert result.metadata["mix_gain"] < 1
    assert np.max(np.abs(result.audio)) <= 0.980001


def test_short_stereo_recording_is_resampled_and_looped(tmp_path):
    time = np.arange(2400) / 12000
    stereo = np.column_stack((0.2 * np.sin(2 * np.pi * 210 * time), 0.1 * np.cos(2 * np.pi * 110 * time)))
    source = tmp_path / "station.wav"
    sf.write(source, stereo, 12000, subtype="FLOAT")
    speech = speech_fixture()
    result = mix_audio(speech, RATE, "station", background_dir=tmp_path, seed=12)
    assert result.audio.size == speech.size + RATE
    assert result.metadata["background_source"] == str(source.resolve())
    assert 0 <= result.metadata["background_offset_seconds"] < 0.2
    lead = RATE // 2
    gain = result.metadata["mix_gain"]
    noise = result.audio[lead : lead + speech.size] - speech * gain
    assert 20 * np.log10(rms(speech * gain) / rms(noise)) == pytest.approx(20, abs=0.001)
    assert np.isfinite(result.audio).all()


def test_silent_recording_is_rejected(tmp_path):
    sf.write(tmp_path / "room.wav", np.zeros(RATE), RATE)
    with pytest.raises(ValueError, match="silent"):
        mix_audio(speech_fixture(), RATE, "room", background_dir=tmp_path)


def test_random_offset_does_not_reintroduce_a_click_inside_the_loop():
    # A ramp has a severe discontinuity when naively tiled or rotated.
    source = np.linspace(-1.0, 1.0, 1000)
    for seed in range(5):
        result, offset = _fit_background(source, 8000, RATE, np.random.default_rng(seed))
        assert 0 <= offset < source.size
        assert np.max(np.abs(np.diff(result))) < 0.01


@pytest.mark.parametrize("speech", [np.array([]), np.zeros(20), np.array([np.nan]), np.array([np.inf])])
def test_invalid_speech_fails_clearly(speech):
    with pytest.raises(ValueError):
        mix_audio(speech, RATE, "cafe")


@pytest.mark.parametrize("kwargs", [{"sample_rate": 0}, {"sample_rate": 8000.5}, {"snr_db": np.nan}, {"snr_db": 999}, {"lead_seconds": -1}, {"seed": -2}, {"background": "../../private"}])
def test_invalid_mix_parameters_fail_clearly(kwargs):
    values = {"sample_rate": RATE, "background": "room"} | kwargs
    with pytest.raises(ValueError):
        mix_audio(speech_fixture(), **values)
