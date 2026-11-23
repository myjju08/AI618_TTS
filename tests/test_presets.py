import pytest

from scenario_tts.presets import ACCENTS, BACKGROUNDS, NOISE_LEVELS, PRESETS, build_instruction


def test_six_coherent_presets_and_two_accents():
    assert len(PRESETS) == 6
    assert set(ACCENTS) == {"american", "british"}
    assert PRESETS["station"].background == "station"
    assert PRESETS["urgent"].background == "street"
    assert PRESETS["documentary"].background == "none"
    assert all(p.background in BACKGROUNDS for p in PRESETS.values())
    assert NOISE_LEVELS == {"weak": 20.0, "medium": 10.0, "strong": 5.0}


def test_style_override_replaces_preset_but_keeps_accent_and_exact_words():
    result = build_instruction("urgent", "british", "Speak slowly and softly.")
    assert ACCENTS["british"] in result
    assert "Speak slowly and softly." in result
    assert PRESETS["urgent"].instruction not in result
    assert "Speak exactly the supplied text." in result
    assert PRESETS["urgent"].instruction in build_instruction("urgent", "american", "  ")


@pytest.mark.parametrize("preset,accent", [("unknown", "american"), ("cafe", "unknown")])
def test_unknown_settings_are_not_silently_accepted(preset, accent):
    with pytest.raises(ValueError):
        build_instruction(preset, accent)
