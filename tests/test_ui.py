"""Regression coverage for UI workflows without downloading a speech model."""

import csv
from pathlib import Path

import gradio as gr
import pytest

from scenario_tts import ui


@pytest.fixture
def plan_args():
    return [
        [{"sentence_id": "001", "text": "The next train will arrive in five minutes."}],
        ["001"],
        ["cafe"],
        ["american"],
        "Ryan",
        "",
        "preset",
        "preset",
        20,
        42,
    ]


def test_generate_and_regenerate_original_twice_preserves_every_variation(
    tmp_path, monkeypatch, plan_args
):
    monkeypatch.setattr(ui, "OUTPUTS", tmp_path)
    result = ui._generate(*plan_args, "demo", "auto", "")
    state = result[0]
    original = state["rows"][0]
    assert original["status"] == "success"
    assert original["sentence_id"] == "001"
    assert original["backend"] == "demo"
    output_dir = Path(state["output_dir"])
    assert output_dir.parent == tmp_path
    original_bytes = {
        key: (output_dir / original[key]).read_bytes() for key in ("clean_path", "mixed_path")
    }
    assert Path(result[4]).is_file()
    assert Path(result[5]).is_file()
    assert Path(result[6]).is_file()

    first = ui._regenerate(state, original["artifact_id"])
    second = ui._regenerate(first[0], original["artifact_id"])
    rows = second[0]["rows"]
    assert len(rows) == 3
    assert len({row["sample_id"] for row in rows}) == 3
    assert len({row["seed"] for row in rows}) == 3
    assert len({row["artifact_id"] for row in rows}) == 3
    assert all(row["status"] == "success" for row in rows)
    for key, content in original_bytes.items():
        assert (output_dir / original[key]).read_bytes() == content
    with (output_dir / "metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3

    with pytest.raises(gr.Error, match="Demo"):
        ui._save_review(second[0], original["artifact_id"], "yes", "yes", "yes", "yes", "")
    assert not (output_dir / "reviews.csv").exists()


def test_review_restores_saved_values_and_updates_without_duplicate_rows(tmp_path):
    # This fixture exercises review persistence only; it is not generated Qwen audio.
    state = {
        "output_dir": str(tmp_path),
        "rows": [
            {
                "artifact_id": "review-fixture",
                "sample_id": "sample-fixture",
                "backend": "qwen",
                "status": "success",
            }
        ],
    }
    assert ui._load_review(state, "review-fixture")[:4] == ("unchecked",) * 4
    result = ui._save_review(
        state, "review-fixture", "yes", "no", "unchecked", "yes", "악센트 추가 확인"
    )
    assert Path(result[1]).is_file()
    restored = ui._load_review(state, "review-fixture")
    assert restored[:5] == ("yes", "no", "unchecked", "yes", "악센트 추가 확인")
    assert restored[5]

    ui._save_review(state, "review-fixture", "yes", "yes", "no", "yes", "재검수")
    with Path(result[1]).open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 1
    assert records[0]["notes"] == "재검수"
    assert ui._load_review(state, "review-fixture")[:5] == ("yes", "yes", "no", "yes", "재검수")
    assert ui._load_review(state, "unreviewed")[:4] == ("unchecked",) * 4


@pytest.mark.parametrize(
    ("index", "value", "message"),
    [
        (1, [], "문장"),
        (2, [], "상황"),
        (3, [], "악센트"),
        (9, None, "seed"),
        (9, float("inf"), "seed"),
        (9, float("nan"), "seed"),
        (9, 1.5, "seed"),
    ],
)
def test_invalid_selection_and_seed_have_friendly_errors(plan_args, index, value, message):
    plan_args[index] = value
    with pytest.raises(gr.Error, match=message):
        ui._plan(*plan_args)


@pytest.mark.parametrize("value", [None, float("inf"), float("nan")])
def test_invalid_custom_snr_has_friendly_error(plan_args, value):
    plan_args[7] = "custom"
    plan_args[8] = value
    with pytest.raises(gr.Error, match="SNR"):
        ui._plan(*plan_args)


@pytest.mark.parametrize(
    "contents",
    ["wrong,text\n001,Hello\n", "sentence_id,text\n001,Hello\n001,Again\n"],
)
def test_malformed_csv_is_reported_as_gradio_error(tmp_path, contents):
    path = tmp_path / "bad.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(gr.Error):
        ui._load_csv(str(path))


def test_csv_reload_preserves_string_ids_and_selects_all(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("sentence_id,text\n001,Hello.\n007,Goodbye.\n", encoding="utf-8")
    records, checkbox, _ = ui._load_csv(str(path))
    assert [record["sentence_id"] for record in records] == ["001", "007"]
    assert checkbox.value == ["001", "007"]
