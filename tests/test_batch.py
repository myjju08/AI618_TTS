import io
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from scenario_tts.backends import DemoBackend
from scenario_tts.batch import (
    Sentence,
    create_jobs,
    generate_batch,
    load_manifest,
    read_sentences,
    regenerate_job,
)
from scenario_tts.cli import main
from scenario_tts.presets import ACCENTS, PRESETS


def sentences():
    return [Sentence(f"{i:03}", f"Sentence number {i} is fixed.") for i in range(1, 6)]


def test_csv_preserves_ids_and_quoted_text():
    result = read_sentences(io.StringIO('\ufeffsentence_id,text\n001,"Hello, friend."\n002,"Line one\nline two"\n'))
    assert result == [Sentence("001", "Hello, friend."), Sentence("002", "Line one\nline two")]


@pytest.mark.parametrize("csv_text", [
    "id,text\n1,Hello", "sentence_id,text\n", "sentence_id,text\n../a,Hello", "sentence_id,text\n1, ",
    "sentence_id,text\n1,Hello\n1,Again", "sentence_id,text\n1,Hello,extra", "sentence_id,text\n1",
    "sentence_id,text,text\n1,Hello,Again",
])
def test_invalid_csv_rejected(csv_text):
    with pytest.raises(ValueError):
        read_sentences(io.StringIO(csv_text))


def test_default_matrix_has_60_stable_jobs_and_separate_instructions():
    jobs = create_jobs(sentences(), PRESETS, ACCENTS)
    assert len(jobs) == len({job.sample_id for job in jobs}) == 60
    assert all(job.text == sentences()[int(job.sentence_id) - 1].text for job in jobs)
    assert all(job.instruct and job.text not in job.instruct for job in jobs)
    reverse = create_jobs(reversed(sentences()), reversed(PRESETS), reversed(ACCENTS))
    assert {job.sample_id: job.seed for job in jobs} == {job.sample_id: job.seed for job in reverse}


def test_resume_regeneration_and_backend_separation(tmp_path):
    class CountingBackend(DemoBackend):
        calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            return super().generate(**kwargs)
    backend = CountingBackend()
    job = create_jobs(sentences()[:1], ["documentary"], ["american"])[0]
    first = generate_batch([job], tmp_path, backend)[0]
    assert first["status"] == "success"
    first_bytes = (tmp_path / first["mixed_path"]).read_bytes()
    generate_batch([job], tmp_path, backend)
    assert backend.calls == 1
    second = generate_batch([regenerate_job(job)], tmp_path, backend)[0]
    assert second["sample_id"] != first["sample_id"]
    assert len(load_manifest(tmp_path)) == 2
    assert (tmp_path / first["mixed_path"]).read_bytes() == first_bytes
    backend.name = "different-engine"
    third = generate_batch([job], tmp_path, backend)[0]
    assert third["artifact_id"] != first["artifact_id"]
    assert len(load_manifest(tmp_path)) == 3


def test_corrupt_file_is_regenerated(tmp_path):
    job = create_jobs(sentences()[:1], ["documentary"], ["american"])[0]
    first = generate_batch([job], tmp_path, DemoBackend())[0]
    (tmp_path / first["mixed_path"]).write_bytes(b"broken")
    result = generate_batch([job], tmp_path, DemoBackend())[0]
    assert result["status"] == "success"
    assert sf.info(tmp_path / result["mixed_path"]).frames > 0
    assert len(load_manifest(tmp_path)) == 1


def test_partial_failure_records_error_and_continues(tmp_path):
    class FailsOnce(DemoBackend):
        calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic failure")
            return super().generate(**kwargs)
    jobs = create_jobs(sentences()[:2], ["documentary"], ["american"])
    callbacks = []
    rows = generate_batch(jobs, tmp_path, FailsOnce(), progress=lambda done, total, row: callbacks.append((done, total)))
    assert [row["status"] for row in rows] == ["failed", "success"]
    assert "synthetic failure" in rows[0]["error"]
    assert callbacks == [(1, 2), (2, 2)]
    assert len(load_manifest(tmp_path)) == 2
    assert not (tmp_path / ".batch.lock").exists()


def test_custom_background_content_changes_cache_identity(tmp_path):
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    path = backgrounds / "cafe.wav"
    sf.write(path, np.sin(np.arange(48000) * 0.2) * 0.1, 24000)
    job = create_jobs(sentences()[:1], ["cafe"], ["american"])[0]
    first = generate_batch([job], tmp_path / "output", DemoBackend(), backgrounds)[0]
    sf.write(path, np.sin(np.arange(48000) * 0.3) * 0.1, 24000)
    second = generate_batch([job], tmp_path / "output", DemoBackend(), backgrounds)[0]
    assert first["status"] == second["status"] == "success"
    assert first["artifact_id"] != second["artifact_id"]


def test_unsafe_direct_job_cannot_escape_output(tmp_path):
    job = create_jobs(sentences()[:1], ["documentary"], ["american"])[0]
    with pytest.raises(ValueError, match="Unsafe sample_id"):
        generate_batch([replace(job, sample_id="../escape")], tmp_path, DemoBackend())
    with pytest.raises(ValueError, match="Unknown background"):
        generate_batch([replace(job, background="../../private")], tmp_path, DemoBackend(), tmp_path)


def test_cli_plan_and_failure_exit_codes(tmp_path, monkeypatch):
    source = tmp_path / "sentences.csv"
    source.write_text("sentence_id,text\n001,The next train arrives soon.\n", encoding="utf-8")
    assert main(["plan", str(source), "--output", str(tmp_path / "plan.csv")]) == 0
    assert len((tmp_path / "plan.csv").read_text(encoding="utf-8-sig").splitlines()) > 12
    assert main(["generate", str(source), "--output", str(tmp_path / "demo"), "--backend", "demo", "--presets", "documentary", "--accents", "american"]) == 0
    def fail(self, **kwargs):
        raise RuntimeError("No GPU test")
    monkeypatch.setattr("scenario_tts.cli.QwenBackend.prepare", lambda self: None)
    monkeypatch.setattr("scenario_tts.cli.QwenBackend.generate", fail)
    assert main(["generate", str(source), "--output", str(tmp_path / "qwen"), "--presets", "documentary", "--accents", "american"]) == 1


def test_preparation_once_per_batch_retry_later_and_no_load_on_cached_resume(tmp_path):
    class PreparedBackend(DemoBackend):
        preparations = 0
        generations = 0
        offline = True

        def prepare(self):
            self.preparations += 1
            if self.offline:
                raise RuntimeError("Model download unavailable")

        def generate(self, **kwargs):
            self.generations += 1
            return super().generate(**kwargs)

    backend = PreparedBackend()
    jobs = create_jobs(sentences()[:2], ["documentary"], ["american"])
    failed = generate_batch(jobs, tmp_path, backend)
    assert backend.preparations == 1 and backend.generations == 0
    assert all(row["status"] == "failed" and "Model download unavailable" in row["error"] for row in failed)

    backend.offline = False
    succeeded = generate_batch(jobs, tmp_path, backend)
    assert backend.preparations == 2 and backend.generations == 2
    assert all(row["status"] == "success" for row in succeeded)
    assert len(load_manifest(tmp_path)) == 2

    backend.offline = True  # Cached output must remain usable without network/model loading.
    resumed = generate_batch(jobs, tmp_path, backend)
    assert backend.preparations == 2 and backend.generations == 2
    assert all(row["status"] == "success" for row in resumed)


def test_missing_background_directory_fails_before_preparation(tmp_path):
    class MustNotLoad(DemoBackend):
        def prepare(self):
            raise AssertionError("Bad background directory must be rejected first")

    jobs = create_jobs(sentences()[:1], ["cafe"], ["american"])
    with pytest.raises(ValueError, match="Background directory"):
        generate_batch(jobs, tmp_path / "output", MustNotLoad(), tmp_path / "typo")
    assert not (tmp_path / "output").exists()


def test_background_read_error_is_per_sample_and_other_samples_continue(tmp_path, monkeypatch):
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    bad_path = backgrounds / "cafe.wav"
    bad_path.write_bytes(b"unreadable")
    from pathlib import Path
    original = Path.read_bytes

    def read_bytes(path):
        if path == bad_path:
            raise PermissionError("test access denied")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    jobs = create_jobs(sentences()[:1], ["cafe", "documentary"], ["american"])
    rows = generate_batch(jobs, tmp_path / "output", DemoBackend(), backgrounds)
    assert [row["status"] for row in rows] == ["failed", "success"]
    assert "test access denied" in rows[0]["error"]
