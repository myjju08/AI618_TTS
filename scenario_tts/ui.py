"""Local Gradio interface for planning, generating and reviewing scenario speech."""

from __future__ import annotations

import csv
import math
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr

from .backends import DemoBackend, QwenBackend
from .batch import (
    Job,
    Sentence,
    create_jobs,
    generate_batch,
    load_manifest,
    read_sentences,
    regenerate_job,
)
from .presets import ACCENTS, BACKGROUNDS, NOISE_LEVELS, PRESETS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "sentences.csv"
OUTPUTS = PROJECT_ROOT / "outputs"
_REVIEW_LOCK = threading.Lock()
PLAN_HEADERS = ["문장 ID", "문장", "상황", "악센트", "화자", "seed", "배경", "SNR (dB)", "instruct"]
RESULT_HEADERS = ["샘플", "문장 ID", "상황", "악센트", "seed", "상태", "배경 출처", "오류"]
REVIEW_FIELDS = [
    "artifact_id",
    "sample_id",
    "backend",
    "reviewed_at",
    "text_correct",
    "tone_distinct",
    "accent_audible",
    "noise_acceptable",
    "notes",
]


@lru_cache(maxsize=4)
def _backend(name: str, device: str):
    if name == "demo":
        return DemoBackend()
    if name == "qwen":
        return QwenBackend(device=device)
    raise ValueError(f"알 수 없는 backend: {name}")


def _sentence_choices(sentences: list[Sentence]) -> list[tuple[str, str]]:
    return [(f"{s.sentence_id} · {s.text}", s.sentence_id) for s in sentences]


def _load_csv(path: str | None):
    try:
        sentences = read_sentences(path or DEFAULT_CSV)
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc
    return (
        [asdict(s) for s in sentences],
        gr.CheckboxGroup(
            choices=_sentence_choices(sentences), value=[s.sentence_id for s in sentences]
        ),
        f"{len(sentences)}개 문장을 읽었습니다. 생성할 문장을 선택하세요.",
    )


def _plan(
    sentence_records,
    selected_ids,
    presets,
    accents,
    speaker,
    tone,
    background,
    noise_level,
    custom_snr,
    seed,
) -> list[Job]:
    selected = set(selected_ids or [])
    sentences = [Sentence(**s) for s in sentence_records if s["sentence_id"] in selected]
    if not sentences:
        raise gr.Error("문장을 하나 이상 선택하세요.")
    if not presets or not accents:
        raise gr.Error("상황과 악센트를 각각 하나 이상 선택하세요.")
    if seed is None or not math.isfinite(seed) or int(seed) != seed:
        raise gr.Error("seed는 정수로 입력하세요.")
    if noise_level == "custom" and (custom_snr is None or not math.isfinite(custom_snr)):
        raise gr.Error("직접 입력 SNR은 유한한 숫자로 입력하세요.")
    snr = (
        None
        if noise_level == "preset"
        else (float(custom_snr) if noise_level == "custom" else NOISE_LEVELS[noise_level])
    )
    try:
        return create_jobs(
            sentences,
            presets,
            accents,
            speaker=speaker,
            base_seed=int(seed),
            tone=tone.strip() or None,
            background=None if background == "preset" else background,
            snr_db=snr,
        )
    except (ValueError, TypeError) as exc:
        raise gr.Error(str(exc)) from exc


def _preview(*args):
    jobs = _plan(*args)
    rows = [
        [
            j.sentence_id,
            j.text,
            j.preset,
            j.accent,
            j.speaker,
            j.seed,
            j.background,
            j.snr_db,
            j.instruct,
        ]
        for j in jobs
    ]
    return f"**{len(jobs)}개 조합** · 문장은 그대로 유지하고 아래 지시만 바꿉니다.", rows


def _result_rows(rows: list[dict]) -> list[list[Any]]:
    return [
        [
            r["artifact_id"],
            r["sentence_id"],
            r["preset"],
            r["accent"],
            r["seed"],
            r["status"],
            r.get("background_source", ""),
            r.get("error", ""),
        ]
        for r in rows
    ]


def _sample_choices(rows: list[dict]) -> list[tuple[str, str]]:
    return [
        (
            f"{r['sentence_id']} · {r['preset']} · {r['accent']} · seed {r['seed']} · {r['status']} ({r['artifact_id']})",
            r["artifact_id"],
        )
        for r in rows
    ]


def _sample(state: dict, artifact_id: str | None):
    if not state or not artifact_id:
        return None, None, "샘플을 선택하세요."
    row = next((r for r in state["rows"] if r["artifact_id"] == artifact_id), None)
    if row is None:
        return None, None, "선택한 샘플을 찾지 못했습니다."
    output_dir = Path(state["output_dir"])
    paths = []
    for key in ("clean_path", "mixed_path"):
        path = (output_dir / row.get(key, "")).resolve()
        paths.append(str(path) if path.is_file() and output_dir.resolve() in path.parents else None)
    details = (
        f"문장: {row['text']}\n\n"
        f"backend: {row.get('backend', '')} | model: {row.get('model_id', '')}\n"
        f"요청 악센트: {row['accent']} (실제 발음은 청취 검수 필요)\n"
        f"배경 출처: {row.get('background_source', '')}\n"
        f"요청 SNR: {row.get('target_snr_db', '')} dB | 측정 SNR: {row.get('achieved_snr_db', '')} dB\n\n"
        f"instruct:\n{row['instruct']}\n\n오류: {row.get('error', '')}"
    )
    return paths[0], paths[1], details


def _results(state: dict, selected: str | None = None):
    rows = state["rows"]
    chosen = selected or next(
        (r["artifact_id"] for r in rows if r["status"] == "success"),
        rows[0]["artifact_id"] if rows else None,
    )
    successful = sum(r["status"] == "success" for r in rows)
    status = f"**성공 {successful} / 전체 {len(rows)}** · 저장 위치: `{state['output_dir']}`"
    if state["backend"] == "demo":
        status += "\n\n**Demo는 말소리가 아닌 테스트 신호입니다. 문장·톤·악센트 품질을 평가할 수 없습니다.**"
    clean, mixed, details = _sample(state, chosen)
    metadata_path = Path(state["output_dir"]) / "metadata.csv"
    return (
        state,
        _result_rows(rows),
        gr.Dropdown(choices=_sample_choices(rows), value=chosen),
        status,
        str(metadata_path) if metadata_path.exists() else None,
        clean,
        mixed,
        details,
    )


def _generate(*args, progress=gr.Progress()):
    plan_args, backend_name, device, background_dir = args[:-3], *args[-3:]
    jobs = _plan(*plan_args)
    bg_dir = Path(background_dir).expanduser().resolve() if background_dir.strip() else None
    if bg_dir is not None and not bg_dir.is_dir():
        raise gr.Error("배경음 폴더가 존재하지 않습니다.")
    output_dir = OUTPUTS / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    state = {
        "output_dir": str(output_dir),
        "backend": backend_name,
        "device": device,
        "background_dir": str(bg_dir) if bg_dir else None,
        "jobs": {j.sample_id: asdict(j) for j in jobs},
        "rows": [],
    }
    try:
        backend = _backend(backend_name, device)
        state["rows"] = generate_batch(
            jobs,
            output_dir,
            backend,
            background_dir=bg_dir,
            progress=lambda done, total, row: progress(
                (done, total), desc=f"{done}/{total} · {row['status']}"
            ),
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise gr.Error(str(exc)) from exc
    return _results(state)


def _regenerate(state: dict, artifact_id: str | None, progress=gr.Progress()):
    if not state or not artifact_id:
        raise gr.Error("먼저 생성한 뒤 재생성할 샘플을 선택하세요.")
    old = next((r for r in state["rows"] if r["artifact_id"] == artifact_id), None)
    if old is None:
        raise gr.Error("선택한 샘플이 없습니다.")
    new_job = regenerate_job(Job(**state["jobs"][old["sample_id"]]))
    while new_job.sample_id in state["jobs"]:
        new_job = regenerate_job(new_job)
    try:
        rows = generate_batch(
            [new_job],
            state["output_dir"],
            _backend(state["backend"], state["device"]),
            background_dir=state["background_dir"],
            progress=lambda done, total, row: progress((done, total), desc="선택한 샘플 재생성 중"),
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise gr.Error(str(exc)) from exc
    state["jobs"][new_job.sample_id] = asdict(new_job)
    state["rows"] = load_manifest(state["output_dir"])
    return _results(state, rows[0]["artifact_id"])


def _load_review(state: dict, artifact_id: str | None):
    empty = ("unchecked", "unchecked", "unchecked", "unchecked", "", "")
    if not state or not artifact_id:
        return empty
    path = Path(state["output_dir"]) / "reviews.csv"
    if not path.exists():
        return empty
    with _REVIEW_LOCK:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            record = next(
                (r for r in csv.DictReader(handle) if r["artifact_id"] == artifact_id), None
            )
    if record is None:
        return empty
    return (
        record["text_correct"],
        record["tone_distinct"],
        record["accent_audible"],
        record["noise_acceptable"],
        record["notes"],
        f"저장된 검수: {record['reviewed_at']}",
    )


def _save_review(
    state, artifact_id, text_correct, tone_distinct, accent_audible, noise_acceptable, notes
):
    if not state or not artifact_id:
        raise gr.Error("검수할 샘플을 먼저 선택하세요.")
    row = next((r for r in state["rows"] if r["artifact_id"] == artifact_id), None)
    if row is None or row["status"] != "success":
        raise gr.Error("성공적으로 생성된 샘플만 검수할 수 있습니다.")
    if row.get("backend") == "demo":
        raise gr.Error(
            "Demo는 비음성 테스트 신호입니다. 음성 품질 검수는 Qwen 결과에서 진행하세요."
        )
    values = [text_correct, tone_distinct, accent_audible, noise_acceptable]
    if any(v not in {"unchecked", "yes", "no"} for v in values):
        raise gr.Error("잘못된 검수 값입니다.")
    record = dict(
        zip(
            REVIEW_FIELDS,
            [
                artifact_id,
                row["sample_id"],
                row["backend"],
                datetime.now(timezone.utc).isoformat(),
                *values,
                notes,
            ],
        )
    )
    path = Path(state["output_dir"]) / "reviews.csv"
    with _REVIEW_LOCK:
        existing = []
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                existing = list(csv.DictReader(handle))
        existing = [r for r in existing if r["artifact_id"] != artifact_id]
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows([*existing, record])
        temporary.replace(path)
    return f"검수 저장: {artifact_id}", str(path)


def create_app() -> gr.Blocks:
    sentences = read_sentences(DEFAULT_CSV)
    initial_jobs = create_jobs(sentences, list(PRESETS), list(ACCENTS))
    with gr.Blocks(title="Scenario TTS · 상황별 배치 음성 생성") as app:
        gr.Markdown(
            "# Scenario TTS\n정해진 영어 문장을 상황에 맞춰 연출하고, 깨끗한 음성과 배경음을 합성한 음성을 비교합니다."
        )
        gr.Markdown(
            "Qwen3-TTS 1.7B CustomVoice · 기본 화자 Ryan · 모델 재학습 없음. **American/British는 요청 조건이며, 실제 악센트 차이는 청취 확인이 필요합니다.**"
        )
        sentence_state = gr.State([asdict(s) for s in sentences])
        run_state = gr.State({})
        with gr.Row():
            with gr.Column(scale=2):
                csv_file = gr.File(
                    label="문장 CSV (sentence_id,text) · 비우면 예제 5개",
                    file_types=[".csv"],
                    type="filepath",
                )
                csv_status = gr.Markdown("예제 5개 문장을 불러왔습니다.")
                selected_sentences = gr.CheckboxGroup(
                    choices=_sentence_choices(sentences),
                    value=[s.sentence_id for s in sentences],
                    label="생성할 문장",
                )
            with gr.Column():
                presets = gr.CheckboxGroup(
                    choices=[(p.label, key) for key, p in PRESETS.items()],
                    value=list(PRESETS),
                    label="상황 preset",
                )
                accents = gr.CheckboxGroup(
                    choices=[("American", "american"), ("British", "british")],
                    value=list(ACCENTS),
                    label="요청 악센트",
                )
        with gr.Accordion("세부 설정 · 톤 / 배경 / 소음 / 실행 환경", open=False):
            with gr.Row():
                speaker = gr.Dropdown(choices=["Ryan", "Aiden"], value="Ryan", label="화자")
                background = gr.Dropdown(
                    choices=[("상황 기본값", "preset"), *[(b, b) for b in BACKGROUNDS]],
                    value="preset",
                    label="배경음 변경",
                )
                noise_level = gr.Dropdown(
                    choices=[
                        ("상황 기본값", "preset"),
                        ("약함 · 20 dB", "weak"),
                        ("보통 · 10 dB", "medium"),
                        ("강함 · 5 dB", "strong"),
                        ("직접 입력", "custom"),
                    ],
                    value="preset",
                    label="소음 크기 (SNR)",
                )
                custom_snr = gr.Number(value=20, label="직접 입력 SNR (dB)")
            tone = gr.Textbox(
                label="톤 지시 덮어쓰기 (영어, 비우면 상황 기본값)",
                placeholder="Speak warmly and slowly to a worried friend.",
                lines=2,
            )
            with gr.Row():
                seed = gr.Number(value=42, precision=0, label="기준 seed")
                backend = gr.Dropdown(
                    choices=[("Qwen · 실제 TTS", "qwen"), ("Demo · 비음성 테스트 신호", "demo")],
                    value="qwen",
                    label="생성 엔진",
                )
                device = gr.Dropdown(
                    choices=["auto", "cpu", "cuda:0"], value="auto", label="Qwen 장치"
                )
            background_dir = gr.Textbox(
                label="배경 WAV 폴더 (선택)",
                placeholder=str(PROJECT_ROOT / "assets" / "backgrounds"),
            )
            gr.Markdown(
                "기본 배경은 **절차적으로 합성한 장소별 근사음**입니다. 실제 카페·역 녹음이 아닙니다. 폴더를 지정하면 `cafe.wav`, `station.wav`, `street.wav`, `room.wav`, `crowd.wav`를 사용할 수 있습니다. SNR은 발화 구간 RMS 기준이며 실제 장소의 소음 수준을 나타내지 않습니다."
            )
        plan_inputs = [
            sentence_state,
            selected_sentences,
            presets,
            accents,
            speaker,
            tone,
            background,
            noise_level,
            custom_snr,
            seed,
        ]
        with gr.Row():
            preview_button = gr.Button("조합과 프롬프트 미리보기")
            generate_button = gr.Button("선택한 조합 생성", variant="primary")
        plan_status = gr.Markdown(f"**{len(initial_jobs)}개 조합** · 문장 5 × 상황 6 × 악센트 2")
        with gr.Accordion("생성 계획 · text와 instruct를 분리해 확인", open=False):
            plan_table = gr.Dataframe(
                headers=PLAN_HEADERS,
                value=[
                    [
                        j.sentence_id,
                        j.text,
                        j.preset,
                        j.accent,
                        j.speaker,
                        j.seed,
                        j.background,
                        j.snr_db,
                        j.instruct,
                    ]
                    for j in initial_jobs
                ],
                interactive=False,
                wrap=True,
            )
        gr.Markdown(
            "## 결과 듣기와 재생성\n처음 실행하면 모델을 내려받습니다. CPU 생성은 오래 걸릴 수 있습니다. 생성 결과와 실패 이유는 각 실행 폴더의 CSV에 기록합니다."
        )
        run_status = gr.Markdown("아직 생성된 결과가 없습니다.")
        results_table = gr.Dataframe(headers=RESULT_HEADERS, interactive=False, wrap=True)
        metadata_file = gr.File(label="생성 정보 CSV 다운로드", interactive=False)
        sample = gr.Dropdown(choices=[], label="비교 / 재생성할 샘플", interactive=True)
        with gr.Row():
            clean_audio = gr.Audio(label="깨끗한 음성 WAV", type="filepath", interactive=False)
            mixed_audio = gr.Audio(label="배경음 합성 WAV", type="filepath", interactive=False)
        details = gr.Textbox(label="선택한 샘플의 문장·지시·음향 설정", lines=9, interactive=False)
        regenerate_button = gr.Button("선택한 샘플만 새 seed로 재생성")
        gr.Markdown(
            "기존 샘플은 유지하고 새 샘플을 추가합니다. 현재 화면에서 생성한 실행 결과를 검수할 수 있습니다."
        )
        with gr.Accordion("청취 검수 · 4개 항목", open=False):
            choices = [("미검수", "unchecked"), ("예", "yes"), ("아니오", "no")]
            with gr.Row():
                text_correct = gr.Radio(
                    choices=choices, value="unchecked", label="문장이 정확한가?"
                )
                tone_distinct = gr.Radio(
                    choices=choices, value="unchecked", label="상황의 톤이 구별되는가?"
                )
                accent_audible = gr.Radio(
                    choices=choices, value="unchecked", label="요청한 악센트가 들리는가?"
                )
                noise_acceptable = gr.Radio(
                    choices=choices, value="unchecked", label="소음이 과하지 않은가?"
                )
            notes = gr.Textbox(label="검수 메모")
            save_review = gr.Button("선택한 샘플 검수 저장")
            review_status = gr.Markdown()
            reviews_file = gr.File(label="검수 CSV 다운로드", interactive=False)
        result_outputs = [
            run_state,
            results_table,
            sample,
            run_status,
            metadata_file,
            clean_audio,
            mixed_audio,
            details,
        ]
        csv_file.change(
            _load_csv, inputs=[csv_file], outputs=[sentence_state, selected_sentences, csv_status]
        )
        preview_button.click(_preview, inputs=plan_inputs, outputs=[plan_status, plan_table])
        generate_button.click(
            _generate,
            inputs=[*plan_inputs, backend, device, background_dir],
            outputs=result_outputs,
            concurrency_limit=1,
            concurrency_id="generation",
        )
        regenerate_button.click(
            _regenerate,
            inputs=[run_state, sample],
            outputs=result_outputs,
            concurrency_limit=1,
            concurrency_id="generation",
        )
        sample.change(
            _sample, inputs=[run_state, sample], outputs=[clean_audio, mixed_audio, details]
        )
        sample.change(
            _load_review,
            inputs=[run_state, sample],
            outputs=[
                text_correct,
                tone_distinct,
                accent_audible,
                noise_acceptable,
                notes,
                review_status,
            ],
        )
        save_review.click(
            _save_review,
            inputs=[
                run_state,
                sample,
                text_correct,
                tone_distinct,
                accent_audible,
                noise_acceptable,
                notes,
            ],
            outputs=[review_status, reviews_file],
            concurrency_limit=1,
        )
    return app


def main() -> None:
    create_app().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
