"""Command line entry point: plan a matrix before spending time on inference."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

from .backends import SPEAKERS, DemoBackend, QwenBackend
from .batch import create_jobs, generate_batch, read_sentences
from .presets import ACCENTS, BACKGROUNDS, PRESETS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed English sentences in coordinated scenario presets.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "generate"):
        sub = commands.add_parser(command)
        sub.add_argument("csv", type=Path, help="UTF-8 CSV with sentence_id,text columns")
        sub.add_argument("--output", type=Path, required=True, help="Plan CSV path or generation output directory")
        sub.add_argument("--presets", nargs="+", choices=list(PRESETS), default=list(PRESETS))
        sub.add_argument("--accents", nargs="+", choices=list(ACCENTS), default=list(ACCENTS))
        sub.add_argument("--speaker", choices=SPEAKERS, default="Ryan")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--tone", help="Override the preset delivery instruction")
        sub.add_argument("--background", choices=BACKGROUNDS)
        sub.add_argument("--snr-db", type=float)
        if command == "generate":
            sub.add_argument("--backend", choices=("qwen", "demo"), default="qwen", help="demo emits NON-SPEECH tones only")
            sub.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
            sub.add_argument("--background-dir", type=Path, help="Optional directory containing cafe.wav, station.wav, etc.")
            sub.add_argument("--attention", choices=("auto", "sdpa", "eager", "flash_attention_2"), default="auto")
            sub.add_argument("--max-new-tokens", type=int, default=2048, help="Generation length bound; overly small values may truncate speech")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        jobs = create_jobs(read_sentences(args.csv), args.presets, args.accents, args.speaker, args.seed, args.tone, args.background, args.snr_db)
        if args.command == "plan":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(asdict(jobs[0])))
                writer.writeheader()
                writer.writerows(asdict(job) for job in jobs)
            print(f"Planned {len(jobs)} samples -> {args.output}")
            return 0
        backend = DemoBackend() if args.backend == "demo" else QwenBackend(device=args.device, attn_implementation=args.attention, max_new_tokens=args.max_new_tokens)
        if args.backend == "demo":
            print("DEMO: non-speech test tones; this does not validate TTS, tone, or accents.")
        def progress(done, total, row):
            print(f"[{done}/{total}] {row['sample_id']}: {row['status']}", flush=True)
        rows = generate_batch(jobs, args.output, backend, args.background_dir, progress)
        failures = [row for row in rows if row["status"] != "success"]
        print(f"{len(rows) - len(failures)}/{len(rows)} succeeded. Metadata: {args.output / 'metadata.csv'}")
        if failures:
            print(f"First error: {failures[0]['error']}", file=sys.stderr)
        return 1 if failures else 0
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
