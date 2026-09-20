"""Command-line interface for running and replaying warehouse episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .controllers import HeuristicController, LayaController, RandomController
from .dataset import dataset_summary, generate_cases, save_cases
from .recording import load_record, run_episode, save_record, verify_record
from .render import render_record


def _controller(args: argparse.Namespace):
    if args.controller == "heuristic":
        return HeuristicController()
    if args.controller == "random":
        return RandomController(args.seed)
    return LayaController(model=args.model, device=args.device)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual warehouse safety simulation for Laya")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one deterministic warehouse episode")
    run.add_argument(
        "--controller",
        choices=("heuristic", "random", "laya"),
        default="heuristic",
    )
    run.add_argument("--seed", type=int, default=0, help="seed used by the random controller")
    run.add_argument(
        "--model",
        default="convaiinnovations/laya",
        help="Laya model ID or local path",
    )
    run.add_argument("--device", help="torch device for Laya, such as cpu, cuda, or mps")
    run.add_argument("--max-ticks", type=int, default=100)
    run.add_argument("--no-safety-shield", action="store_true")
    run.add_argument("--output", type=Path, default=Path("results/latest.json"))
    run.add_argument("--headless", action="store_true")
    run.add_argument("--fps", type=int, default=4)

    replay = subparsers.add_parser("replay", help="verify and replay a recorded episode")
    replay.add_argument("record", type=Path)
    replay.add_argument("--headless", action="store_true")
    replay.add_argument("--fps", type=int, default=4)

    dataset = subparsers.add_parser(
        "make-dataset", help="generate labeled states from a frozen scenario split"
    )
    dataset.add_argument(
        "--split",
        choices=("train", "validation", "iid_test", "ood_test"),
        default="train",
    )
    dataset.add_argument("--max-ticks", type=int, default=30)
    dataset.add_argument("--output", type=Path, default=Path("results/train.jsonl"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        record = run_episode(
            _controller(args),
            max_ticks=args.max_ticks,
            safety_shield=not args.no_safety_shield,
        )
        save_record(record, args.output)
        print(
            f"outcome={record['outcome']} ticks={record['ticks']} "
            f"controller={record['controller']} record={args.output}"
        )
        if not args.headless:
            render_record(record, fps=args.fps)
        return 0

    if args.command == "make-dataset":
        cases = generate_cases(split=args.split, max_ticks=args.max_ticks)
        save_cases(cases, args.output)
        summary = dataset_summary(cases)
        print(f"cases={summary['cases']} labels={summary['labels']} dataset={args.output}")
        return 0

    record = load_record(args.record)
    verify_record(record)
    print(f"verified {len(record['frames'])} frames: outcome={record['outcome']}")
    if not args.headless:
        render_record(record, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
