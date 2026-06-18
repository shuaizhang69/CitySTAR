from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import available_configs, load_pipeline, stage_names
from .runner import command_plan, doctor, format_command, run_plan


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="", help="Dataset config name, e.g. cityrefer or cityanchor.")
    parser.add_argument("--config", default="", help="Path to a *.pipeline.json config.")
    parser.add_argument("--split", default="ND", help="Dataset split, usually ND or NO.")


def load_from_args(args: argparse.Namespace) -> dict:
    return load_pipeline(dataset=args.dataset or None, config_path=args.config or None)


def cmd_list(args: argparse.Namespace) -> int:
    if not args.dataset and not args.config:
        for path in available_configs():
            print(path.name)
        return 0
    pipeline = load_from_args(args)
    print(f"{pipeline.get('name', pipeline.get('dataset', 'pipeline'))}")
    for name in stage_names(pipeline):
        print(f"  {name}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    pipeline = load_from_args(args)
    plan = command_plan(
        pipeline,
        split=args.split,
        names=args.stage or None,
        from_stage=args.from_stage or None,
        to_stage=args.to_stage or None,
    )
    for item in plan:
        print(f"\n[{item['name']}] {item.get('description', '')}")
        print(f"cwd: {item['cwd']}")
        print(format_command(item["command"]))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    pipeline = load_from_args(args)
    report = doctor(
        pipeline,
        split=args.split,
        names=args.stage or None,
        from_stage=args.from_stage or None,
        to_stage=args.to_stage or None,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1

    print(f"config: {pipeline.get('_config_path')}")
    print(f"status: {'ok' if report['ok'] else 'missing requirements'}")
    for stage in report["stages"]:
        print(f"\n[{stage['name']}]")
        script = stage.get("script")
        if script:
            print(f"script: {'ok' if script['exists'] else 'missing'} {script['path']}")
        for item in stage["inputs"]:
            label = "ok" if item["exists"] else ("missing" if item["required"] else "optional-missing")
            print(f"input: {label} {item['path']}")
        for item in stage["env"]:
            print(f"env: {'ok' if item['present'] else 'missing'} {item['name']}")
    return 0 if report["ok"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    pipeline = load_from_args(args)
    plan = command_plan(
        pipeline,
        split=args.split,
        names=args.stage or None,
        from_stage=args.from_stage or None,
        to_stage=args.to_stage or None,
    )
    run_plan(plan, dry_run=args.dry_run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="city-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List pipeline configs or stages.")
    add_config_args(p_list)
    p_list.set_defaults(func=cmd_list)

    p_plan = sub.add_parser("plan", help="Print commands without running them.")
    add_config_args(p_plan)
    p_plan.add_argument("--stage", action="append", help="Run only this stage. Repeatable.")
    p_plan.add_argument("--from-stage", default="")
    p_plan.add_argument("--to-stage", default="")
    p_plan.set_defaults(func=cmd_plan)

    p_doctor = sub.add_parser("doctor", help="Check scripts, inputs, and required env vars.")
    add_config_args(p_doctor)
    p_doctor.add_argument("--stage", action="append", help="Check only this stage. Repeatable.")
    p_doctor.add_argument("--from-stage", default="")
    p_doctor.add_argument("--to-stage", default="")
    p_doctor.add_argument("--json", action="store_true")
    p_doctor.set_defaults(func=cmd_doctor)

    p_run = sub.add_parser("run", help="Run selected stages.")
    add_config_args(p_run)
    p_run.add_argument("--stage", action="append", help="Run only this stage. Repeatable.")
    p_run.add_argument("--from-stage", default="")
    p_run.add_argument("--to-stage", default="")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
