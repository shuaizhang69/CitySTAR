from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import (
    iter_stages,
    render_command,
    render_context,
    render_cwd,
    render_value,
    resolve_ref,
    stage_names,
)


def format_command(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def select_stages(
    pipeline: Dict[str, Any],
    names: List[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> List[Dict[str, Any]]:
    stages = list(iter_stages(pipeline))
    all_names = stage_names(pipeline)

    if names:
        missing = [name for name in names if name not in all_names]
        if missing:
            raise ValueError(f"Unknown stage(s): {', '.join(missing)}")
        return [stage for stage in stages if stage["name"] in names]

    start = all_names.index(from_stage) if from_stage else 0
    end = all_names.index(to_stage) + 1 if to_stage else len(stages)
    return stages[start:end]


def command_plan(
    pipeline: Dict[str, Any],
    split: str,
    names: List[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> List[Dict[str, Any]]:
    context = render_context(split)
    result = []
    for stage in select_stages(pipeline, names, from_stage, to_stage):
        result.append(
            {
                "name": stage["name"],
                "description": stage.get("description", ""),
                "cwd": str(render_cwd(stage, context)),
                "command": render_command(stage, context),
                "optional": bool(stage.get("optional", False)),
            }
        )
    return result


def check_path(item: Any, context: Dict[str, str]) -> Dict[str, Any]:
    if isinstance(item, str):
        item = {"path": item}
    path_ref = item["path"]
    required = bool(item.get("required", True))
    kind = item.get("kind", "any")
    path = resolve_ref(path_ref, context)
    exists = path.exists()
    if kind == "file":
        exists = path.is_file()
    elif kind == "dir":
        exists = path.is_dir()
    return {
        "path": str(path),
        "required": required,
        "kind": kind,
        "exists": exists,
    }


def doctor(
    pipeline: Dict[str, Any],
    split: str,
    names: List[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> Dict[str, Any]:
    context = render_context(split)
    stage_reports = []
    ok = True
    for stage in select_stages(pipeline, names, from_stage, to_stage):
        stage_optional = bool(stage.get("optional", False))
        command = render_command(stage, context)
        script = stage.get("script")
        script_report = None
        if script:
            script_path = resolve_ref(script, context)
            script_report = {"path": str(script_path), "exists": script_path.is_file()}
            if not stage_optional:
                ok = ok and script_path.is_file()

        inputs = [check_path(item, context) for item in stage.get("inputs", [])]
        for item in inputs:
            if not stage_optional and item["required"] and not item["exists"]:
                ok = False

        env = []
        for name in render_value(stage.get("env_required", []), context):
            present = bool(os.environ.get(name))
            env.append({"name": name, "present": present})
            if not stage_optional and not present:
                ok = False

        stage_reports.append(
            {
                "name": stage["name"],
                "optional": stage_optional,
                "script": script_report,
                "inputs": inputs,
                "env": env,
                "command": command,
            }
        )
    return {"ok": ok, "stages": stage_reports}


def run_plan(plan: List[Dict[str, Any]], dry_run: bool = False) -> None:
    for item in plan:
        print(f"\n[{item['name']}]", flush=True)
        print(format_command(item["command"]), flush=True)
        if dry_run:
            continue
        cwd = Path(item["cwd"])
        cwd.mkdir(parents=True, exist_ok=True)
        subprocess.run(item["command"], cwd=str(cwd), check=True)
