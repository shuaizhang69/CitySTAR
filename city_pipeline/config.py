from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def repo_root() -> Path:
    override = os.environ.get("CITY_PIPELINE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_REPO_ROOT


def config_dir() -> Path:
    return repo_root() / "configs"


def available_configs() -> List[Path]:
    return sorted(config_dir().glob("*.pipeline.json"))


def default_config_path(dataset: str) -> Path:
    return config_dir() / f"{dataset.lower()}.pipeline.json"


def load_pipeline(dataset: str | None = None, config_path: str | None = None) -> Dict[str, Any]:
    if config_path:
        path = Path(config_path)
    elif dataset:
        path = default_config_path(dataset)
    else:
        raise ValueError("Either dataset or config_path is required.")

    if not path.is_absolute():
        path = repo_root() / path
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_config_path"] = str(path)
    return data


def render_context(split: str | None = None) -> Dict[str, str]:
    split_value = split or ""
    split_upper = split_value.upper()
    cityanchor_split_dir = "CityAnchor" if split_upper == "ND" else "city_Anchor"
    return {
        "repo": str(repo_root()),
        "python": sys.executable,
        "split": split_upper,
        "split_lower": split_value.lower(),
        "cityanchor_split_dir": cityanchor_split_dir,
        "sep": os.sep,
    }


def render_value(value: Any, context: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format_map(SafeFormatDict(context))
    if isinstance(value, list):
        return [render_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, context) for key, item in value.items()}
    return value


def resolve_ref(ref: str, context: Dict[str, str]) -> Path:
    rendered = render_value(ref, context)
    if rendered.startswith("repo://"):
        return repo_root() / rendered[len("repo://") :]
    return Path(rendered)


def iter_stages(pipeline: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    yield from pipeline.get("stages", [])


def stage_names(pipeline: Dict[str, Any]) -> List[str]:
    return [stage["name"] for stage in iter_stages(pipeline)]


def render_command(stage: Dict[str, Any], context: Dict[str, str]) -> List[str]:
    return [str(part) for part in render_value(stage.get("command", []), context)]


def render_cwd(stage: Dict[str, Any], context: Dict[str, str]) -> Path:
    cwd = stage.get("cwd", "repo://")
    return resolve_ref(cwd, context)
