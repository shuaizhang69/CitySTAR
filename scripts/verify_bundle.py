from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]+")


def main() -> int:
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            hits.append(("cache", path))
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".example"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if SECRET_RE.search(text):
            hits.append(("secret", path))

    if hits:
        for kind, path in hits:
            print(f"{kind}: {path.relative_to(ROOT)}")
        return 1
    print("bundle verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
