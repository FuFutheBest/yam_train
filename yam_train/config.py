"""YAML experiment config.

    cfg = load_config("configs/banana_abs.yaml", overrides=["train.num_epochs=10"])

- `_base_: other.yaml` (path relative to the file) is loaded first and deep-merged.
- overrides are `dotted.key=value`; the value is parsed as YAML (numbers, bools, lists).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"config inheritance cycle: {' -> '.join(map(str, (*seen, path)))}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"config {path} must be a mapping")
    base = data.pop("_base_", None)
    if base is None:
        return data
    return _merge(_load(path.parent / base, (*seen, path)), data)


def apply_override(cfg: dict, item: str) -> None:
    key, sep, raw = item.partition("=")
    if not sep or not key:
        raise ValueError(f"override must look like a.b=value, got {item!r}")
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise TypeError(f"override {item!r}: {p} is not a section")
    node[leaf] = yaml.safe_load(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = _load(Path(path))
    for item in overrides or []:
        apply_override(cfg, item)
    cfg.setdefault("name", Path(path).stem)
    return cfg
