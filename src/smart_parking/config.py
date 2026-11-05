"""Configuration loading with dot-path access and project-relative path resolution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Read-only configuration wrapper supporting ``cfg.get('a.b.c', default)``."""

    def __init__(self, data: dict[str, Any], sources: list[str] | None = None):
        self._data = data
        self.sources = sources or []

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required configuration key: {dotted}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def path(self, dotted: str, default: str | None = None) -> Path:
        raw = self.get(dotted, default)
        if raw is None:
            raise KeyError(f"Missing path configuration: {dotted}")
        path = Path(str(raw))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


def load_config(*names: str) -> Config:
    """Merge YAML configs in order. Later files override earlier ones."""
    merged: dict[str, Any] = {}
    sources: list[str] = []
    for name in names:
        file_path = name if str(name).endswith((".yaml", ".yml")) else f"configs/{name}.yaml"
        data = load_yaml(file_path)
        merged = deep_merge(merged, data)
        sources.append(str(file_path))
    return Config(merged, sources)


def load_app_config() -> Config:
    """Base application configuration (paths, server, validation, datasets)."""
    return load_config("app", "datasets")


def load_component_config(component: str, *extra: str) -> Config:
    """Load the app configuration plus one model component without key collisions."""
    names = ["app", *extra, component]
    return load_config(*names)
