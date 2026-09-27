"""Registry so new tasks can be plugged in without touching the server code."""

from __future__ import annotations

from typing import Callable

from .base import Environment

_REGISTRY: dict[str, Callable[[], Environment]] = {}


def register_task(task_id: str):
    """Class decorator: `@register_task("score_goal")`."""

    def _wrap(cls):
        _REGISTRY[task_id] = cls
        return cls

    return _wrap


def create_task(task_id: str) -> Environment:
    if task_id not in _REGISTRY:
        raise KeyError(f"Unknown task '{task_id}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[task_id]()


def list_tasks() -> list[dict[str, str]]:
    out = []
    for task_id, cls in _REGISTRY.items():
        out.append(
            {
                "id": task_id,
                "name": getattr(cls, "name", task_id),
                "description": getattr(cls, "description", ""),
            }
        )
    return out
