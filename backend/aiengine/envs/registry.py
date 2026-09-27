"""In-memory registry of tasks generated from user descriptions this session.

There is no fixed menu of tasks: every entry here was produced by the
generator (`aiengine.generation.task_generator`) from free text typed by
the user, and is interpreted generically by `ParametricEnv`.
"""

from __future__ import annotations

from .base import Environment
from .parametric import ParametricEnv

_TASKS: dict[str, dict] = {}


def register_task(task_id: str, spec: dict) -> None:
    _TASKS[task_id] = spec


def create_task(task_id: str) -> Environment:
    if task_id not in _TASKS:
        raise KeyError(f"Unknown task '{task_id}'. Available: {list(_TASKS)}")
    return ParametricEnv(_TASKS[task_id])


def get_spec(task_id: str) -> dict:
    return _TASKS[task_id]


def list_tasks() -> list[dict[str, str]]:
    return [
        {"id": task_id, "name": spec.get("name", task_id), "description": spec.get("description", "")}
        for task_id, spec in _TASKS.items()
    ]
