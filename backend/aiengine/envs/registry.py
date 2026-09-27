"""In-memory registry of tasks generated from user descriptions this session.

There is no fixed menu of tasks: every entry here was produced by the
generator (`aiengine.generation.task_generator`) from free text typed by
the user. Two kinds exist:

- "control": interpreted generically by `ParametricEnv` + PPO.
- "classification": a persistent `TextClassifierTask` instance (it needs to
  remember examples added by the user across requests, and keep whatever
  model has been trained so far so `/predict` can use it).
"""

from __future__ import annotations

from aiengine.classification.engine import TextClassifierTask

from .base import Environment
from .parametric import ParametricEnv

_TASKS: dict[str, dict] = {}


def register_task(task_id: str, spec: dict) -> None:
    entry: dict = {"spec": spec}
    if spec.get("kind") == "classification":
        entry["instance"] = TextClassifierTask(spec)
    _TASKS[task_id] = entry


def create_env(task_id: str) -> Environment:
    """Only valid for kind='control' tasks."""
    return ParametricEnv(get_spec(task_id))


def get_classifier(task_id: str) -> TextClassifierTask:
    if task_id not in _TASKS or "instance" not in _TASKS[task_id]:
        raise KeyError(f"Unknown classification task '{task_id}'.")
    return _TASKS[task_id]["instance"]


def get_spec(task_id: str) -> dict:
    if task_id not in _TASKS:
        raise KeyError(f"Unknown task '{task_id}'. Available: {list(_TASKS)}")
    return _TASKS[task_id]["spec"]


def list_tasks() -> list[dict[str, str]]:
    return [
        {
            "id": task_id,
            "name": entry["spec"].get("name", task_id),
            "description": entry["spec"].get("description", ""),
            "kind": entry["spec"].get("kind", "control"),
        }
        for task_id, entry in _TASKS.items()
    ]
