"""Base contract every task/environment in the platform must implement.

Any user who wants to train an agent on a new task subclasses `Environment`
and registers it (see `registry.py`). The RL engine and the renderer only
ever talk to this interface, never to a specific task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Space:
    """Minimal description of an observation or action space (Box only)."""

    low: np.ndarray
    high: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high).astype(np.float32)


class Environment(ABC):
    """A single-agent RL task with a renderable state.

    Subclasses implement the physics/logic of the task; the training loop
    and the frontend renderer are completely generic over this interface.
    """

    observation_space: Space
    action_space: Space

    #: Human-readable metadata shown in the UI's task picker.
    name: str = "unnamed_task"
    description: str = ""

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the episode and return the initial observation."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance one timestep.

        Returns (observation, reward, done, info), mirroring the Gymnasium API
        so existing algorithms/tools can be swapped in later if desired.
        """

    @abstractmethod
    def render_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the world for the 3D viewer.

        Shape is up to the task, but by convention it is a dict with an
        "entities" list of {id, kind, position, rotation, color, ...}.
        """
