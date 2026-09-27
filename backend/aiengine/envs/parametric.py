"""Generic physics environment driven entirely by a declarative TaskSpec.

This is the core of "write your task in plain language": instead of one
hand-coded Environment per task, a single ParametricEnv interprets a JSON
spec (entities + a controlled entity + composable reward terms) built by
the LLM-backed generator in `aiengine.generation`. No task-specific code
is ever executed - only this fixed, audited interpreter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pymunk

from .base import Environment, Space

DT = 1.0 / 30.0


class ParametricEnv(Environment):
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.name = spec.get("name", "generated_task")
        self.description = spec.get("description", "")

        self._entity_ids = [e["id"] for e in spec["entities"]]
        self._controlled_id = spec["action"]["controlled_entity"]
        self._max_force = float(spec["action"].get("max_force", 8.0))
        self._max_steps = int(spec.get("max_steps", 300))
        self._field_w = float(spec["field"]["width"])
        self._field_h = float(spec["field"]["height"])
        self._field_theme = spec["field"].get("theme", "plain")

        obs_dim = len(self._entity_ids) * 4
        self.observation_space = Space(
            low=np.full(obs_dim, -1.0, dtype=np.float32),
            high=np.full(obs_dim, 1.0, dtype=np.float32),
        )
        self.action_space = Space(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self._space: pymunk.Space | None = None
        self._bodies: dict[str, pymunk.Body] = {}
        self._rng = np.random.default_rng()
        self._steps = 0
        self._prev_distances: dict[int, float] = {}

    def _build_world(self) -> None:
        space = pymunk.Space()
        space.damping = 0.85
        w, h = self._field_w, self._field_h
        for a, b in [((0, 0), (w, 0)), ((0, h), (w, h)), ((0, 0), (0, h)), ((w, 0), (w, h))]:
            seg = pymunk.Segment(space.static_body, a, b, 0.05)
            seg.elasticity = 0.6
            space.add(seg)

        bodies: dict[str, pymunk.Body] = {}
        for ent in self.spec["entities"]:
            is_static = bool(ent.get("static", False))
            body = pymunk.Body(body_type=pymunk.Body.STATIC) if is_static else pymunk.Body(
                mass=float(ent.get("mass", 1.0)), moment=10
            )

            pos = ent.get("initial_position", "random")
            if pos == "random" or pos is None:
                x = self._rng.uniform(w * 0.1, w * 0.9)
                y = self._rng.uniform(h * 0.1, h * 0.9)
            else:
                x, y = float(pos[0]), float(pos[1])
            body.position = (x, y)

            if ent.get("kind") == "box":
                sw, sh = ent.get("size", [1.0, 1.0])
                shape = pymunk.Poly.create_box(body, (sw, sh))
            else:
                shape = pymunk.Circle(body, float(ent.get("radius", 0.5)))
            shape.elasticity = 0.5
            shape.friction = 0.4
            space.add(body, shape)
            bodies[ent["id"]] = body

        self._space = space
        self._bodies = bodies

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._build_world()
        self._steps = 0
        self._prev_distances = {
            i: self._pair_distance(t["from"], t["to"])
            for i, t in enumerate(self.spec.get("reward_terms", []))
            if t["type"] == "distance_delta"
        }
        return self._obs()

    def _pair_distance(self, a_id: str, b_id: str) -> float:
        a, b = self._bodies[a_id].position, self._bodies[b_id].position
        return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)

    def step(self, action: np.ndarray):
        assert self._space is not None
        action = np.clip(action, -1.0, 1.0)
        controlled = self._bodies[self._controlled_id]
        fx, fy = float(action[0]) * self._max_force, float(action[1]) * self._max_force
        controlled.apply_force_at_world_point((fx, fy), controlled.position)

        self._space.step(DT)
        self._steps += 1

        reward, done, info = self._evaluate_reward_terms()
        if self._steps >= self._max_steps:
            done = True

        return self._obs(), reward, done, info

    def _evaluate_reward_terms(self) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        done = False
        info: dict[str, Any] = {}
        w, h = self._field_w, self._field_h

        for i, term in enumerate(self.spec.get("reward_terms", [])):
            kind = term["type"]
            weight = float(term.get("weight", 1.0))

            if kind == "distance_delta":
                dist = self._pair_distance(term["from"], term["to"])
                reward += (self._prev_distances[i] - dist) * weight
                self._prev_distances[i] = dist

            elif kind == "reach_bonus":
                dist = self._pair_distance(term["from"], term["to"])
                if dist <= float(term.get("threshold", 1.0)):
                    reward += float(term.get("bonus", 10.0))
                    info[f"reached_{term['from']}_{term['to']}"] = True
                    if term.get("ends_episode", False):
                        done = True

            elif kind == "time_penalty":
                reward -= weight

            elif kind == "velocity_penalty":
                v = self._bodies[term["entity"]].velocity
                reward -= (v.x ** 2 + v.y ** 2) ** 0.5 * weight

            elif kind == "out_of_bounds_penalty":
                p = self._bodies[term["entity"]].position
                if not (0 <= p.x <= w and 0 <= p.y <= h):
                    reward -= float(term.get("penalty", 5.0))
                    if term.get("ends_episode", False):
                        done = True

        return reward, done, info

    def _obs(self) -> np.ndarray:
        w, h = self._field_w, self._field_h
        values = []
        for eid in self._entity_ids:
            body = self._bodies[eid]
            p, v = body.position, body.velocity
            values.extend([p.x / w, p.y / h, v.x / 10.0, v.y / 10.0])
        return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)

    def render_state(self) -> dict[str, Any]:
        entities = []
        for ent in self.spec["entities"]:
            body = self._bodies[ent["id"]]
            entities.append(
                {
                    "id": ent["id"],
                    "kind": ent.get("kind", "circle"),
                    "visual": ent.get("visual", "plain"),
                    "radius": ent.get("radius", 0.5),
                    "size": ent.get("size", [1.0, 1.0]),
                    "position": [body.position.x, 0, body.position.y],
                    "velocity": [body.velocity.x, 0, body.velocity.y],
                    "color": ent.get("color", "#4f8cff"),
                }
            )
        return {
            "field": {"w": self._field_w, "h": self._field_h, "theme": self._field_theme},
            "entities": entities,
        }
