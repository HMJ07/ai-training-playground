"""Example task: kick a ball into a goal.

Demonstrates the full contract a task must satisfy: physics via pymunk,
a shaped reward, and a render_state() the three.js frontend can draw.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pymunk

from aiengine.envs.base import Environment, Space
from aiengine.envs.registry import register_task

FIELD_W, FIELD_H = 20.0, 12.0
GOAL_Y_RANGE = (FIELD_H / 2 - 2.0, FIELD_H / 2 + 2.0)
PLAYER_RADIUS = 0.5
BALL_RADIUS = 0.3
MAX_FORCE = 8.0
DT = 1.0 / 30.0
MAX_STEPS = 300


@register_task("score_goal")
class ScoreGoalEnv(Environment):
    name = "Marcar un gol"
    description = (
        "Un jugador debe empujar el balon dentro de la porteria. "
        "Accion continua: fuerza (fx, fy) sobre el jugador."
    )

    # obs = [player_x, player_y, player_vx, player_vy,
    #        ball_x, ball_y, ball_vx, ball_vy, goal_center_y]
    observation_space = Space(
        low=np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
        high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
    )
    action_space = Space(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
    )

    def __init__(self) -> None:
        self._space: pymunk.Space | None = None
        self._player: pymunk.Body | None = None
        self._ball: pymunk.Body | None = None
        self._steps = 0
        self._prev_ball_dist = 0.0
        self._rng = np.random.default_rng()

    def _build_world(self) -> None:
        space = pymunk.Space()
        space.damping = 0.85

        for a, b in [
            ((0, 0), (FIELD_W, 0)),
            ((0, FIELD_H), (FIELD_W, FIELD_H)),
            ((0, 0), (0, FIELD_H)),
        ]:
            seg = pymunk.Segment(space.static_body, a, b, 0.05)
            seg.elasticity = 0.6
            space.add(seg)

        player_body = pymunk.Body(mass=5, moment=10)
        player_body.position = (3.0, FIELD_H / 2)
        player_shape = pymunk.Circle(player_body, PLAYER_RADIUS)
        player_shape.elasticity = 0.4
        player_shape.friction = 0.5
        space.add(player_body, player_shape)

        ball_body = pymunk.Body(mass=1, moment=1)
        ball_body.position = (8.0, FIELD_H / 2)
        ball_shape = pymunk.Circle(ball_body, BALL_RADIUS)
        ball_shape.elasticity = 0.7
        ball_shape.friction = 0.3
        space.add(ball_body, ball_shape)

        self._space = space
        self._player = player_body
        self._ball = ball_body

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._build_world()
        self._steps = 0
        assert self._player and self._ball
        self._prev_ball_dist = self._goal_dist(self._ball.position)
        return self._obs()

    def step(self, action: np.ndarray):
        assert self._space and self._player and self._ball
        action = np.clip(action, -1.0, 1.0)
        fx, fy = float(action[0]) * MAX_FORCE, float(action[1]) * MAX_FORCE
        self._player.apply_force_at_world_point((fx, fy), self._player.position)

        self._space.step(DT)
        self._steps += 1

        ball_dist = self._goal_dist(self._ball.position)
        reward = (self._prev_ball_dist - ball_dist) * 1.0
        reward -= 0.01  # time penalty encourages speed
        self._prev_ball_dist = ball_dist

        done = False
        scored = self._ball.position.x >= FIELD_W and GOAL_Y_RANGE[0] <= self._ball.position.y <= GOAL_Y_RANGE[1]
        if scored:
            reward += 50.0
            done = True
        if self._steps >= MAX_STEPS:
            done = True

        return self._obs(), reward, done, {"scored": scored, "steps": self._steps}

    def _goal_dist(self, pos: pymunk.Vec2d) -> float:
        goal_center = (FIELD_W, (GOAL_Y_RANGE[0] + GOAL_Y_RANGE[1]) / 2)
        return float(((pos.x - goal_center[0]) ** 2 + (pos.y - goal_center[1]) ** 2) ** 0.5)

    def _obs(self) -> np.ndarray:
        assert self._player and self._ball
        p, pv = self._player.position, self._player.velocity
        b, bv = self._ball.position, self._ball.velocity
        goal_center_y = (GOAL_Y_RANGE[0] + GOAL_Y_RANGE[1]) / 2
        raw = np.array(
            [p.x, p.y, pv.x, pv.y, b.x, b.y, bv.x, bv.y, goal_center_y],
            dtype=np.float32,
        )
        scale = np.array([FIELD_W, FIELD_H, 10, 10, FIELD_W, FIELD_H, 10, 10, FIELD_H], dtype=np.float32)
        return np.clip(raw / scale, -1.0, 1.0)

    def render_state(self) -> dict[str, Any]:
        assert self._player and self._ball
        return {
            "field": {"w": FIELD_W, "h": FIELD_H, "goalYRange": list(GOAL_Y_RANGE)},
            "entities": [
                {
                    "id": "player",
                    "kind": "sphere",
                    "radius": PLAYER_RADIUS,
                    "position": [self._player.position.x, 0, self._player.position.y],
                    "color": "#4f8cff",
                },
                {
                    "id": "ball",
                    "kind": "sphere",
                    "radius": BALL_RADIUS,
                    "position": [self._ball.position.x, 0, self._ball.position.y],
                    "color": "#ffffff",
                },
            ],
        }
