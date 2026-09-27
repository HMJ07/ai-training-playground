"""Local web server: serves the frontend and streams training in real time.

Run with `python run.py` from the repo root. Everything happens on
localhost, nothing is sent anywhere else.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from aiengine.envs.registry import create_task, list_tasks
from aiengine.training.ppo import PPOConfig, PPOTrainer, TrainingMetrics

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

app = FastAPI(title="AI Training Playground")


class TrainingSession:
    """Owns at most one running trainer and broadcasts its ticks."""

    def __init__(self) -> None:
        self.trainer: PPOTrainer | None = None
        self.thread: threading.Thread | None = None
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def start(self, task_id: str) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("A training session is already running")

        env = create_task(task_id)
        self.trainer = PPOTrainer(env, PPOConfig(), on_tick=self._on_tick)
        self.thread = threading.Thread(target=self.trainer.train, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.trainer is not None:
            self.trainer.request_stop()

    def _on_tick(self, render_state: dict, metrics: TrainingMetrics) -> None:
        if not self.clients or self.loop is None:
            return
        payload = json.dumps(
            {
                "type": "tick",
                "render": render_state,
                "metrics": {
                    "step": metrics.step,
                    "episode": metrics.episode,
                    "lastReturn": metrics.last_episode_return,
                    "meanReturn100": metrics.mean_return_100,
                },
            }
        )
        # This runs on the trainer thread; hand off to the event loop.
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)

    async def _broadcast(self, payload: str) -> None:
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


session = TrainingSession()


@app.get("/api/tasks")
def get_tasks():
    return list_tasks()


@app.post("/api/train/start/{task_id}")
def start_training(task_id: str):
    session.start(task_id)
    return {"status": "started", "task": task_id}


@app.post("/api/train/stop")
def stop_training():
    session.stop()
    return {"status": "stopping"}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session.loop = asyncio.get_event_loop()
    session.clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        session.clients.discard(websocket)


# Import example tasks so their @register_task decorators run.
def _load_builtin_tasks() -> None:
    from examples import score_goal  # noqa: F401


_load_builtin_tasks()

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
