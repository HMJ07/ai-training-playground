"""Local web server: serves the frontend and streams training in real time.

Run with `python run.py` from the repo root. Everything happens on
localhost except the one call to the Claude API that turns your task
description into a TaskSpec.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aiengine.envs.registry import create_task, get_spec, list_tasks, register_task
from aiengine.generation.task_generator import TaskGenerationError, generate_task_spec
from aiengine.training.ppo import PPOConfig, PPOTrainer, TrainingMetrics

load_dotenv()

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


class GenerateTaskRequest(BaseModel):
    description: str


@app.post("/api/tasks/generate")
def generate_task(req: GenerateTaskRequest):
    description = req.description.strip()
    if not description:
        return {"error": "Describe la tarea que quieres entrenar."}
    try:
        spec = generate_task_spec(description)
    except TaskGenerationError as exc:
        return {"error": str(exc)}

    task_id = f"{spec.get('name', 'task').lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
    register_task(task_id, spec)
    return {"task": {"id": task_id, "name": spec["name"], "description": spec["description"]}, "spec": spec}


@app.get("/api/tasks")
def get_tasks():
    return list_tasks()


@app.get("/api/tasks/{task_id}")
def get_task_spec(task_id: str):
    return get_spec(task_id)


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


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
