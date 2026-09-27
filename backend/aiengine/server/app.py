"""Local web server: serves the frontend and streams training in real time.

Run with `python run.py` from the repo root. Everything happens on
localhost except the one call to Groq or Claude (both free-tier friendly)
that turns your task description into a TaskSpec.
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

from aiengine.envs.registry import create_env, get_classifier, get_spec, list_tasks, register_task
from aiengine.generation.task_generator import TaskGenerationError, generate_task_spec
from aiengine.training.ppo import PPOConfig, PPOTrainer, TrainingMetrics

load_dotenv()

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

app = FastAPI(title="AI Training Playground")


class TrainingSession:
    """Owns at most one running trainer (of either kind) and broadcasts its ticks."""

    def __init__(self) -> None:
        self.runner = None  # PPOTrainer or TextClassifierTask, whichever is active
        self.thread: threading.Thread | None = None
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def start(self, task_id: str) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("A training session is already running")

        spec = get_spec(task_id)
        if spec.get("kind") == "classification":
            classifier = get_classifier(task_id)
            self.runner = classifier
            self.thread = threading.Thread(
                target=classifier.train, kwargs={"on_tick": self._on_classification_tick}, daemon=True
            )
        else:
            env = create_env(task_id)
            trainer = PPOTrainer(env, PPOConfig(), on_tick=self._on_control_tick)
            self.runner = trainer
            self.thread = threading.Thread(target=trainer.train, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.runner is not None:
            self.runner.request_stop()

    def _on_control_tick(self, render_state: dict, metrics: TrainingMetrics) -> None:
        self._broadcast_sync(
            {
                "type": "tick",
                "render": render_state,
                "metrics": {
                    "kind": "control",
                    "step": metrics.step,
                    "episode": metrics.episode,
                    "lastReturn": metrics.last_episode_return,
                    "meanReturn100": metrics.mean_return_100,
                },
            }
        )

    def _on_classification_tick(self, render_state: dict, metrics: dict) -> None:
        self._broadcast_sync(
            {
                "type": "tick",
                "render": render_state,
                "metrics": {
                    "kind": "classification",
                    "epoch": metrics["epoch"],
                    "totalEpochs": metrics["totalEpochs"],
                    "accuracy": metrics["accuracy"],
                    "numExamples": metrics["numExamples"],
                },
            }
        )

    def _broadcast_sync(self, payload: dict) -> None:
        if not self.clients or self.loop is None:
            return
        text = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._broadcast(text), self.loop)

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


class AddExampleRequest(BaseModel):
    text: str
    label: str


class PredictRequest(BaseModel):
    text: str


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

    task_info = {"id": task_id, "name": spec["name"], "description": spec["description"], "kind": spec["kind"]}
    if spec["kind"] == "classification":
        task_info["labels"] = spec["labels"]
        task_info["numSeedExamples"] = len(spec["seed_examples"])
    return {"task": task_info, "spec": spec}


@app.get("/api/tasks")
def get_tasks():
    return list_tasks()


@app.get("/api/tasks/{task_id}")
def get_task_spec(task_id: str):
    return get_spec(task_id)


@app.post("/api/tasks/{task_id}/examples")
def add_example(task_id: str, req: AddExampleRequest):
    try:
        classifier = get_classifier(task_id)
        classifier.add_example(req.text.strip(), req.label)
    except (KeyError, ValueError) as exc:
        return {"error": str(exc)}
    return {"status": "ok", "numExamples": len(classifier.examples)}


@app.get("/api/tasks/{task_id}/examples")
def list_examples(task_id: str):
    classifier = get_classifier(task_id)
    return {"labels": classifier.labels, "examples": classifier.examples}


@app.post("/api/tasks/{task_id}/predict")
def predict(task_id: str, req: PredictRequest):
    try:
        classifier = get_classifier(task_id)
        return classifier.predict(req.text.strip())
    except (KeyError, RuntimeError) as exc:
        return {"error": str(exc)}


@app.post("/api/train/start/{task_id}")
def start_training(task_id: str):
    try:
        session.start(task_id)
    except RuntimeError as exc:
        return {"error": str(exc)}
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
