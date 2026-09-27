"""Incremental text classifier: the "training engine" for classification tasks.

Mirrors the role PPOTrainer plays for physics tasks, but for supervised
text classification (spam vs not spam, sentiment, topic tagging, ...).
Uses scikit-learn (TF-IDF + linear SGDClassifier) - free, local, and fast
enough to retrain from scratch on every "epoch" tick for a live demo.

To reuse the same 3D viewer the physics tasks use, each training tick also
projects every example's TF-IDF vector down to 2D (via TruncatedSVD) and
reports it as a small colored sphere - so you literally watch the classes
separate in space as the model learns.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

PALETTE = ["#4f8cff", "#ff6b6b", "#ffd166", "#06d6a0", "#c77dff", "#f77f00"]
FIELD_SIZE = 20.0


class TextClassifierTask:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.name = spec.get("name", "classification_task")
        self.description = spec.get("description", "")
        self.labels: list[str] = spec["labels"]
        self.input_description = spec.get("input_description", "texto")
        self.examples: list[dict[str, str]] = list(spec.get("seed_examples", []))

        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
        self.model = SGDClassifier(loss="log_loss", random_state=0)
        self._fitted = False
        self._stop_flag = False
        self._label_color = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(self.labels)}

    def add_example(self, text: str, label: str) -> None:
        if label not in self.labels:
            raise ValueError(f"Etiqueta desconocida: {label}. Debe ser una de {self.labels}")
        self.examples.append({"text": text, "label": label})

    def request_stop(self) -> None:
        self._stop_flag = True

    def train(self, epochs: int = 40, on_tick: Callable[[dict, dict], None] | None = None) -> None:
        self._stop_flag = False
        texts = [e["text"] for e in self.examples]
        labels = np.array([e["label"] for e in self.examples])
        classes = np.array(self.labels)

        X = self.vectorizer.fit_transform(texts)
        n = X.shape[0]

        rng = np.random.default_rng(0)
        idx = np.arange(n)
        rng.shuffle(idx)
        split = max(1, int(n * 0.8))
        train_idx = idx[:split]
        val_idx = idx[split:] if split < n else idx[:split]

        coords_2d = self._project(X)

        for epoch in range(1, epochs + 1):
            if self._stop_flag:
                break
            shuffled = train_idx.copy()
            rng.shuffle(shuffled)
            self.model.partial_fit(X[shuffled], labels[shuffled], classes=classes)
            self._fitted = True

            preds = self.model.predict(X[val_idx])
            accuracy = float((preds == labels[val_idx]).mean())

            if on_tick is not None:
                on_tick(
                    self._render_state(coords_2d, labels),
                    {"epoch": epoch, "totalEpochs": epochs, "accuracy": accuracy, "numExamples": n},
                )

    def _project(self, X) -> np.ndarray:
        n_features = X.shape[1]
        if X.shape[0] < 3 or n_features < 2:
            return np.zeros((X.shape[0], 2))
        svd = TruncatedSVD(n_components=2, random_state=0)
        coords = svd.fit_transform(X)
        mins, maxs = coords.min(axis=0), coords.max(axis=0)
        ranges = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
        return (coords - mins) / ranges

    def _render_state(self, coords_2d: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        entities = []
        for i in range(coords_2d.shape[0]):
            x, z = coords_2d[i, 0] * FIELD_SIZE, coords_2d[i, 1] * FIELD_SIZE
            entities.append(
                {
                    "id": f"ex_{i}",
                    "kind": "circle",
                    "radius": 0.35,
                    "color": self._label_color.get(labels[i], "#888888"),
                    "position": [x, 0, z],
                }
            )
        return {"field": {"w": FIELD_SIZE, "h": FIELD_SIZE}, "entities": entities}

    def predict(self, text: str) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("El modelo aun no se ha entrenado. Pulsa 'Entrenar' primero.")
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]
        result = {"label": pred, "probabilities": {}}
        try:
            proba = self.model.predict_proba(X)[0]
            for cls, p in zip(self.model.classes_, proba):
                result["probabilities"][cls] = float(p)
        except AttributeError:
            pass
        return result
