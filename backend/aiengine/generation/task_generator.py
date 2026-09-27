"""Turns a natural-language task description into a validated TaskSpec.

Calls the Claude API with a forced tool call so the model can only return
structured data matching TASK_SPEC_TOOL's schema - never free-form code.
"""

from __future__ import annotations

import os

import anthropic

from .schema import SYSTEM_PROMPT, TASK_SPEC_TOOL

MODEL = "claude-sonnet-5"


class TaskGenerationError(RuntimeError):
    pass


def generate_task_spec(description: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise TaskGenerationError(
            "Falta ANTHROPIC_API_KEY. Copia .env.example a .env y pon tu clave de "
            "https://console.anthropic.com/ antes de generar una tarea."
        )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[TASK_SPEC_TOOL],
            tool_choice={"type": "tool", "name": "define_task"},
            messages=[{"role": "user", "content": description}],
        )
    except anthropic.APIError as exc:
        raise TaskGenerationError(f"Error llamando a la API de Claude: {exc}") from exc

    for block in response.content:
        if block.type == "tool_use" and block.name == "define_task":
            spec = block.input
            _validate_spec(spec)
            return spec

    raise TaskGenerationError("El modelo no devolvio una definicion de tarea valida.")


def _validate_spec(spec: dict) -> None:
    entity_ids = {e["id"] for e in spec.get("entities", [])}
    if not entity_ids:
        raise TaskGenerationError("La tarea generada no tiene entidades.")

    controlled = spec.get("action", {}).get("controlled_entity")
    if controlled not in entity_ids:
        raise TaskGenerationError("La entidad controlada no existe entre las entidades definidas.")

    for term in spec.get("reward_terms", []):
        for key in ("from", "to", "entity"):
            if key in term and term[key] not in entity_ids:
                raise TaskGenerationError(f"El termino de recompensa referencia una entidad desconocida: {term[key]}")
