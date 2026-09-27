"""Turns a natural-language task description into a validated TaskSpec.

Supports two interchangeable providers, picked from whichever API key is
present in the environment (Anthropic takes priority if both are set):

- Anthropic (Claude): native tool-use, forced to call `define_task`.
- Groq: OpenAI-compatible chat-completions API with forced function calling.

Either way the model can only return structured data matching
TASK_SPEC_TOOL's schema - never free-form code.
"""

from __future__ import annotations

import json
import os

import requests

from .schema import SYSTEM_PROMPT, TASK_SPEC_TOOL

ANTHROPIC_MODEL = "claude-sonnet-5"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class TaskGenerationError(RuntimeError):
    pass


def generate_task_spec(description: str) -> dict:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    if anthropic_key:
        spec = _generate_with_anthropic(description, anthropic_key)
    elif groq_key:
        spec = _generate_with_groq(description, groq_key)
    else:
        raise TaskGenerationError(
            "Falta una clave de API. Copia .env.example a .env y pon "
            "ANTHROPIC_API_KEY o GROQ_API_KEY antes de generar una tarea."
        )

    _validate_spec(spec)
    return spec


def _generate_with_anthropic(description: str, api_key: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
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
            return block.input

    raise TaskGenerationError("El modelo no devolvio una definicion de tarea valida.")


def _generate_with_groq(description: str, api_key: str) -> dict:
    tool = {
        "type": "function",
        "function": {
            "name": TASK_SPEC_TOOL["name"],
            "description": TASK_SPEC_TOOL["description"],
            "parameters": TASK_SPEC_TOOL["input_schema"],
        },
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": "define_task"}},
        "max_tokens": 2048,
    }

    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise TaskGenerationError(f"Error de red llamando a Groq: {exc}") from exc

    if resp.status_code != 200:
        raise TaskGenerationError(f"Error de la API de Groq ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    try:
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        arguments = tool_calls[0]["function"]["arguments"]
    except (KeyError, IndexError) as exc:
        raise TaskGenerationError(f"Respuesta inesperada de Groq: {data}") from exc

    try:
        return json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise TaskGenerationError(f"Groq devolvio JSON invalido: {arguments}") from exc


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
        # "weight" and "penalty" must always be non-negative by contract (see schema.py);
        # models occasionally flip the sign, which would invert the intended incentive.
        for key in ("weight", "penalty"):
            if key in term and term[key] is not None:
                term[key] = abs(term[key])
