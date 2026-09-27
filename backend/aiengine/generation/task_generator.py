"""Turns a natural-language task description into a validated TaskSpec.

Supports two interchangeable providers, picked from whichever API key is
present in the environment (Anthropic takes priority if both are set):

- Anthropic (Claude): native tool-use, model picks one of two tools.
- Groq: OpenAI-compatible chat-completions API, model picks one of two
  functions (both providers have generous free tiers - no paid API required).

The model always returns structured data (a control TaskSpec or a
classification TaskSpec) - never free-form code.
"""

from __future__ import annotations

import json
import os

import requests

from .schema import CONTROL_TASK_TOOL, SYSTEM_PROMPT, TOOLS

ANTHROPIC_MODEL = "claude-sonnet-5"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class TaskGenerationError(RuntimeError):
    pass


def _looks_like_schema_error(message: str) -> bool:
    """Heuristic for 'the model's own output violated the tool schema' errors.

    Worth one automatic retry with a corrective nudge, unlike a rate limit
    or network error which would just fail the same way again immediately.
    """
    m = message.lower()
    return "tool_use_failed" in m or "did not match schema" in m or "invalid_request_error" in m


def generate_task_spec(description: str) -> dict:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    if not anthropic_key and not groq_key:
        raise TaskGenerationError(
            "Falta una clave de API. Copia .env.example a .env y pon "
            "ANTHROPIC_API_KEY o GROQ_API_KEY (ambas tienen capa gratuita) "
            "antes de generar una tarea."
        )

    extra_note = None
    for attempt in range(2):
        try:
            if anthropic_key:
                tool_name, spec = _generate_with_anthropic(description, anthropic_key, extra_note)
            else:
                tool_name, spec = _generate_with_groq(description, groq_key, extra_note)
            break
        except TaskGenerationError as exc:
            if attempt == 0 and _looks_like_schema_error(str(exc)):
                extra_note = (
                    f"Your previous attempt was rejected: {exc}. Re-generate the task, this time "
                    "strictly respecting every minimum/maximum/maxItems limit in the tool's schema."
                )
                continue
            raise

    if tool_name == CONTROL_TASK_TOOL["name"]:
        spec["kind"] = "control"
        _validate_control_spec(spec)
    else:
        spec["kind"] = "classification"
        _validate_classification_spec(spec)

    return spec


def _generate_with_anthropic(description: str, api_key: str, extra_note: str | None = None) -> tuple[str, dict]:
    import anthropic

    user_content = description if not extra_note else f"{description}\n\n{extra_note}"
    messages = [{"role": "user", "content": user_content}]

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "any"},
            messages=messages,
        )
    except anthropic.APIError as exc:
        raise TaskGenerationError(f"Error llamando a la API de Claude: {exc}") from exc

    for block in response.content:
        if block.type == "tool_use":
            return block.name, block.input

    raise TaskGenerationError("El modelo no devolvio una definicion de tarea valida.")


def _generate_with_groq(description: str, api_key: str, extra_note: str | None = None) -> tuple[str, dict]:
    tools = [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in TOOLS
    ]
    user_content = description if not extra_note else f"{description}\n\n{extra_note}"
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "tools": tools,
        "tool_choice": "required",
        "max_tokens": 4096,
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
        tool_call = data["choices"][0]["message"]["tool_calls"][0]
        tool_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
    except (KeyError, IndexError) as exc:
        raise TaskGenerationError(f"Respuesta inesperada de Groq: {data}") from exc

    try:
        return tool_name, json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise TaskGenerationError(f"Groq devolvio JSON invalido: {arguments}") from exc


def _validate_control_spec(spec: dict) -> None:
    field = spec.get("field", {})
    field_w, field_h = float(field.get("width", 20)), float(field.get("height", 20))
    if field.get("theme") not in ("pitch", "room", "plain"):
        field["theme"] = "plain"

    # Defense in depth against the model ignoring the "no boundary entity"
    # rule in the prompt: drop any static box whose footprint covers most of
    # the field - it would otherwise render/physically act as a solid block
    # filling the whole sandbox (the walls are already built in automatically).
    kept_entities = []
    for ent in spec.get("entities", []):
        if ent.get("kind") == "box" and ent.get("static"):
            sw, sh = ent.get("size", [1, 1])
            if sw >= field_w * 0.6 and sh >= field_h * 0.6:
                continue
        kept_entities.append(ent)
    spec["entities"] = kept_entities

    # Defense in depth against out-of-range positions: the model sometimes
    # assumes a center-origin coordinate system and emits negative or
    # out-of-bounds coordinates, which places the entity outside the field
    # and outside the camera's framing - it looks like the entity vanished.
    margin = 0.5
    for ent in spec["entities"]:
        pos = ent.get("initial_position")
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            x, y = float(pos[0]), float(pos[1])
            if not (0 <= x <= field_w and 0 <= y <= field_h):
                ent["initial_position"] = [
                    min(max(x, margin), field_w - margin),
                    min(max(y, margin), field_h - margin),
                ]

    entity_ids = {e["id"] for e in spec["entities"]}
    if not entity_ids:
        raise TaskGenerationError("La tarea generada no tiene entidades utilizables.")

    controlled = spec.get("action", {}).get("controlled_entity")
    if controlled not in entity_ids:
        raise TaskGenerationError("La entidad controlada no existe entre las entidades definidas.")

    # Fields each reward term type actually needs to be evaluated; the JSON
    # schema can't express "required depending on type", so models sometimes
    # omit one (e.g. out_of_bounds_penalty without "entity"). Drop those
    # instead of crashing training on the very first step.
    required_fields = {
        "distance_delta": ("from", "to"),
        "reach_bonus": ("from", "to"),
        "time_penalty": (),
        "velocity_penalty": ("entity",),
        "out_of_bounds_penalty": ("entity",),
    }

    valid_terms = []
    for term in spec.get("reward_terms", []):
        needed = required_fields.get(term.get("type"))
        if needed is None or any(term.get(k) is None for k in needed):
            continue
        if any(term[k] not in entity_ids for k in needed if k in ("from", "to", "entity")):
            continue
        # "weight" and "penalty" must always be non-negative by contract (see schema.py);
        # models occasionally flip the sign, which would invert the intended incentive.
        for key in ("weight", "penalty"):
            if key in term and term[key] is not None:
                term[key] = abs(term[key])
        valid_terms.append(term)

    if not valid_terms:
        raise TaskGenerationError("Ningun termino de recompensa generado era valido.")
    spec["reward_terms"] = valid_terms


def _validate_classification_spec(spec: dict) -> None:
    labels = spec.get("labels", [])
    if len(labels) < 2:
        raise TaskGenerationError("La tarea de clasificacion necesita al menos 2 etiquetas.")

    examples = spec.get("seed_examples", [])
    if len(examples) < 4:
        raise TaskGenerationError("La tarea de clasificacion necesita al menos unos ejemplos semilla.")

    label_set = set(labels)
    bad = [e for e in examples if e.get("label") not in label_set]
    if bad:
        # Drop examples with a label the model invented outside the declared set,
        # rather than failing the whole generation over a minor slip.
        spec["seed_examples"] = [e for e in examples if e.get("label") in label_set]
    if len(spec["seed_examples"]) < 4:
        raise TaskGenerationError("Los ejemplos generados no coinciden con las etiquetas declaradas.")
