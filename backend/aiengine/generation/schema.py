"""JSON schema for a TaskSpec, and the tool definition sent to the LLM.

A TaskSpec never contains executable code - only data describing entities,
which one the agent controls, and a list of composable reward terms. This
is what makes it safe to build from an LLM's output and interpret with
`aiengine.envs.parametric.ParametricEnv`.
"""

TASK_SPEC_TOOL = {
    "name": "define_task",
    "description": (
        "Define a 2D physics reinforcement-learning task as data: the entities "
        "in the scene, which single entity the agent controls with a 2D force, "
        "and a list of reward terms that combine to score the agent's behaviour."
    ),
    "input_schema": {
        "type": "object",
        "required": ["name", "description", "field", "entities", "action", "reward_terms"],
        "properties": {
            "name": {"type": "string", "description": "Short human-readable task name."},
            "description": {"type": "string", "description": "One sentence describing the task."},
            "field": {
                "type": "object",
                "required": ["width", "height"],
                "properties": {
                    "width": {"type": "number", "minimum": 5, "maximum": 40},
                    "height": {"type": "number", "minimum": 5, "maximum": 40},
                },
            },
            "entities": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "required": ["id", "kind", "color"],
                    "properties": {
                        "id": {"type": "string", "description": "Unique snake_case id, e.g. 'player', 'ball', 'goal'."},
                        "kind": {"type": "string", "enum": ["circle", "box"]},
                        "radius": {"type": "number", "description": "For kind=circle."},
                        "size": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "[width, height] for kind=box.",
                        },
                        "color": {"type": "string", "description": "CSS hex color, e.g. '#4f8cff'."},
                        "mass": {"type": "number", "minimum": 0.1, "maximum": 50},
                        "static": {
                            "type": "boolean",
                            "description": "True for immovable markers/obstacles/goals.",
                        },
                        "initial_position": {
                            "description": "[x, y] within the field, or the string 'random'.",
                        },
                    },
                },
            },
            "action": {
                "type": "object",
                "required": ["controlled_entity"],
                "properties": {
                    "controlled_entity": {"type": "string", "description": "id of the entity the agent pushes around."},
                    "max_force": {"type": "number", "minimum": 1, "maximum": 30},
                },
            },
            "reward_terms": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "required": ["type"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "distance_delta",
                                "reach_bonus",
                                "time_penalty",
                                "velocity_penalty",
                                "out_of_bounds_penalty",
                            ],
                        },
                        "from": {"type": "string", "description": "Entity id (distance_delta, reach_bonus)."},
                        "to": {"type": "string", "description": "Entity id (distance_delta, reach_bonus)."},
                        "entity": {"type": "string", "description": "Entity id (velocity_penalty, out_of_bounds_penalty)."},
                        "weight": {
                            "type": "number",
                            "minimum": 0,
                            "description": (
                                "ALWAYS a positive number, never negative. For distance_delta: reward "
                                "given each step equals (previous_distance - current_distance) * weight, "
                                "so a positive weight rewards getting closer. For time_penalty and "
                                "velocity_penalty this positive amount is SUBTRACTED each step, so a "
                                "positive weight discourages wasting time / moving fast."
                            ),
                        },
                        "threshold": {"type": "number", "description": "Distance to count as 'reached' (reach_bonus)."},
                        "bonus": {
                            "type": "number",
                            "description": (
                                "Added to the reward when reach_bonus triggers. Positive to reward "
                                "reaching a target (e.g. a goal); negative to punish reaching something "
                                "bad (e.g. an obstacle or a wrong-colored zone)."
                            ),
                        },
                        "penalty": {
                            "type": "number",
                            "minimum": 0,
                            "description": (
                                "ALWAYS a positive number. This amount is SUBTRACTED from the reward "
                                "when out_of_bounds_penalty triggers."
                            ),
                        },
                        "ends_episode": {"type": "boolean"},
                    },
                },
            },
            "max_steps": {"type": "integer", "minimum": 50, "maximum": 2000},
        },
    },
}

SYSTEM_PROMPT = """You translate a plain-language task description into a TaskSpec \
for a 2D physics reinforcement-learning sandbox, by calling the `define_task` tool.

Rules:
- The agent always controls exactly ONE entity by pushing it with a 2D force \
  (fx, fy). Everything else in the scene is either physics-driven (a ball that \
  gets bumped) or static (a goal marker, an obstacle).
- Represent goals, targets, and zones as entities too (usually `static: true`), \
  and reward reaching them with `reach_bonus` and/or shape the approach with \
  `distance_delta`.
- Always include a small `time_penalty` so the agent is encouraged to act quickly.
- Keep entities to simple circles/boxes - this is a physics sandbox, not a \
  photorealistic simulator. If the user's request is not really a physical \
  control task (e.g. pure image classification), do your best to turn it into \
  an analogous physical task (e.g. represent categories as colored zones the \
  agent must push a matching-colored object into).
- Pick sensible field size, forces and reward weights so the task is learnable \
  in a few hundred thousand steps of PPO.
"""
