"""JSON schemas for TaskSpecs, and the tool definitions sent to the LLM.

A TaskSpec never contains executable code - only data. Two kinds exist,
covering two very different families of "things you might want to train":

- CONTROL_TASK_TOOL: a 2D physics task (push/reach/avoid) - interpreted by
  `aiengine.envs.parametric.ParametricEnv` and trained with PPO.
- CLASSIFICATION_TASK_TOOL: a text classification task (spam vs not spam,
  sentiment, topic, ...) - interpreted by
  `aiengine.classification.engine.TextClassifierTask` and trained with an
  incremental linear classifier over TF-IDF features.

The LLM picks whichever tool matches the user's request; both only ever
produce data, never code, so whichever comes back is safe to run locally.
"""

CONTROL_TASK_TOOL = {
    "name": "define_control_task",
    "description": (
        "Define a 2D physics reinforcement-learning task as data: the entities "
        "in the scene, which single entity the agent controls with a 2D force, "
        "and a list of reward terms that combine to score the agent's behaviour. "
        "Use this for anything about an agent moving, pushing, reaching, chasing, "
        "avoiding or balancing something in a physical space."
    ),
    "input_schema": {
        "type": "object",
        "required": ["name", "description", "field", "entities", "action", "reward_terms"],
        "properties": {
            "name": {"type": "string", "description": "Short human-readable task name."},
            "description": {"type": "string", "description": "One sentence describing the task."},
            "field": {
                "type": "object",
                "required": ["width", "height", "theme"],
                "properties": {
                    "width": {"type": "number", "minimum": 5, "maximum": 40},
                    "height": {"type": "number", "minimum": 5, "maximum": 40},
                    "theme": {
                        "type": "string",
                        "enum": ["pitch", "room", "plain"],
                        "description": (
                            "How to dress the ground/surroundings - purely visual, no effect on "
                            "physics. 'pitch': an outdoor sports field (grass, line markings, a "
                            "stadium backdrop) - only for tasks that are actually about a sports "
                            "pitch/court. 'room': an indoor space (a salon, office, house, warehouse "
                            "- anything happening inside a building) - a plain floor, no stadium. "
                            "'plain': an abstract/neutral surface for anything else (e.g. a generic "
                            "physics puzzle with no real-world setting)."
                        ),
                    },
                },
            },
            "entities": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "required": ["id", "kind", "color"],
                    "properties": {
                        "id": {"type": "string", "description": "Unique snake_case id, e.g. 'player', 'ball', 'goal'."},
                        "kind": {
                            "type": "string",
                            "enum": ["circle", "box"],
                            "description": (
                                "PHYSICS shape only (a circle or a box collider) - does not affect how "
                                "it looks. Use `visual` for appearance."
                            ),
                        },
                        "visual": {
                            "type": "string",
                            "enum": ["humanoid", "ball", "goal", "marker", "plain"],
                            "description": (
                                "How this entity is DRAWN, independent of its physics `kind`. "
                                "'humanoid': a person/character/animal-like actor (a player, a robot with "
                                "a body, anything that acts) - rendered as a simple articulated figure "
                                "with a walk/kick animation; always use `kind: circle` with it. "
                                "'ball': a ball or projectile - rendered as a textured sphere; use with "
                                "`kind: circle`. 'goal': ONLY for a literal sports goal/net/hoop (soccer, "
                                "hockey, basketball) - rendered as a real goal frame with posts and a "
                                "net; use with `kind: box`. 'marker': any other abstract target/checkpoint/ "
                                "zone the agent must reach that is NOT a real-world object (a spot to "
                                "stand on, a parking space, a 'cut here' point, a finish line) - rendered "
                                "as a flat glowing disc on the ground; use with `kind: box` or `circle`. "
                                "'plain': a real physical object that isn't a person/ball/goal (furniture, "
                                "a wall segment, an obstacle, a chair, a box to push) - a plain shape."
                            ),
                        },
                        "radius": {"type": "number", "description": "For kind=circle."},
                        "size": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": (
                                "For kind=box: [x_extent, y_extent], the box's footprint on the ground "
                                "in the SAME 2D plane as everything else (not a width+vertical-height "
                                "pair - there is no vertical axis in this physics model). For a goal, "
                                "make one extent small (e.g. 0.5, the goal line's thinness) and the "
                                "other the actual mouth width (e.g. 5-8) - the renderer figures out "
                                "which side the mouth faces from this shape and the entity's position."
                            ),
                        },
                        "color": {"type": "string", "description": "CSS hex color, e.g. '#4f8cff'."},
                        "mass": {"type": "number", "minimum": 0.1, "maximum": 200},
                        "static": {
                            "type": "boolean",
                            "description": "True for immovable markers/obstacles/goals.",
                        },
                        "initial_position": {
                            "description": (
                                "[x, y], or the string 'random'. The origin (0, 0) is the BOTTOM-LEFT "
                                "corner of the field, NOT the center - valid values are 0 <= x <= "
                                "field.width and 0 <= y <= field.height. Never use negative numbers or "
                                "numbers beyond field.width/field.height."
                            ),
                        },
                    },
                },
            },
            "action": {
                "type": "object",
                "required": ["controlled_entity"],
                "properties": {
                    "controlled_entity": {"type": "string", "description": "id of the entity the agent pushes around."},
                    "max_force": {"type": "number", "minimum": 1, "maximum": 100},
                },
            },
            "reward_terms": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
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

CLASSIFICATION_TASK_TOOL = {
    "name": "define_classification_task",
    "description": (
        "Define a text classification task as data: the possible labels, a "
        "description of what the input text looks like, and a seed set of "
        "labeled examples to bootstrap training. Use this for anything about "
        "deciding which category a piece of text belongs to - e.g. spam vs not "
        "spam, positive vs negative review, urgent vs not urgent, topic tagging."
    ),
    "input_schema": {
        "type": "object",
        "required": ["name", "description", "labels", "input_description", "seed_examples"],
        "properties": {
            "name": {"type": "string", "description": "Short human-readable task name."},
            "description": {"type": "string", "description": "One sentence describing the task."},
            "labels": {
                "type": "array",
                "minItems": 2,
                "maxItems": 6,
                "items": {"type": "string"},
                "description": "The possible class names, e.g. ['spam', 'not_spam'].",
            },
            "input_description": {
                "type": "string",
                "description": "What the text being classified represents, e.g. 'the transcript or description of a phone call'.",
            },
            "seed_examples": {
                "type": "array",
                "minItems": 20,
                "maxItems": 60,
                "description": (
                    "Realistic synthetic examples to bootstrap the classifier, roughly "
                    "balanced across all labels. These are a starting point, not real "
                    "data - the user can add real examples afterwards."
                ),
                "items": {
                    "type": "object",
                    "required": ["text", "label"],
                    "properties": {
                        "text": {"type": "string"},
                        "label": {"type": "string", "description": "Must be one of `labels`."},
                    },
                },
            },
        },
    },
}

TOOLS = [CONTROL_TASK_TOOL, CLASSIFICATION_TASK_TOOL]

SYSTEM_PROMPT = """You translate a plain-language task description into a TaskSpec \
by calling exactly one tool: `define_control_task` for a physical control task \
(an agent moving, pushing, reaching, chasing, avoiding or balancing something in \
a 2D space), or `define_classification_task` for a text classification task \
(deciding which category a piece of text belongs to, e.g. spam detection, \
sentiment, topic tagging, urgency, intent). Pick whichever tool actually matches \
what the user described - do not force a classification task into a physical \
metaphor or vice versa.

Rules for `define_control_task`:
- The agent always controls exactly ONE entity by pushing it with a 2D force \
  (fx, fy). Everything else in the scene is either physics-driven (a ball that \
  gets bumped) or static (a goal marker, an obstacle).
- If the description names a distinct actor that acts on an object (a player \
  kicking a ball, a robot pushing a box, someone catching something), create \
  TWO entities - the actor (which the agent controls) and the object - so the \
  actor physically has to reach and push the object, instead of the agent \
  teleporting/controlling the object directly. Only control the object \
  directly when the description has no separate actor (e.g. "a ball that \
  learns to reach a target").
- NEVER create an entity to represent the field's outer walls or boundary \
  (e.g. no "field_boundary", "wall", "border" entity covering the whole \
  field) - the sandbox already renders and physically simulates solid walls \
  around the whole field automatically. Adding one yourself creates a \
  duplicate solid block the size of the field that traps everything inside it.
- Represent goals, targets, zones and obstacles as entities (usually \
  `static: true`), each noticeably smaller than the field itself, and reward \
  reaching them with `reach_bonus` and/or shape the approach with \
  `distance_delta`.
- Always include a small `time_penalty` so the agent is encouraged to act quickly.
- Pick sensible field size, forces and reward weights so the task is learnable \
  in a few hundred thousand steps of PPO.
- Set `visual` on every entity for realistic rendering: any person/character/ \
  animal/robot-with-a-body actor gets `visual: "humanoid"`; a ball or \
  projectile gets `visual: "ball"`; a LITERAL sports goal/net/hoop gets \
  `visual: "goal"`; an abstract target/checkpoint/zone that is not a real \
  object (where to stand, where to cut, a finish line) gets `visual: \
  "marker"`; any other real object (furniture, a wall, an obstacle, \
  something to push) gets `visual: "plain"`.
- Set `field.theme` to match the setting the description actually implies - \
  most tasks are NOT a sports pitch. A hairdresser's salon, an office, a \
  warehouse, a kitchen: `"room"`. Football/basketball/a literal field or \
  court: `"pitch"`. Anything abstract with no real-world setting: `"plain"`.

Rules for `define_classification_task`:
- Generate at least 20 seed examples, balanced across labels as evenly as \
  possible, and genuinely varied in phrasing, length and style so the \
  classifier doesn't just memorize a template.
- Make the seed examples realistic for the domain the user described (e.g. for \
  spam-call detection, write short call-summary-style texts, not essay-length text).
"""
