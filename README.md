# AI Training Playground

Plataforma **open-source** que corre 100% en local: describe con tus
propias palabras la tarea que quieres que una IA aprenda a hacer, y la
plataforma la construye, la entrena por refuerzo y te muestra el resultado
en un motor gráfico 3D en tiempo real, directamente en tu navegador.

No hay un menú de tareas predefinidas. Escribes algo como:

> "Un agente que empuje un balón hasta una portería, evitando un obstáculo por el medio"

y la IA traduce eso en una tarea entrenable (entidades, física, recompensa)
que el motor genérico interpreta y entrena al momento.

## Arquitectura

```
backend/
  aiengine/
    envs/
      base.py       -> contrato Environment (reset/step/render_state)
      parametric.py -> motor genérico: interpreta un TaskSpec (JSON) con
                        pymunk, sin importar qué tarea describa
      registry.py   -> tareas generadas en esta sesión (en memoria)
    generation/
      schema.py         -> esquema del TaskSpec + prompt de sistema
      task_generator.py -> llama a la API de Claude (tool-use forzado) para
                            traducir texto libre a un TaskSpec validado
    training/
      ppo.py         -> motor PPO (PyTorch) propio, agnóstico a la tarea
    server/
      app.py         -> servidor FastAPI (REST + WebSocket) que sirve el frontend
frontend/
  index.html, js/, css/ -> visor 3D (three.js) + panel de control, sin build step
run.py            -> arranca todo con un solo comando
```

Nada de esto ejecuta código generado por la IA: el modelo solo devuelve
datos (un JSON con un esquema fijo — entidades, cuál controla el agente,
términos de recompensa), y `ParametricEnv` es el único código que
realmente simula la física e interpreta esos datos. Esto hace que
cualquier tarea que la IA proponga sea segura de correr en tu máquina.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
cp .env.example .env   # y pon tu ANTHROPIC_API_KEY (console.anthropic.com)
python run.py
```

Esto abre `http://127.0.0.1:8000` en tu navegador. Escribe la tarea que
quieres entrenar, pulsa "Generar tarea con IA", y cuando esté lista pulsa
"Iniciar entrenamiento" para verla aprender en vivo junto con sus métricas
(recompensa por episodio, media móvil).

## Cómo funciona el TaskSpec

Un TaskSpec describe la tarea completamente como datos:

- **`entities`**: círculos o cajas en un plano 2D (posición, color, masa,
  si son estáticas como un objetivo/obstáculo).
- **`action`**: qué entidad controla el agente, empujándola con una fuerza
  2D continua.
- **`reward_terms`**: piezas componibles (acercarse a algo, alcanzar un
  objetivo, penalizar salirse del campo, penalizar tiempo/velocidad).

Ver `backend/aiengine/generation/schema.py` para el esquema completo y
`backend/aiengine/envs/parametric.py` para cómo se interpreta.

## Estado del proyecto

MVP funcional pensado como base extensible. Ideas de siguientes pasos
(contribuciones bienvenidas):

- Guardado/carga de checkpoints y reanudación de entrenamientos.
- Entrenamiento vectorizado (múltiples entornos en paralelo) para acelerar.
- Persistir tareas generadas entre sesiones (hoy solo viven en memoria).
- Soporte de proveedores de IA locales (Ollama) como alternativa a la API.
- Ampliar el TaskSpec (multi-agente, más formas, sensores tipo raycast).

## Licencia

MIT. Ver [LICENSE](LICENSE).
