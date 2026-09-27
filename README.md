# AI Training Playground

Plataforma **open-source** que corre 100% en local: entrena un agente de
aprendizaje por refuerzo para **cualquier tarea que definas** y observa el
resultado en un motor gráfico 3D en tiempo real, directamente en tu
navegador.

Ejemplos de lo que puedes construir con el mismo motor:
- Un agente que aprenda a meter un balón en una portería.
- Un agente que aprenda a diferenciar colores o clasificar caras.
- Cualquier tarea que puedas expresar como un entorno con estado, acciones
  y una recompensa.

## Arquitectura

```
backend/
  aiengine/
    envs/        -> contrato Environment + registro de tareas
    training/     -> motor PPO (PyTorch), agnóstico a la tarea
    server/       -> servidor FastAPI (REST + WebSocket) que sirve el frontend
  examples/
    score_goal.py -> tarea de ejemplo: marcar un gol (física con pymunk)
frontend/
  index.html, js/, css/ -> visor 3D (three.js) + panel de control, sin build step
run.py            -> arranca todo con un solo comando
```

El backend y el frontend hablan mediante un contrato genérico
(`render_state()` en cada tarea → JSON de entidades) para que el motor
gráfico nunca necesite saber nada específico de la tarea.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python run.py
```

Esto abre `http://127.0.0.1:8000` en tu navegador. Elige una tarea, pulsa
"Iniciar entrenamiento" y verás al agente aprendiendo en vivo, junto con
sus métricas (recompensa por episodio, media móvil).

## Cómo añadir tu propia tarea

1. Crea un fichero en `backend/examples/mi_tarea.py`.
2. Define una clase que herede de `aiengine.envs.base.Environment` e
   implemente `reset`, `step` y `render_state` (ver `score_goal.py` como
   plantilla).
3. Decórala con `@register_task("mi_tarea")`.
4. Impórtala en `aiengine/server/app.py::_load_builtin_tasks` (o crea tu
   propio paquete de tareas e impórtalo igual).
5. Reinicia el servidor: tu tarea aparecerá en el selector automáticamente.

El motor de entrenamiento (PPO) es completamente agnóstico a la tarea:
solo necesita `observation_space`, `action_space`, `reset` y `step`.

## Estado del proyecto

Esto es un MVP funcional pensado como base extensible, no un producto
terminado. Ideas de siguientes pasos (contribuciones bienvenidas):

- Soporte para espacios de acción discretos y observaciones tipo imagen.
- Más algoritmos (SAC, DQN) seleccionables desde la UI.
- Guardado/carga de checkpoints y reanudación de entrenamientos.
- Entrenamiento vectorizado (múltiples entornos en paralelo) para acelerar.
- Editor visual de tareas (definir entidades, física y recompensa sin código).

## Licencia

MIT. Ver [LICENSE](LICENSE).
