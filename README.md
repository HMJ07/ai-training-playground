# 🎮 AI Training Playground

**Escribe la tarea que quieres que una IA aprenda a hacer, en tu propio idioma. La plataforma la construye, la entrena por refuerzo, y ves el resultado en 3D en vivo, en tu navegador.**

100% open-source. Corre en tu propio ordenador, con tus propios datos.
No hay ningún menú de tareas predefinidas: tú decides qué se entrena.

```
Tú escribes:  "Un agente que empuje un balón hasta una portería,
               evitando un obstáculo por el medio"
                        │
                        ▼
        La IA construye la tarea automáticamente
                        │
                        ▼
   Se entrena un agente de refuerzo (PPO) y lo ves aprender
        en vivo, en 3D, en tu propio navegador
```

No necesitas saber nada de aprendizaje por refuerzo, física, ni redes
neuronales para usarlo. Si sabes escribir una frase, sabes usar esto.

---

## 🚀 Empezar en 3 pasos

**Requisitos:** tener [Python 3.10+](https://www.python.org/downloads/) instalado (marca "Add to PATH" durante la instalación en Windows).

### 1. Descarga el proyecto

```bash
git clone https://github.com/HMJ07/ai-training-playground.git
cd ai-training-playground
```

(o pulsa el botón verde "Code → Download ZIP" en GitHub si no usas git)

### 2. Instálalo con un solo comando

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**macOS / Linux:**
```bash
./setup.sh
```

Esto crea un entorno virtual, instala todo lo necesario (tarda unos
minutos la primera vez por PyTorch) y te prepara un fichero `.env`.

### 3. Consigue una clave de API gratis y arráncalo

Abre el fichero `.env` que se acaba de crear y pon **una** de estas dos
claves (con cualquiera de las dos funciona):

- **Groq** (recomendado, gratis y muy rápido): consíguela en [console.groq.com/keys](https://console.groq.com/keys)
- **Anthropic / Claude**: consíguela en [console.anthropic.com](https://console.anthropic.com/)

```
GROQ_API_KEY=tu_clave_aqui
```

Y arranca:

```bash
# Windows
.venv\Scripts\python run.py

# macOS / Linux
.venv/bin/python run.py
```

Se abrirá `http://127.0.0.1:8000` en tu navegador automáticamente. Escribe
tu tarea, pulsa **"Generar tarea con IA"**, y luego **"Iniciar
entrenamiento"**. Verás al agente aprendiendo en vivo en 3D, junto con sus
métricas (recompensa por episodio, media móvil).

> **¿Algo no funciona?** Mira la sección [Solución de problemas](#-solución-de-problemas) más abajo.

---

## 💡 Ejemplos de tareas que puedes escribir

- "Un agente que aprenda a esquivar obstáculos mientras cruza el campo"
- "Un jugador que empuje una pelota roja hacia una zona verde, evitando la zona azul"
- "Un agente que persiga un objetivo que se mueve"
- "Algo que aprenda a mantener el equilibrio en el centro del campo"

La IA traduce cualquier descripción razonable a una tarea física 2D
(entidades, física, recompensa). Cuanto más concreta seas sobre qué debe
tocar, evitar o alcanzar el agente, mejor sale la tarea generada.

---

## 🧠 Cómo funciona por dentro

Nada de "caja negra": cuando escribes una tarea, un modelo de lenguaje
(Groq o Claude, a tu elección) no genera código - genera **datos**, un
`TaskSpec` en JSON con un esquema fijo:

- **`entities`**: círculos o cajas en un plano 2D (posición, color, masa,
  si son estáticas como un objetivo o un obstáculo).
- **`action`**: qué entidad controla el agente, empujándola con una fuerza
  2D continua.
- **`reward_terms`**: piezas de recompensa componibles (acercarse a algo,
  alcanzar un objetivo, penalizar salirse del campo, penalizar tiempo o
  velocidad).

Ese JSON lo interpreta siempre el mismo código auditado
(`ParametricEnv`, con física real vía [pymunk](https://www.pymunk.org/)) -
la IA nunca ejecuta código arbitrario en tu máquina, solo rellena una
plantilla segura. El agente se entrena con un motor **PPO propio, escrito
en PyTorch** (no una caja negra externa), y el resultado se transmite en
vivo al navegador por WebSocket, donde un motor gráfico en
[three.js](https://threejs.org/) lo dibuja en 3D.

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
      task_generator.py -> llama a Groq o Claude (tool-use/function-calling
                            forzado) para traducir texto libre a un
                            TaskSpec validado
    training/
      ppo.py         -> motor PPO (PyTorch) propio, agnóstico a la tarea
    server/
      app.py         -> servidor FastAPI (REST + WebSocket) que sirve el frontend
frontend/
  index.html, js/, css/ -> visor 3D (three.js) + panel de control, sin build step
run.py              -> arranca todo con un solo comando
setup.sh / setup.ps1 -> instalación con un solo comando
```

Ver `backend/aiengine/generation/schema.py` para el esquema completo del
TaskSpec y `backend/aiengine/envs/parametric.py` para cómo se interpreta.

---

## 🛠️ Solución de problemas

**"ModuleNotFoundError: No module named 'torch'"**
Estás usando el Python del sistema en vez del entorno virtual del
proyecto. Usa `.venv\Scripts\python run.py` (Windows) o
`.venv/bin/python run.py` (macOS/Linux), no solo `python run.py`.

**"Falta una clave de API"**
No has puesto ninguna clave en `.env`, o el fichero no se llama
exactamente `.env` (no `.env.txt`). Revisa el paso 3.

**El puerto 8000 ya está en uso**
Cierra cualquier otro proceso que lo esté usando, o cambia `PORT` en
`run.py`.

**La instalación tarda mucho / se cuelga**
PyTorch pesa varios cientos de MB; en una conexión lenta puede tardar
varios minutos. Es normal.

**"rate_limit_exceeded" al generar una tarea con Groq**
La capa gratuita de Groq tiene un límite de tokens por minuto bastante
bajo. Espera unos segundos y vuelve a intentarlo, o usa una clave de
Anthropic en su lugar.

---

## 🗺️ Estado del proyecto y roadmap

Esto es un MVP funcional pensado como base extensible, no un producto
cerrado. Contribuciones bienvenidas. Ideas de siguientes pasos:

- [ ] Guardado/carga de checkpoints y reanudación de entrenamientos.
- [ ] Entrenamiento vectorizado (múltiples entornos en paralelo) para acelerar.
- [ ] Persistir tareas generadas entre sesiones (hoy solo viven en memoria).
- [ ] Soporte de proveedores de IA locales (Ollama) para no depender de ninguna API.
- [ ] Ampliar el TaskSpec (multi-agente, más formas, sensores tipo raycast).
- [ ] Exportar/compartir tareas generadas como fichero JSON.

## 🤝 Contribuir

Los pull requests son bienvenidos. Si añades un tipo de término de
recompensa nuevo o una forma de entidad nueva, actualiza también
`schema.py` (el esquema que ve la IA) y `parametric.py` (cómo se
interpreta) a la vez.

## 📄 Licencia

MIT. Ver [LICENSE](LICENSE) - úsalo, modifícalo y compártelo libremente.
