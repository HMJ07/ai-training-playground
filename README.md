# 🎮 AI Training Playground

**Describe en lenguaje natural una tarea de control físico simple (mover, perseguir, esquivar, alcanzar) o de clasificación de texto (spam, sentimiento, tema...), y entrena un agente para ella en vivo, en 3D, en tu navegador.**

100% open-source y **100% gratis** (sin ninguna API de pago). Corre en tu
propio ordenador, con tus propios datos.

## ⚠️ Qué es esto (y qué NO es)

Esto **no** es una IA general que "hace lo que le pidas". Es un motor
concreto con dos moldes fijos:

- 🎯 **Control físico 2D**: un agente que empuja/mueve un objeto simple
  (círculo o caja) en un plano, con una recompensa numérica por
  acercarse/alcanzar/evitar algo. Se entrena de verdad con PPO
  (aprendizaje por refuerzo) en segundos-minutos, en tu CPU.
- 🏷️ **Clasificación de texto**: separar frases en 2-6 categorías
  (spam/no-spam, positivo/negativo...). Se entrena de verdad con un
  clasificador TF-IDF + SGD local.

**Si le pides algo que no encaja en ninguno de los dos moldes** (un mundo
3D realista, personajes con movimiento humano de verdad, entender y
ejecutar instrucciones distintas por "cliente" o "pedido", cualquier cosa
que necesite razonamiento o memoria más allá de una recompensa numérica),
**la IA generadora improvisará una aproximación con círculos, cajas y
"acércate a esto"** - no porque falle al azar, sino porque no tiene otra
pieza que ofrecer. El resultado se notará forzado. Eso no es un bug que
se vaya a arreglar con un parche: es el límite real de lo que este tipo
de motor (RL simple + clasificador lineal, gratis, en tu CPU) puede
representar. Mira los ejemplos de abajo para hacerte una idea de qué
funciona bien.

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

**Control físico:**
- "Un agente que aprenda a esquivar obstáculos mientras cruza el campo"
- "Un jugador que empuje una pelota roja hacia una zona verde, evitando la zona azul"
- "Un agente que persiga un objetivo que se mueve"
- "Algo que aprenda a mantener el equilibrio en el centro del campo"

**Clasificación de texto:**
- "Que aprenda a distinguir llamadas de spam de las que no lo son"
- "Un clasificador de reseñas positivas y negativas"
- "Detectar si un email es urgente o puede esperar"
- "Clasificar mensajes de soporte por tema: facturación, técnico o cuenta"

La IA decide sola cuál de las dos familias encaja con lo que describes.
Cuanto más concreta seas sobre qué debe tocar/evitar/alcanzar (control) o
qué categorías existen y cómo distinguirlas (clasificación), mejor sale
la tarea generada. Fuera de estos dos patrones (mover algo simple / meter
frases en categorías), no esperes que el resultado tenga sentido.

---

## 🧠 Cómo funciona por dentro

Nada de "caja negra": cuando escribes una tarea, un modelo de lenguaje
(Groq o Claude, a tu elección - ambos con capa gratuita) no genera código,
genera **datos**. Tiene dos "moldes" disponibles y elige el que encaja:

**Si es control físico** → devuelve un `TaskSpec` con:
- **`entities`**: círculos o cajas en un plano 2D (posición, color, masa,
  si son estáticas como un objetivo o un obstáculo).
- **`action`**: qué entidad controla el agente, empujándola con una fuerza
  2D continua.
- **`reward_terms`**: piezas de recompensa componibles (acercarse a algo,
  alcanzar un objetivo, penalizar salirse del campo, penalizar tiempo o
  velocidad).

**Si es clasificación de texto** → devuelve un `TaskSpec` con:
- **`labels`**: las clases posibles (ej. `spam`, `not_spam`).
- **`seed_examples`**: 20-60 ejemplos sintéticos generados por la IA para
  arrancar el entrenamiento (tú puedes añadir ejemplos reales después).

Cualquiera de los dos JSON lo interpreta siempre el mismo código auditado
- `ParametricEnv` con física real vía [pymunk](https://www.pymunk.org/), o
`TextClassifierTask` con TF-IDF + un clasificador lineal de
[scikit-learn](https://scikit-learn.org/) - la IA nunca ejecuta código
arbitrario en tu máquina, solo rellena una plantilla segura.

El resultado se transmite en vivo al navegador por WebSocket, donde el
**mismo motor gráfico** en [three.js](https://threejs.org/) lo dibuja en
3D: en control físico son cuerpos con física real; en clasificación, cada
ejemplo es un punto cuya posición viene de proyectar su vector de texto a
2D (TruncatedSVD), coloreado por clase - así ves las clases separarse en
el espacio a medida que el modelo aprende.

```
backend/
  aiengine/
    envs/
      base.py       -> contrato Environment (reset/step/render_state)
      parametric.py -> motor genérico de control físico: interpreta un
                        TaskSpec con pymunk, sin importar qué tarea describa
      registry.py   -> tareas generadas en esta sesión (en memoria)
    classification/
      engine.py     -> motor de clasificación: TF-IDF + SGDClassifier
                        incremental, con proyección 2D para visualizar
    generation/
      schema.py         -> los dos esquemas de TaskSpec + prompt de sistema
      task_generator.py -> llama a Groq o Claude (tool-use/function-calling
                            forzado, el modelo elige qué esquema usar) y
                            valida/repara lo que devuelve
    training/
      ppo.py         -> motor PPO (PyTorch) propio, para tareas de control
    server/
      app.py         -> servidor FastAPI (REST + WebSocket), enruta cada
                        tarea a su motor y sirve el frontend
frontend/
  index.html, js/, css/ -> visor 3D (three.js) + panel de control, sin build step
run.py              -> arranca todo con un solo comando
setup.sh / setup.ps1 -> instalación con un solo comando
```

Ver `backend/aiengine/generation/schema.py` para los esquemas completos,
`backend/aiengine/envs/parametric.py` y `backend/aiengine/classification/engine.py`
para cómo se interpretan.

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

**`http://localhost:8000` da "Not Found", pero `http://127.0.0.1:8000` sí funciona**
Si tienes WSL o Docker Desktop instalados, es habitual que también tengan
algo escuchando en el puerto 8000 en la dirección IPv6 (`::1`), y
`localhost` a veces resuelve ahí antes que a IPv4. La app siempre escucha
en `127.0.0.1` (IPv4) - usa esa URL directamente, o deja que `run.py`
abra la pestaña por ti (ya usa la URL correcta automáticamente).

**La instalación tarda mucho / se cuelga**
PyTorch pesa varios cientos de MB; en una conexión lenta puede tardar
varios minutos. Es normal.

**"rate_limit_exceeded" al generar una tarea con Groq**
La capa gratuita de Groq tiene un límite de tokens por minuto bastante
bajo. Espera unos segundos y vuelve a intentarlo, o usa una clave de
Anthropic en su lugar.

**"El modelo aun no se ha entrenado" al probar una clasificación**
Pulsa "Iniciar entrenamiento" al menos una vez antes de usar el cuadro
"Probar el modelo entrenado".

---

## 🗺️ Estado del proyecto y roadmap

Esto es un MVP funcional pensado como base extensible, no un producto
cerrado. Contribuciones bienvenidas. Ideas de siguientes pasos:

- [ ] Guardado/carga de checkpoints y reanudación de entrenamientos.
- [ ] Entrenamiento vectorizado (múltiples entornos en paralelo) para acelerar.
- [ ] Persistir tareas generadas entre sesiones (hoy solo viven en memoria).
- [ ] Soporte de proveedores de IA locales (Ollama) para no depender de ninguna API externa.
- [ ] Ampliar el TaskSpec de control (multi-agente, más formas, sensores tipo raycast).
- [ ] Subir un CSV propio como datos iniciales para clasificación, en vez de solo ejemplos sintéticos.
- [ ] Clasificación de más de un campo de texto a la vez (ej. asunto + cuerpo de un email).
- [ ] Exportar/compartir tareas generadas como fichero JSON.

## 🤝 Contribuir

Los pull requests son bienvenidos. Si añades un tipo de término de
recompensa nuevo o una forma de entidad nueva, actualiza también
`schema.py` (el esquema que ve la IA) y `parametric.py` (cómo se
interpreta) a la vez.

## 📄 Licencia

MIT. Ver [LICENSE](LICENSE) - úsalo, modifícalo y compártelo libremente.
